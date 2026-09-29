"""Resumable evaluation runner.

Produces predictions only; scoring is a separate step so a finished run can be
re-scored with different metrics or normalisation without touching the model.

Output directory layout::

    run.json        manifest: what was run (dataset hash, model, revision, parameters)
    results.jsonl   one record per example, appended as each finishes

Re-running into the same directory skips finished examples. The manifest is
checked first, so results from a different dataset, model or parameter set are
never silently mixed. Records store errors as ``Type: message`` for AtlasForge
errors and only the exception *type* for anything unexpected, so prompt text can
never leak into results through an exception message.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

from atlasforge import __version__
from atlasforge.errors import AtlasForgeError, ConfigError, DatasetError
from atlasforge.types import GenParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from atlasforge.backends.base import Backend
    from atlasforge.eval.dataset import Dataset, Example
    from atlasforge.types import Lang

MANIFEST_NAME: Final = "run.json"
RESULTS_NAME: Final = "results.jsonl"
SCHEMA_VERSION: Final = 1
_IDENTITY_KEYS: Final = ("task", "dataset_sha256", "model", "revision", "gen_params", "lang")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """Run settings. ``concurrency > 1`` suits HTTP backends; keep 1 for local GPU weights."""

    gen_params: GenParams = field(default_factory=GenParams)
    lang: Lang | None = None
    concurrency: int = 1
    retry_errors: bool = True

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ConfigError("concurrency must be >= 1.")


@dataclass(frozen=True, slots=True, kw_only=True)
class Record:
    """Outcome for one example. Exactly one of ``prediction`` / ``error`` is set."""

    id: str
    prediction: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSummary:
    """What a call to :func:`run` did."""

    out_dir: Path
    total: int
    skipped: int  # already finished in an earlier run
    ok: int
    failed: int


def run(
    backend: Backend,
    dataset: Dataset,
    out_dir: str | Path,
    *,
    config: RunConfig | None = None,
    on_result: Callable[[Record], None] | None = None,
) -> RunSummary:
    """Run ``dataset`` through ``backend``, appending to ``out_dir/results.jsonl``."""
    cfg = config or RunConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    langs = _resolve_langs(dataset, cfg)

    manifest = _manifest(backend, dataset, cfg)
    manifest_path = out / MANIFEST_NAME
    if manifest_path.exists():
        _check_resume(manifest_path, manifest)
    else:
        _write_json_atomic(manifest_path, manifest)

    results_path = out / RESULTS_NAME
    _drop_partial_tail(results_path)
    existing = read_results(results_path)
    finished = {i for i, r in existing.items() if r.ok or not cfg.retry_errors}
    pending = [e for e in dataset.examples if e.id not in finished]

    with _open_append(results_path) as handle:

        def commit(record: Record) -> None:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()
            if on_result is not None:
                on_result(record)

        if cfg.concurrency == 1:
            for example in pending:
                commit(_execute(backend, example, cfg, langs.get(example.id)))
        else:
            _run_threaded(backend, pending, cfg, langs, commit)

    final = read_results(results_path)
    wanted = [final[e.id] for e in dataset.examples if e.id in final]
    return RunSummary(
        out_dir=out,
        total=len(dataset),
        skipped=len(dataset) - len(pending),
        ok=sum(r.ok for r in wanted),
        failed=sum(not r.ok for r in wanted),
    )


def read_results(path: str | Path) -> dict[str, Record]:
    """Load results keyed by id; the latest record for an id wins.

    A truncated final line (crash mid-write) is ignored. A corrupt line anywhere
    else raises, because silently skipping it would hide data loss.
    """
    file = Path(path)
    if not file.exists():
        return {}
    numbered = enumerate(file.read_text(encoding="utf-8").split("\n"), start=1)
    lines = [(number, text) for number, text in numbered if text.strip()]
    records: dict[str, Record] = {}
    for index, (number, text) in enumerate(lines):
        try:
            record = Record(**json.loads(text))
        except (json.JSONDecodeError, TypeError) as exc:
            if index == len(lines) - 1:
                continue
            raise DatasetError(f"corrupt results file {file}", line=number) from exc
        records[record.id] = record
    return records


def _execute(backend: Backend, example: Example, cfg: RunConfig, lang: Lang | None) -> Record:
    try:
        if lang is not None:
            if example.audio is None:
                return Record(id=example.id, error="InternalError: ASR example has no audio")
            transcript = backend.transcribe(example.audio, lang)
            return Record(
                id=example.id, prediction=transcript.text, latency_ms=transcript.latency_ms
            )
        generation = backend.generate(example.messages, cfg.gen_params)
        return Record(
            id=example.id,
            prediction=generation.text,
            latency_ms=generation.latency_ms,
            request_id=generation.request_id,
        )
    except AtlasForgeError as exc:
        return Record(id=example.id, error=f"{type(exc).__name__}: {exc.message}")
    except Exception as exc:
        return Record(id=example.id, error=f"UnexpectedError: {type(exc).__name__}")


def _run_threaded(
    backend: Backend,
    pending: list[Example],
    cfg: RunConfig,
    langs: dict[str, Lang],
    commit: Callable[[Record], None],
) -> None:
    pool = ThreadPoolExecutor(max_workers=cfg.concurrency)
    try:
        futures = [pool.submit(_execute, backend, e, cfg, langs.get(e.id)) for e in pending]
        for future in as_completed(futures):
            commit(future.result())
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def _resolve_langs(dataset: Dataset, cfg: RunConfig) -> dict[str, Lang]:
    """For ASR, every example needs a language up front (its own, or the run default)."""
    if dataset.task != "asr":
        return {}
    resolved: dict[str, Lang] = {}
    for example in dataset.examples:
        lang = example.lang or cfg.lang
        if lang is None:
            raise ConfigError(
                f"Example {example.id!r} has no language for ASR.",
                hint="Add a 'lang' field per example or set a default language for the run.",
            )
        resolved[example.id] = lang
    return resolved


def _manifest(backend: Backend, dataset: Dataset, cfg: RunConfig) -> dict[str, Any]:
    info = backend.info()
    return {
        "schema_version": SCHEMA_VERSION,
        "atlasforge_version": __version__,
        "task": dataset.task,
        "dataset_sha256": dataset.sha256,
        "n_examples": len(dataset),
        "backend": info.backend,
        "model": info.model,
        "revision": info.revision,
        "device": info.device,
        "dtype": info.dtype,
        "capabilities": sorted(info.capabilities),
        "gen_params": asdict(cfg.gen_params),
        "lang": cfg.lang,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _check_resume(path: Path, new: dict[str, Any]) -> None:
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read existing manifest {path}: {exc}") from exc
    changed = [k for k in _IDENTITY_KEYS if old.get(k) != new.get(k)]
    if changed:
        raise ConfigError(
            f"{path.parent} holds a different run (changed: {', '.join(changed)}).",
            hint="Use a new output directory, or delete this one to start over.",
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _drop_partial_tail(path: Path) -> None:
    """Remove a half-written last line (crash mid-write). Its example simply runs again."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    with path.open("r+b") as raw:
        raw.truncate(data.rfind(b"\n") + 1)


def _open_append(path: Path) -> TextIO:
    return path.open("a", encoding="utf-8", newline="\n")
