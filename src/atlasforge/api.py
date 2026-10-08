"""The library API the CLI is a thin wrapper over.

:func:`evaluate` and :func:`compare_runs` cover the whole workflow. A run that is already
finished can be re-scored without a model — :func:`score_finished_run` returns the report,
:func:`write_reports` and :func:`write_comparison` write it out.

Importing this module costs nothing: only the standard library is imported at module
level, and every AtlasForge import happens inside the function that needs it. That keeps
``import atlasforge`` free of any machine-learning framework.

**Why not ``compare``.** The name ``atlasforge.compare`` already belongs to the comparison
subpackage, and Python would let the two overwrite each other depending on import order.
The entry point is therefore :func:`compare_runs`, which is the same call as the lower-level
:func:`atlasforge.compare.compare_runs` plus path loading and report writing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from atlasforge.backends.base import Backend
    from atlasforge.compare.compare import ComparisonReport
    from atlasforge.eval.custom import MetricFn
    from atlasforge.eval.dataset import Dataset, Task
    from atlasforge.eval.runner import Record, RunConfig, RunSummary
    from atlasforge.eval.score import MetricSummary, ScoreReport

REPORT_JSON = "report.json"
REPORT_MD = "report.md"
REPORT_HTML = "report.html"
COMPARISON_JSON = "comparison.json"
COMPARISON_MD = "comparison.md"
COMPARISON_HTML = "comparison.html"


@dataclass(frozen=True, slots=True)
class Evaluation:
    """What one :func:`evaluate` call produced: the run, its scores and where they are."""

    dataset: Dataset
    run: RunSummary
    scores: ScoreReport
    out_dir: Path

    def metric(self, key: str) -> MetricSummary:
        """One metric's summary — the same numbers that went into ``report.json``."""
        return self.scores.metric(key)


def evaluate(
    dataset: str | Path | Dataset,
    *,
    out_dir: str | Path,
    task: Task = "generation",
    backend: Backend | str = "openai",
    metrics: Sequence[str | MetricFn] | None = None,
    config: RunConfig | None = None,
    on_result: Callable[[Record], None] | None = None,
    write_report: bool = True,
    silence_aware: bool = False,
    **backend_options: Any,
) -> Evaluation:
    """Run ``dataset`` through ``backend``, score it and return the :class:`Evaluation`.

    ``out_dir`` is resumable: examples already finished there are skipped, so an
    interrupted run can simply be run again.

    The backend is either a name (``"openai"``, ``"local"``) plus options, which AtlasForge
    creates and closes for you, or a ready-made :class:`~atlasforge.backends.base.Backend`
    that you own and close yourself. Passing ``task`` is only necessary when ``dataset`` is
    a path; a :class:`~atlasforge.eval.dataset.Dataset` already knows its own task.

    Speech datasets are wrapped with the long-audio splitter automatically, so an example
    may be longer than the 30 s the ASR models take in one call. ``silence_aware=True``
    moves each split to the nearest pause rather than a fixed grid; see
    :func:`atlasforge.asr.plan_windows_silence_aware`.
    """
    from atlasforge.asr.chunking import LongAudioBackend
    from atlasforge.backends.factory import build_backend
    from atlasforge.eval.runner import RunConfig as _RunConfig
    from atlasforge.eval.runner import run as run_dataset

    data = _as_dataset(dataset, task)
    engine: Backend
    if isinstance(backend, str):
        engine = build_backend(backend, **backend_options)
        owned = True  # we created it, so we close it
    else:
        engine = backend
        owned = False  # the caller's connection outlives this call
    if data.task == "asr":
        engine = LongAudioBackend(engine, silence_aware=silence_aware)
    out = Path(out_dir)
    try:
        summary = run_dataset(engine, data, out, config=config or _RunConfig(), on_result=on_result)
    finally:
        if owned:
            engine.close()
    scores = (
        write_reports(out, data, metrics=metrics)
        if write_report
        else score_finished_run(out, data, metrics=metrics)
    )
    return Evaluation(dataset=data, run=summary, scores=scores, out_dir=out)


def compare_runs(
    dataset: str | Path | Dataset,
    base: str | Path,
    candidate: str | Path,
    *,
    task: Task = "generation",
    metrics: Sequence[str | MetricFn] | None = None,
    slice_fields: Sequence[str] | None = None,
    primary: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    min_slice_n: int | None = None,
    out_dir: str | Path | None = None,
) -> ComparisonReport:
    """Score two finished runs on the same dataset and compare them.

    This is :func:`atlasforge.compare.compare_runs` with two conveniences: the dataset may
    be a path (then ``task`` says how to read it), and ``out_dir`` writes ``comparison.md``
    and ``comparison.json`` in the same call. No model is needed for any of it.

    Both runs must have been made from exactly this dataset (checked by hash), or the
    pairing of examples would be meaningless.
    """
    from atlasforge.compare.compare import compare_runs
    from atlasforge.compare.slices import BUILTIN_FIELDS, MIN_SLICE_N

    report = compare_runs(
        _as_dataset(dataset, task),
        base,
        candidate,
        metrics=metrics,
        slice_fields=BUILTIN_FIELDS if slice_fields is None else slice_fields,
        primary=primary,
        n_boot=n_boot,
        seed=seed,
        min_slice_n=MIN_SLICE_N if min_slice_n is None else min_slice_n,
    )
    if out_dir is not None:
        write_comparison(report, out_dir)
    return report


def score_finished_run(
    run_dir: str | Path,
    dataset: str | Path | Dataset,
    task: Task = "generation",
    metrics: Sequence[str | MetricFn] | None = None,
) -> ScoreReport:
    """Score the results already in ``run_dir``. No model, no network."""
    from atlasforge.eval.runner import RESULTS_NAME, read_results
    from atlasforge.eval.score import score_run

    out = Path(run_dir)
    return score_run(_as_dataset(dataset, task), read_results(out / RESULTS_NAME), metrics=metrics)


def write_reports(
    run_dir: str | Path,
    dataset: str | Path | Dataset,
    task: Task = "generation",
    metrics: Sequence[str | MetricFn] | None = None,
) -> ScoreReport:
    """Score a finished run and write ``report.md``, ``report.json`` and ``report.html``.

    The three carry the same numbers: JSON for a script, Markdown for pasting, HTML for reading
    in a browser. The HTML is self-contained, so it needs no server and no network.
    """
    from atlasforge.eval.report import to_markdown
    from atlasforge.eval.runner import read_manifest
    from atlasforge.eval.score import write_report_json
    from atlasforge.html import report_html

    out = Path(run_dir)
    report = score_finished_run(out, dataset, task, metrics)
    manifest = read_manifest(out)
    write_report_json(report, out / REPORT_JSON)
    (out / REPORT_MD).write_text(to_markdown(report, manifest), encoding="utf-8")
    (out / REPORT_HTML).write_text(report_html(report, manifest), encoding="utf-8")
    return report


def write_comparison(report: ComparisonReport, out_dir: str | Path) -> Path:
    """Write ``comparison.md``, ``comparison.json`` and ``comparison.html`` into ``out_dir``."""
    from atlasforge.compare.report import to_markdown
    from atlasforge.html import comparison_html

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / COMPARISON_MD).write_text(to_markdown(report), encoding="utf-8")
    (out / COMPARISON_HTML).write_text(comparison_html(report), encoding="utf-8")
    (out / COMPARISON_JSON).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def _as_dataset(dataset: str | Path | Dataset, task: Task) -> Dataset:
    """Accept either a loaded dataset or a path to load one from."""
    from atlasforge.eval.dataset import Dataset as _Dataset
    from atlasforge.eval.dataset import load_dataset

    return dataset if isinstance(dataset, _Dataset) else load_dataset(dataset, task)
