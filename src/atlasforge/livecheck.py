"""Evidence from a machine that has the real N-ATLaS models.

``planning/21_NATLAS_DISCOVERY.md`` carries an eight-row verification log that stays empty until a
machine with the gated weights, a token and (for the LLM) a GPU has run it. ``planning/23`` calls
that empty log the submission blocker (G3). This module is the other half of it: it runs the checks
a machine can answer, records what it actually saw, and renders ``evidence/live_<date>.md``.

Three rules, all of them about not overstating.

* **Nothing is assumed.** A check that cannot run — no GPU, no token, no audio, a model that will
  not load — is written down as ``not run`` or ``failed``, with the exception's own words. A blank
  cell is more honest than a plausible one, and a fabricated pass is the one thing this file must
  never contain. The status words are literal, so a reader can grep for them.
* **A failure is a row, not a crash.** The point of a run is to learn which of the eight checks
  work, so a step that raises is caught, recorded and the next one still runs.
* **No credential reaches the file.** Every string is redacted as it is collected — the known
  environment values first, then anything token-shaped that no variable declared.

What the file does quote is the script's own fixed probe sentences and the model's answers to them.
Those are ours, not a user's: no user data, no audio and no credential is in this file.

Everything above :func:`live_steps` is pure and unit-tested, so a whole report can be assembled and
rendered with no GPU, no network and no credentials — as are the guards below it, and the arithmetic
they do on a result. What only a real machine can answer is the small number of functions marked
``# pragma: no cover``: they reach for torch, the weights or a running server, **they have never
run**, and filling in their rows is the whole point of :func:`live_steps`. A check whose probe
cannot run is recorded as ``not run`` rather than guessed at.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final, Literal

from atlasforge import __version__, doctor
from atlasforge.asr.models import ASR_MODELS, accept_licence_url
from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.security import mask_secret

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from atlasforge.doctor import Access

Status = Literal["verified", "failed", "not run"]

#: The eight checks ``planning/21`` lists, in its order and its words, so the rendered table can be
#: compared with the log it is meant to fill in.
CHECKS: Final[tuple[str, ...]] = (
    "Gated access granted (5 repos)",
    "LLM loads fp16 / 4-bit, VRAM used",
    "Chat template present / output sane in 4 languages",
    "`config.json` max_position_embeddings",
    "vLLM serve + chat completion",
    "Each ASR model transcribes a 10 s sample",
    "ASR `return_timestamps` behaviour",
    "Commit SHAs of all 5 repos",
)

#: Every repository a run must be able to reach: the LLM and the four monolingual ASR models.
REPOS: Final[tuple[str, ...]] = (DEFAULT_MODEL, *ASR_MODELS.values())

#: Credentials that could reach the file through an exception message echoing a request header.
SECRET_ENV: Final[tuple[str, ...]] = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ATLASFORGE_API_KEY")

_TOKEN_SHAPED: Final = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")

#: The script's own probe sentences, one per language. Deliberately simple and factual — the answer
#: is Abuja in all four — so a reader who does not speak the language can still judge whether the
#: output is sane, which is the whole point of the row that uses them.
PROBES: Final[dict[str, str]] = {
    "ha": "Menene babban birnin Najeriya?",
    "yo": "Kí ni olú ìlú Nàìjíríà?",
    "ig": "Gịnị bụ isi obodo Naịjirịa?",
    "en": "What is the capital of Nigeria?",
}

#: For labelling a sample, since a two-letter code reads poorly in a report.
LANG_NAMES: Final[dict[str, str]] = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "en": "Nigerian English",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def evidence_filename(now: datetime | None = None) -> str:
    """``live_2026-10-07.md`` — one file per day, so a re-run never silently replaces a record."""
    return f"live_{(now or _utcnow()).date().isoformat()}.md"


def environment_secrets(env: Mapping[str, str] | None = None) -> list[str]:
    """The credentials this process is holding, for :func:`redact` to remove."""
    source = os.environ if env is None else env
    return [value for key in SECRET_ENV if (value := source.get(key))]


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove credentials from ``text``: the ones we hold, then anything that looks like one.

    The known values are the ones that could leak through an exception echoing a request header.
    The pattern is the backstop for a token nobody declared — an ``hf_`` value long enough to be a
    real one is masked whether or not an environment variable named it.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, mask_secret(secret))
    return _TOKEN_SHAPED.sub("hf_****", text)


# ------------------------------------------------------------------------------ environment facts


def _package_version(package: str) -> str:
    """The installed version, read from metadata so a core install never imports the package."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def environment_facts(
    *,
    version: Callable[[str], str] = _package_version,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, str]:
    """The machine, not the models. Local checks only: no network, no credentials."""
    return {
        "atlasforge": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "huggingface_hub": version("huggingface_hub"),
        "ffmpeg": which("ffmpeg") or "not found",
    }


def _query_torch_gpus() -> list[tuple[str, float]]:  # pragma: no cover - needs torch and a GPU
    """``(name, total GiB)`` per visible CUDA device. Raises if torch is not installed."""
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        return []
    return [
        (
            str(torch.cuda.get_device_name(index)),
            float(torch.cuda.get_device_properties(index).total_memory) / 1024**3,
        )
        for index in range(int(torch.cuda.device_count()))
    ]


def gpu_facts(
    query: Callable[[], Sequence[tuple[str, float]]] = _query_torch_gpus,
) -> dict[str, str]:
    """Name and total memory per GPU, or an honest reason there is no number."""
    try:
        gpus = query()
    except ImportError as exc:
        return {"gpu": f"not checked ({exc})"}
    except Exception as exc:  # a driver fault is a fact about the machine, not a crash
        return {"gpu": f"unknown ({type(exc).__name__}: {exc})"}
    if not gpus:
        return {"gpu": "none detected"}
    return {"gpu": ", ".join(f"{name} ({gib:.0f} GB)" for name, gib in gpus)}


# --------------------------------------------------------------------------------- the record


@dataclass(frozen=True, slots=True)
class Sample:
    """A real input and the model's real answer to it."""

    what: str
    prompt: str
    output: str
    seconds: float


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one check found. ``status`` defaults to the honest one: a step that returned, passed."""

    result: str
    status: Status = "verified"
    sample: Sample | None = None


@dataclass(frozen=True, slots=True)
class Step:
    """One line of the log: what it is called, and how to find out."""

    check: str
    run: Callable[[], Outcome]


@dataclass(frozen=True, slots=True)
class Row:
    """A filled verification-log line."""

    check: str
    result: str
    status: Status


@dataclass(frozen=True, slots=True)
class Evidence:
    """Everything the file states, gathered before anything is rendered."""

    generated_at: str
    facts: Mapping[str, str]
    shas: Mapping[str, str]
    rows: Sequence[Row]
    samples: Sequence[Sample] = ()

    @property
    def verified(self) -> int:
        """How many rows passed. The number a reader should be able to check by counting."""
        return sum(1 for row in self.rows if row.status == "verified")


def collect(
    steps: Sequence[Step],
    *,
    shas: Mapping[str, str],
    facts: Mapping[str, str] | None = None,
    secrets: Iterable[str] = (),
    now: Callable[[], datetime] = _utcnow,
) -> Evidence:
    """Run every step and keep what each one saw.

    A step that raises becomes a ``failed`` row carrying the exception's own words: the run exists
    to find out which checks work, so one broken check must not hide the other seven.
    """
    hidden = list(secrets)
    rows: list[Row] = []
    samples: list[Sample] = []
    for step in steps:
        try:
            outcome = step.run()
        except Exception as exc:
            rows.append(Row(step.check, redact(f"{type(exc).__name__}: {exc}", hidden), "failed"))
            continue
        rows.append(Row(step.check, redact(outcome.result, hidden), outcome.status))
        if outcome.sample is not None:
            samples.append(
                Sample(
                    what=redact(outcome.sample.what, hidden),
                    prompt=redact(outcome.sample.prompt, hidden),
                    output=redact(outcome.sample.output, hidden),
                    seconds=outcome.sample.seconds,
                )
            )
    rows.append(_sha_row(shas))
    return Evidence(
        generated_at=now().isoformat(timespec="seconds"),
        facts=dict(facts if facts is not None else environment_facts()),
        shas={repo: redact(sha, hidden) for repo, sha in shas.items()},
        rows=rows,
        samples=samples,
    )


def _sha_row(shas: Mapping[str, str]) -> Row:
    """The one row that is derived rather than run: what the Hub says each repo is at."""
    missing = [repo for repo, sha in shas.items() if not sha or "unknown" in sha]
    if missing:
        return Row(CHECKS[7], f"unknown for {', '.join(missing)}", "not run")
    return Row(CHECKS[7], f"{len(shas)} repositories, all resolved", "verified")


# ------------------------------------------------------------------------------------ rendering


def render(evidence: Evidence) -> str:
    """The Markdown file.

    Already redacted — :func:`collect` strips credentials as it gathers, so there is no path from
    an environment variable to this output for this function to get wrong.
    """
    lines = [
        "# Live evidence",
        "",
        (
            f"Generated by `scripts/live_smoke.py` at {evidence.generated_at}. "
            "Every line below was observed by that run."
        ),
        "",
        (
            "**What this file is not.** Nothing here is estimated, and a check that could not run "
            "says so rather than leaving a plausible number behind. Where a row reads `not run`, "
            "that check has still never been performed on this project."
        ),
        "",
        (
            "The prompts quoted in the sample output are the script's own fixed probe sentences, "
            "not a user's data. No audio and no credential appears in this file."
        ),
        "",
        "## Machine",
        "",
        "| | |",
        "|---|---|",
        *(f"| {key} | {value} |" for key, value in evidence.facts.items()),
        "",
        "## Verification log (`planning/21_NATLAS_DISCOVERY.md`)",
        "",
        f"{evidence.verified} of {len(evidence.rows)} checks verified.",
        "",
        "| Check | Result | Status |",
        "|---|---|---|",
        *(f"| {row.check} | {row.result} | {row.status} |" for row in evidence.rows),
        "",
        "## Commit SHAs",
        "",
        "| Repository | Commit |",
        "|---|---|",
        *(f"| `{repo}` | `{sha or 'unknown'}` |" for repo, sha in sorted(evidence.shas.items())),
    ]
    if evidence.samples:
        lines += ["", "## Sample output", ""]
        for sample in evidence.samples:
            lines += [
                f"### {sample.what}",
                "",
                f"- Prompt: {sample.prompt}",
                f"- Output: {sample.output}",
                f"- Seconds: {sample.seconds:.2f}",
                "",
            ]
    lines += [
        "",
        "## Gated pages",
        "",
        "The licence is accepted per model page, while logged in:",
        "",
        *(f"- {accept_licence_url(repo)}" for repo in sorted(evidence.shas)),
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- the probes
#
# Everything above is pure and tested. What follows is split in two on purpose: the guards and the
# arithmetic are covered by the suite, and the handful of functions that genuinely need a GPU, a
# token or a running server carry `# pragma: no cover` so the coverage number keeps meaning what it
# means elsewhere. Those excluded lines have never executed — that is what this run finds out, and
# `docs/help/status.md` says so.


def _access_step(
    models: Sequence[str] = REPOS, access: Callable[[str], Access] = doctor.model_access
) -> Step:
    """Whether the Awarri licence has been accepted on every gated repository."""

    def run() -> Outcome:
        verdicts = {model: access(model) for model in models}
        denied = [m for m, verdict in verdicts.items() if verdict == "denied"]
        unknown = [m for m, verdict in verdicts.items() if verdict == "unknown"]
        if denied:
            return Outcome(f"licence not accepted on {', '.join(denied)}", "failed")
        if unknown:
            return Outcome(f"could not check {len(unknown)} of {len(verdicts)}", "not run")
        return Outcome(f"all {len(verdicts)} accessible")

    return Step(CHECKS[0], run)


def _revision(api: Any, model: str) -> str:
    """One repository's commit, or an ``unknown (...)`` note naming why it could not be read."""
    try:
        return str(api.model_info(model).sha or "")
    except Exception as exc:
        return f"unknown ({type(exc).__name__})"


def _hub_api() -> Any:  # pragma: no cover - the real Hub client
    return importlib.import_module("huggingface_hub").HfApi()


def hub_revisions(
    models: Sequence[str] = REPOS, api: Callable[[], Any] = _hub_api
) -> dict[str, str]:
    """The commit SHA each repository currently resolves to, or an ``unknown (...)`` note.

    Recorded because a model name is not a fixed thing: the weights behind ``NCAIR1/N-ATLaS`` can be
    replaced under the same name, so a result is only reproducible next to the SHA it ran against.
    """
    client = api()
    return {model: _revision(client, model) for model in models}


def _vram_used(torch: Any) -> float | None:
    """Peak VRAM in GiB, or ``None`` on a CPU-only machine so the row says nothing rather than 0."""
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return float(cuda.max_memory_allocated()) / 1024**3


def _backend(quantize: str, device: str, model: str = DEFAULT_MODEL) -> Any:  # pragma: no cover
    from atlasforge.backends.factory import build_backend  # noqa: PLC0415 - live path only

    return build_backend("local", model=model, quantize=quantize, device=device)


def _load_step(model: str = DEFAULT_MODEL, quantize: str = "none", device: str = "auto") -> Step:
    """Load the LLM and report what it cost, rather than what the model card estimates."""
    return Step(CHECKS[1], lambda: _probe_load(model, quantize, device))


def _probe_load(model: str, quantize: str, device: str) -> Outcome:  # pragma: no cover - needs GPU
    torch = importlib.import_module("torch")
    before = _vram_used(torch)
    backend = _backend(quantize, device, model)
    started = time.monotonic()
    info = backend.info()
    seconds = time.monotonic() - started
    after = _vram_used(torch)
    backend.close()
    where, dtype = info.device or device, info.dtype or "unreported"
    if before is None or after is None:
        return Outcome(f"loaded {quantize} on {where}, dtype {dtype}, in {seconds:.0f}s")
    card = 16 if quantize == "none" else 6
    return Outcome(
        f"loaded {quantize} on {where}, dtype {dtype}, in {seconds:.0f}s; "
        f"{after - before:.1f} GB VRAM used (the model card estimates {card} GB)"
    )


def _chat_step(model: str = DEFAULT_MODEL, quantize: str = "none", device: str = "auto") -> Step:
    """One generation per language, and whether the tokenizer ships a chat template."""
    return Step(CHECKS[2], lambda: _probe_chat(model, quantize, device))


def _probe_chat(model: str, quantize: str, device: str) -> Outcome:  # pragma: no cover - needs GPU
    transformers = importlib.import_module("transformers")
    template = getattr(transformers.AutoTokenizer.from_pretrained(model), "chat_template", None)
    backend = _backend(quantize, device, model)
    try:
        answers = []
        for lang, prompt in PROBES.items():
            started = time.monotonic()
            text = backend.generate([{"role": "user", "content": prompt}]).text
            answers.append(
                Sample(f"{LANG_NAMES[lang]} ({lang})", prompt, text, time.monotonic() - started)
            )
    finally:
        backend.close()
    empty = [sample.what.split(" ")[0] for sample in answers if not sample.output.strip()]
    summary = (
        f"chat template {'present' if template else 'ABSENT'}; {len(answers)} languages answered"
        + (f"; empty output for {', '.join(empty)}" if empty else "")
    )
    status: Status = "failed" if empty or not template else "verified"
    return Outcome(summary, status, sample=answers[0] if answers else None)


def _context_step(model: str = DEFAULT_MODEL) -> Step:
    """The real context length, read from the weights' own config rather than the card's prose."""
    return Step(CHECKS[3], lambda: _probe_context(model))


def _probe_context(model: str) -> Outcome:  # pragma: no cover - needs the weights
    transformers = importlib.import_module("transformers")
    window = getattr(
        transformers.AutoConfig.from_pretrained(model), "max_position_embeddings", None
    )
    if window is None:
        return Outcome("max_position_embeddings is not in the config", "not run")
    return Outcome(f"{window} (the model card says 8,092)")


def _server_step(base_url: str | None, model: str = DEFAULT_MODEL) -> Step:
    """The one check that needs a *server*, so it is skipped unless one was named."""

    def run() -> Outcome:
        if not base_url:
            return Outcome("no --base-url given; see docs/guides/serve-n-atlas.md", "not run")
        return _probe_server(base_url, model)

    return Step(CHECKS[4], run)


def _probe_server(base_url: str, model: str) -> Outcome:  # pragma: no cover - needs a server
    from atlasforge.backends.factory import build_backend  # noqa: PLC0415 - live path only

    backend = build_backend("openai", base_url=base_url, model=model)
    prompt = PROBES["en"]
    try:
        started = time.monotonic()
        text = backend.generate([{"role": "user", "content": prompt}]).text
        seconds = time.monotonic() - started
    finally:
        backend.close()
    if not text.strip():
        return Outcome(f"served {model} but returned nothing", "failed")
    return Outcome(
        f"served {model} answered in {seconds:.2f}s",
        sample=Sample(f"server {base_url}", prompt, text, seconds),
    )


def _audio_for(audio_dir: Path, lang: str) -> Path | None:
    """The clip for ``lang``, if the operator supplied a directory containing one."""
    for suffix in (".wav", ".ogg", ".m4a", ".mp3", ".flac"):
        candidate = audio_dir / f"{lang}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _asr_step(audio_dir: Path | None) -> Step:
    """Transcribe one real clip per language. Without clips this row cannot be filled honestly."""

    def run() -> Outcome:
        if audio_dir is None or not audio_dir.is_dir():
            return Outcome("no --audio-dir given, so no audio was transcribed", "not run")
        return _probe_asr(audio_dir)

    return Step(CHECKS[5], run)


def _probe_asr(audio_dir: Path) -> Outcome:  # pragma: no cover - needs the weights and audio
    transformers = importlib.import_module("transformers")
    pieces = []
    first: Sample | None = None
    for lang, repo in ASR_MODELS.items():
        clip = _audio_for(audio_dir, lang)
        if clip is None:
            pieces.append(f"{lang}: no clip")
            continue
        started = time.monotonic()
        pipe = transformers.pipeline("automatic-speech-recognition", model=repo)
        text = str(pipe(str(clip))["text"])
        seconds = time.monotonic() - started
        pieces.append(f"{lang}: {len(text.split())} words in {seconds:.1f}s")
        if first is None:
            first = Sample(f"asr {lang} ({repo})", clip.name, text, seconds)
    missing = [piece for piece in pieces if piece.endswith("no clip")]
    status: Status = "failed" if missing or not pieces else "verified"
    return Outcome("; ".join(pieces) or "nothing ran", status, sample=first)


def _timestamps_step(audio_dir: Path | None) -> Step:
    """Whether ``return_timestamps`` works — an UNKNOWN in ``planning/21`` for every ASR model."""

    def run() -> Outcome:
        if audio_dir is None or not audio_dir.is_dir():
            return Outcome("no --audio-dir given", "not run")
        clip = next((c for lang in ASR_MODELS if (c := _audio_for(audio_dir, lang))), None)
        if clip is None:
            return Outcome("no clip found in --audio-dir", "not run")
        return _probe_timestamps(clip)

    return Step(CHECKS[6], run)


def _probe_timestamps(clip: Path) -> Outcome:  # pragma: no cover - needs the weights and audio
    transformers = importlib.import_module("transformers")
    pipe = transformers.pipeline(
        "automatic-speech-recognition", model=ASR_MODELS["ha"], return_timestamps=True
    )
    result = pipe(str(clip))
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if not chunks:
        return Outcome("accepted return_timestamps but returned no chunks", "failed")
    spans = ", ".join(
        f"{chunk.get('timestamp', (None, None))[0]}-{chunk.get('timestamp', (None, None))[1]}"
        for chunk in chunks[:4]
    )
    return Outcome(f"{len(chunks)} timestamped chunks, first: {spans}")


def live_steps(
    *,
    model: str = DEFAULT_MODEL,
    quantize: str = "none",
    device: str = "auto",
    base_url: str | None = None,
    audio_dir: Path | None = None,
    models: Sequence[str] = REPOS,
) -> list[Step]:
    """The seven checks that have to be run, in the order ``planning/21`` lists them.

    The eighth row, the commit SHAs, is derived from :func:`hub_revisions` rather than run.
    """
    return [
        _access_step(models),
        _load_step(model, quantize, device),
        _chat_step(model, quantize, device),
        _context_step(model),
        _server_step(base_url, model),
        _asr_step(audio_dir),
        _timestamps_step(audio_dir),
    ]
