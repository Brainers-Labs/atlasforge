"""Environment checks behind ``atlasforge doctor``.

Each check is a small pure function with injectable dependencies, so it can be
tested without a GPU, network or real environment.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from atlasforge import config
from atlasforge.asr.models import ASR_MODELS, accept_licence_url
from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.errors import ConfigError
from atlasforge.security import mask_secret

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

Status = Literal["ok", "warn", "fail"]

#: Whether a gated repo's licence has been accepted. ``unknown`` means we could not
#: tell (no ``huggingface_hub``, no network) — never reported as success.
Access = Literal["granted", "denied", "unknown"]

#: Every model a user must accept the Awarri licence on before they can download it.
GATED_MODELS: Final[tuple[str, ...]] = (DEFAULT_MODEL, *ASR_MODELS.values())

MIN_PYTHON: Final = (3, 10)
MIN_FREE_DISK_GB: Final = 25
# Rough VRAM needs for the 8B LLM. INFERRED, not yet measured (planning/21).
FP16_VRAM_GB: Final = 16
QUANT4_VRAM_GB: Final = 6

_EXTRAS: Final[dict[str, tuple[str, ...]]] = {
    "local": ("torch", "transformers", "accelerate"),
    "asr": ("transformers", "librosa", "soundfile"),
    "finetune": ("peft", "trl", "datasets"),
}


@dataclass(frozen=True, slots=True)
class Check:
    """Outcome of one environment check."""

    name: str
    status: Status
    detail: str
    hint: str | None = None


def check_python(version: tuple[int, int, int] | None = None) -> Check:
    """Python must be new enough."""
    current = version or (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    text = ".".join(map(str, current))
    if current[:2] >= MIN_PYTHON:
        return Check("python", "ok", text)
    need = ".".join(map(str, MIN_PYTHON))
    return Check("python", "fail", f"{text} is too old", f"AtlasForge needs Python {need}+.")


def check_ffmpeg(which: Callable[[str], str | None] = shutil.which) -> Check:
    """ffmpeg converts ogg/opus/m4a/mp3 audio for ASR."""
    path = which("ffmpeg")
    if path:
        return Check("ffmpeg", "ok", path)
    return Check(
        "ffmpeg",
        "warn",
        "not found (needed for ogg/opus/m4a/mp3 audio)",
        "Windows: winget install Gyan.FFmpeg | macOS: brew install ffmpeg | Linux: apt install ffmpeg",
    )


def check_config(find: Callable[[], Path | None] = config.find_file) -> Check:
    """A project file silently changes every run, so `doctor` says which one was found."""
    try:
        path = find()
    except ConfigError as exc:
        return Check("config", "fail", str(exc), exc.hint)
    if path is None:
        return Check("config", "ok", f"no {config.FILE_NAME}; using flag, environment, defaults")
    try:
        loaded = config.load_file(path)
    except ConfigError as exc:
        return Check("config", "fail", str(exc), exc.hint)
    settings = ", ".join(f"{key}={value}" for key, value in sorted(loaded.values.items()))
    return Check("config", "ok", f"{path} ({settings or 'no settings'})")


def check_hf_token(env: Mapping[str, str], home: Path | None = None) -> Check:
    """A Hugging Face token is required to download the gated NCAIR1 models."""
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return Check("hf-token", "ok", f"found ({mask_secret(token)})")
    hf_home = env.get("HF_HOME")
    cache = Path(hf_home) if hf_home else (home or Path.home()) / ".cache" / "huggingface"
    if (cache / "token").is_file():
        return Check("hf-token", "ok", "cached login found")
    return Check(
        "hf-token",
        "warn",
        "no token found (needed for the gated NCAIR1 models)",
        "Create a token at https://huggingface.co/settings/tokens, accept the licence on each "
        "NCAIR1 model page, then set HF_TOKEN or run `huggingface-cli login`.",
    )


def classify_access_error(exc: Exception) -> Access:
    """Read a Hub exception as ``denied`` (a real refusal) or ``unknown`` (anything else).

    Only a gated/401/403 answer means "the licence is not accepted". A timeout or a
    DNS failure says nothing about the licence, so it must not be reported as denied.
    """
    if "Gated" in type(exc).__name__ or "RepositoryNotFound" in type(exc).__name__:
        return "denied"
    text = str(exc)
    return "denied" if "401" in text or "403" in text else "unknown"


def _hf_probe(model: str) -> Access:
    """Ask the Hub whether we may read ``model``. ``unknown`` when we cannot tell."""
    try:
        # Imported by name, like the other optional deps: a core install has no
        # huggingface_hub, and `doctor` must still run.
        hub = importlib.import_module("huggingface_hub")
    except ImportError:
        return "unknown"
    try:
        hub.HfApi().model_info(model)
    except Exception as exc:  # any failure means "we could not tell" - never a crash
        return classify_access_error(exc)
    return "granted"


def model_access(model: str, probe: Callable[[str], Access] = _hf_probe) -> Access:
    """Whether we may read ``model``: the probe, without the verdict ``Check`` wrapped round it.

    ``check_gated_access`` is the right shape for a table of checks; this is the right shape for a
    caller that wants the answer for one repository and will decide what it means.
    """
    return probe(model)


def check_gated_access(
    probe: Callable[[str], Access] = _hf_probe,
    *,
    models: Sequence[str] = GATED_MODELS,
) -> Check:
    """Whether the Awarri licence is accepted on every gated NCAIR1 repo.

    A token proves you are logged in; it does not prove the licence was accepted on a
    given model page, which is a separate click. This is the check that catches that.
    """
    verdicts = {model: probe(model) for model in models}
    denied = [m for m, verdict in verdicts.items() if verdict == "denied"]
    if denied:
        return Check(
            "model-access",
            "warn",
            f"licence not accepted on {', '.join(denied)}",
            "Open each model page and accept the licence while logged in: "
            + " | ".join(accept_licence_url(m) for m in denied),
        )
    if any(verdict == "unknown" for verdict in verdicts.values()):
        unknown = [m for m, verdict in verdicts.items() if verdict == "unknown"]
        return Check(
            "model-access",
            "warn",
            f"could not check {len(unknown)} of {len(models)} models",
            "Install huggingface_hub and go online to check gated access, or open "
            + accept_licence_url(unknown[0])
            + " to confirm by hand.",
        )
    return Check("model-access", "ok", f"licence accepted on all {len(models)} models")


def check_extras(
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> list[Check]:
    """Report which optional dependency groups are installed."""
    checks: list[Check] = []
    for extra, modules in _EXTRAS.items():
        missing = [m for m in modules if find_spec(m) is None]
        if not missing:
            checks.append(Check(f"extra:{extra}", "ok", "installed"))
        else:
            checks.append(
                Check(
                    f"extra:{extra}",
                    "warn",
                    f"missing {', '.join(missing)}",
                    f'pip install "brainers-atlasforge[{extra}]"',
                )
            )
    return checks


def _query_gpus(
    which: Callable[[str], str | None],
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> list[tuple[str, int]]:
    """Return ``(name, memory_mib)`` per NVIDIA GPU, or ``[]`` when none can be queried."""
    exe = which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        name, _, mem = line.rpartition(",")
        try:
            gpus.append((name.strip(), int(mem.strip())))
        except ValueError:
            continue
    return gpus


def check_gpu(
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Check:
    """Look for an NVIDIA GPU big enough for the 8B LLM."""
    gpus = _query_gpus(which, run)
    if not gpus:
        return Check(
            "gpu",
            "warn",
            "no NVIDIA GPU detected",
            "ASR models run on CPU. The 8B LLM needs a GPU or a remote OpenAI-compatible endpoint.",
        )
    detail = ", ".join(f"{name} ({mib / 1024:.0f} GB)" for name, mib in gpus)
    best_gb = max(mib for _, mib in gpus) / 1024
    if best_gb >= FP16_VRAM_GB:
        return Check("gpu", "ok", detail)
    if best_gb >= QUANT4_VRAM_GB:
        return Check(
            "gpu",
            "warn",
            detail,
            f"fp16 needs about {FP16_VRAM_GB} GB (estimate). Use 4-bit quantisation "
            f"(about {QUANT4_VRAM_GB} GB, estimate) or a remote endpoint.",
        )
    return Check("gpu", "warn", detail, "Too little VRAM for the 8B LLM. Use a remote endpoint.")


def check_disk(path: Path | None = None) -> Check:
    """Model weights need tens of GB of free disk."""
    target = path or Path.home()
    try:
        free_gb = shutil.disk_usage(target).free / 1024**3
    except OSError as exc:
        return Check("disk", "warn", f"could not read free space: {exc}")
    if free_gb >= MIN_FREE_DISK_GB:
        return Check("disk", "ok", f"{free_gb:.0f} GB free")
    return Check(
        "disk",
        "warn",
        f"{free_gb:.0f} GB free",
        f"The LLM weights are about 16 GB. Keep at least {MIN_FREE_DISK_GB} GB free.",
    )


def run_checks(
    env: Mapping[str, str] | None = None,
    probe: Callable[[str], Access] = _hf_probe,
) -> list[Check]:
    """Run every check against the real environment."""
    environment = os.environ if env is None else env
    return [
        check_python(),
        check_gpu(),
        check_disk(),
        check_ffmpeg(),
        check_config(),
        check_hf_token(environment),
        check_gated_access(probe),
        *check_extras(),
    ]


def exit_code(checks: list[Check]) -> int:
    """1 if any check failed outright, else 0. Warnings do not fail the run."""
    return 1 if any(c.status == "fail" for c in checks) else 0
