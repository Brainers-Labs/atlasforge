"""Audio decoding: any ffmpeg-readable input to 16 kHz mono 16-bit PCM.

WhatsApp voice notes (``.ogg``/opus), ``.m4a``, ``.mp3`` and ``.wav`` all go through
ffmpeg, so the models always see the same format. ffmpeg is a system tool, not a
Python dependency; ``atlasforge doctor`` checks for it.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from atlasforge.errors import AudioError, ResourceError

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from atlasforge.types import AudioInput

SAMPLE_RATE: Final = 16_000
BYTES_PER_SAMPLE: Final = 2
_STDERR_CHARS: Final = 300

_INSTALL_HINT: Final = (
    "Windows: winget install Gyan.FFmpeg | macOS: brew install ffmpeg | Linux: apt install ffmpeg"
)


def decode_audio(
    audio: AudioInput,
    *,
    timeout: float = 300.0,
    which: Callable[[str], str | None] = shutil.which,
) -> bytes:
    """Decode ``audio`` (a path, or the raw bytes of a file) to 16 kHz mono s16le PCM."""
    exe = which("ffmpeg")
    if exe is None:
        raise ResourceError("ffmpeg was not found on PATH.", hint=_INSTALL_HINT)

    if isinstance(audio, bytes):
        with tempfile.TemporaryDirectory() as directory:
            # A real file, not a pipe: containers such as m4a/mp4 need a seekable input.
            path = Path(directory) / "input"
            path.write_bytes(audio)
            return _run_ffmpeg(exe, path, timeout)

    path = Path(audio)
    if not path.is_file():
        raise AudioError(f"Audio file not found: {path}", hint="Check the path.")
    return _run_ffmpeg(exe, path, timeout)


def _run_ffmpeg(exe: str, path: Path, timeout: float) -> bytes:
    command = [
        exe, "-nostdin", "-v", "error", "-i", str(path),
        "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
    ]  # fmt: skip
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, executable resolved by shutil.which
            command, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError(
            f"ffmpeg timed out after {timeout:g}s.", hint="Try a shorter file."
        ) from exc
    except OSError as exc:
        raise ResourceError(
            f"Could not run ffmpeg: {exc.strerror or exc}", hint=_INSTALL_HINT
        ) from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()[:_STDERR_CHARS]
        raise AudioError(
            f"ffmpeg could not decode the audio: {detail or 'unknown error'}",
            hint="Is it a valid audio file? Try converting it to wav yourself to check.",
        )
    if not proc.stdout:
        raise AudioError("The audio decoded to zero samples.", hint="Is the file empty or silent?")
    return proc.stdout


def duration_s(pcm: bytes) -> float:
    """Length in seconds of 16 kHz mono s16le PCM."""
    return len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap PCM in a WAV container (what most transcription endpoints accept)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def pcm_to_float32(pcm: bytes) -> npt.NDArray[np.float32]:
    """PCM to float32 samples in [-1, 1), the input format of Whisper feature extractors."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
