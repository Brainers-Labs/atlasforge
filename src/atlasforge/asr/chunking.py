"""Long-audio handling: the official ASR models take at most 30 seconds per call.

Audio is cut into overlapping windows, each window is transcribed, and the texts are
merged by removing the words duplicated across the overlap. The chunk boundaries
recorded in the result are *ours*, not model-produced timestamps.

Fixed windows are a v0.1 choice: a word cut by a boundary can be misheard. The overlap
makes that less likely, but it is a known limitation. Silence-aware splitting is future work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from atlasforge.asr.audio import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    decode_audio,
    duration_s,
    pcm_to_wav,
)
from atlasforge.errors import ConfigError
from atlasforge.eval.normalize import normalize, tone_insensitive
from atlasforge.types import Chunk, Transcript

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from atlasforge.backends.base import Backend
    from atlasforge.types import AudioInput, Lang

MODEL_LIMIT_S: Final = 30.0
DEFAULT_WINDOW_S: Final = 28.0
DEFAULT_OVERLAP_S: Final = 2.0
MIN_OVERLAP_WORDS: Final = 2
MAX_OVERLAP_WORDS: Final = 15


def plan_windows(
    n_samples: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    window_s: float = DEFAULT_WINDOW_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
) -> list[tuple[int, int]]:
    """Sample ranges ``(start, end)`` covering ``n_samples`` with the given overlap.

    Audio that fits in one window gets one range. Otherwise windows advance by
    ``window_s - overlap_s`` and the last one ends exactly at the end of the audio.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not 0 < window_s <= MODEL_LIMIT_S:
        raise ConfigError(
            f"window_s must be in (0, {MODEL_LIMIT_S:g}], got {window_s:g}.",
            hint="The official ASR models accept at most 30 seconds per call.",
        )
    if not 0 <= overlap_s < window_s:
        raise ConfigError("overlap_s must be >= 0 and smaller than window_s.")

    window = round(window_s * sample_rate)
    step = round((window_s - overlap_s) * sample_rate)
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + window, n_samples)
        windows.append((start, end))
        if end == n_samples:
            return windows
        start += step


def merge_texts(parts: Sequence[str]) -> str:
    """Join chunk transcripts, dropping words duplicated across each overlap.

    A duplicate is only removed when at least ``MIN_OVERLAP_WORDS`` consecutive words
    at the seam match (case, tone marks and punctuation ignored). Erring toward keeping
    a repeated word is deliberate: a legitimate repeat must not be deleted.
    """
    merged: list[str] = []
    for part in parts:
        words = part.split()
        if not words:
            continue
        merged.extend(words[_overlap(merged, words) :])
    return " ".join(merged)


def _overlap(left: Sequence[str], right: Sequence[str]) -> int:
    """Length of the longest suffix of ``left`` that equals a prefix of ``right``."""
    limit = min(len(left), len(right), MAX_OVERLAP_WORDS)
    tail = [_key(w) for w in left[len(left) - limit :]]
    head = [_key(w) for w in right[:limit]]
    for size in range(limit, MIN_OVERLAP_WORDS - 1, -1):
        if tail[limit - size :] == head[:size]:
            return size
    return 0


def _key(word: str) -> str:
    # Tone-insensitive: the same word is often transcribed with different tone marks on each
    # side of a chunk seam, and a missed match would leave the word duplicated.
    return normalize(word, tone_insensitive())


def transcribe_long(
    backend: Backend,
    audio: AudioInput,
    lang: Lang,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    decoder: Callable[[AudioInput], bytes] = decode_audio,
) -> Transcript:
    """Transcribe audio of any length with any backend that handles up to 30 seconds.

    Decodes to 16 kHz mono, windows it, sends each window to ``backend.transcribe`` as a
    WAV, and merges the results. ``Transcript.chunks`` carries our window boundaries.
    """
    pcm = decoder(audio)
    windows = plan_windows(len(pcm) // BYTES_PER_SAMPLE, window_s=window_s, overlap_s=overlap_s)
    chunks: list[Chunk] = []
    latency_ms = 0.0
    for start, end in windows:
        result = backend.transcribe(
            pcm_to_wav(pcm[start * BYTES_PER_SAMPLE : end * BYTES_PER_SAMPLE]), lang
        )
        latency_ms += result.latency_ms
        chunks.append(
            Chunk(start_s=start / SAMPLE_RATE, end_s=end / SAMPLE_RATE, text=result.text.strip())
        )
    return Transcript(
        text=merge_texts([c.text for c in chunks]),
        lang=lang,
        chunks=tuple(chunks),
        latency_ms=latency_ms,
    )


__all__ = [
    "DEFAULT_OVERLAP_S",
    "DEFAULT_WINDOW_S",
    "MODEL_LIMIT_S",
    "duration_s",
    "merge_texts",
    "plan_windows",
    "transcribe_long",
]
