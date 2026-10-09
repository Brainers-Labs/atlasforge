"""Long-audio handling: the official ASR models take at most 30 seconds per call.

Audio is cut into overlapping windows, each window is transcribed, and the texts are
merged by removing the words duplicated across the overlap. The chunk boundaries
recorded in the result are *ours*, not model-produced timestamps.

Two splitters are available. :func:`plan_windows` places boundaries on a fixed grid; a
word cut by a boundary can be misheard, and the overlap only makes that less likely.
:func:`plan_windows_silence_aware` moves each boundary to the quietest nearby moment
instead, so the cut lands in a pause rather than inside a word -- which is what the
overlap was compensating for. It is opt-in (``silence_aware=True``), because moving a
boundary changes the transcripts and the choice belongs to a run that has been listened
to; see ``docs/guides/transcribe-speech.md``.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np

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

    import numpy.typing as npt

    from atlasforge.backends.base import Backend
    from atlasforge.types import (
        AudioInput,
        BackendInfo,
        Generation,
        GenParams,
        Lang,
        Message,
    )

MODEL_LIMIT_S: Final = 30.0
DEFAULT_WINDOW_S: Final = 28.0
DEFAULT_OVERLAP_S: Final = 2.0
MIN_OVERLAP_WORDS: Final = 2
MAX_OVERLAP_WORDS: Final = 15

#: Frame length for the energy envelope used to find pauses.
FRAME_MS: Final = 20.0
#: How far a boundary may move looking for a pause.
DEFAULT_SEARCH_S: Final = 2.0
#: A frame at or below this level (dBFS) counts as a pause.
SILENCE_DB: Final = -40.0
#: Digital silence has no logarithm; report it as this instead of ``-inf``.
_FLOOR_DB: Final = -120.0


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


def frame_levels(
    pcm: bytes, *, sample_rate: int = SAMPLE_RATE, frame_ms: float = FRAME_MS
) -> npt.NDArray[np.float32]:
    """The energy envelope: RMS level of each frame, in dBFS.

    ``0`` dB is full scale and digital silence floors at ``-120`` rather than ``-inf``,
    so the result is always a number a threshold can be compared against. A trailing
    partial frame is dropped — it is shorter than every other frame, so its level is not
    comparable, and a boundary cannot be usefully placed inside it.
    """
    if frame_ms <= 0:
        raise ConfigError("frame_ms must be > 0.")
    frame = max(1, round(frame_ms * sample_rate / 1000))
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    frames = samples.size // frame
    if frames == 0:
        return np.empty(0, dtype=np.float32)
    rms = np.sqrt(np.mean(samples[: frames * frame].reshape(frames, frame) ** 2, axis=1))
    # The level, then floored and narrowed. The intermediate is named and annotated rather than
    # written into one expression on purpose: numpy 2.2's stubs -- the newest a 3.10 runner can
    # install -- fail to match the outer ``np.maximum`` when its first argument is an
    # unannotated nested call, so mypy falls back to ``Any`` and then fails ``warn_return_any``.
    # With the annotation the overload resolves identically on numpy 2.2 and 2.5.
    level_db: npt.NDArray[np.float64] = 20.0 * np.log10(np.maximum(rms, 1e-6))
    return np.maximum(level_db, _FLOOR_DB).astype(np.float32)


def plan_windows_silence_aware(
    pcm: bytes,
    *,
    sample_rate: int = SAMPLE_RATE,
    window_s: float = DEFAULT_WINDOW_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    search_s: float = DEFAULT_SEARCH_S,
    frame_ms: float = FRAME_MS,
    silence_db: float = SILENCE_DB,
) -> list[tuple[int, int]]:
    """Like :func:`plan_windows`, but each boundary seeks the quietest nearby moment.

    Every boundary of the fixed grid is moved up to ``search_s`` seconds, in either
    direction, to the middle of the quietest frame within reach — provided that frame is
    at or below ``silence_db``, so a boundary is never moved on the strength of "less
    loud than the rest". The number of windows and their overlap are unchanged.

    A move is only taken if it keeps every window it touches inside
    ``MODEL_LIMIT_S``: moving one boundary lengthens one window and shortens its
    neighbour, since each window ends at a boundary and begins at the previous one minus
    the overlap. A boundary with no acceptable move simply stays on the grid, so a pause
    found in one place is not thrown away because of a constraint somewhere else.
    """
    if search_s < 0:
        raise ConfigError("search_s must be >= 0.")
    n_samples = len(pcm) // BYTES_PER_SAMPLE
    nominal = plan_windows(
        n_samples, sample_rate=sample_rate, window_s=window_s, overlap_s=overlap_s
    )
    if len(nominal) < 2:
        return nominal  # nothing to move: the audio fits in one call

    overlap = round(overlap_s * sample_rate)
    limit = round(MODEL_LIMIT_S * sample_rate)
    # A boundary that moved further than the slack between the nominal window and the
    # models' limit could push a window past it, so the search is capped there as well.
    slack = max(0, round((MODEL_LIMIT_S - window_s) * sample_rate))
    search = min(round(search_s * sample_rate), slack)
    levels = frame_levels(pcm, sample_rate=sample_rate, frame_ms=frame_ms)
    if search == 0 or levels.size == 0:
        return nominal

    targets = [end for _start, end in nominal[:-1]]
    frame = max(1, round(frame_ms * sample_rate / 1000))
    boundaries: list[int] = []
    for target in targets:
        # The window ending here starts at the previous boundary, or at zero for the
        # first, and a boundary that moved earlier leaves its successor less room. The
        # final window needs no such bound: the fixed grid already ends inside the limit
        # there, and the search cap above means moving its start earlier cannot push it
        # past one.
        previous = boundaries[-1] if boundaries else 0
        longest = limit if not boundaries else previous + limit - overlap
        low = max(target - search, previous + 1)
        high = min(target + search, longest)
        found = (
            None
            if low > high
            else _quietest(levels, target, low=low, high=high, frame=frame, silence_db=silence_db)
        )
        boundaries.append(target if found is None else found)

    return _windows_from(boundaries, n_samples=n_samples, overlap=overlap)


def _quietest(
    levels: npt.NDArray[np.float32],
    target: int,
    *,
    low: int,
    high: int,
    frame: int,
    silence_db: float,
) -> int | None:
    """Sample offset of the middle of the quietest frame centred in ``[low, high]``.

    ``None`` when every frame in reach is louder than ``silence_db`` — the boundary then
    stays where the fixed grid put it. Ties go to the frame nearest ``target``, so the
    result does not depend on which end of the range the scan started from.
    """
    first = max(0, -(-low // frame))  # ceil: the first frame whose middle reaches low
    last = high // frame
    if last < first or first >= levels.size:
        return None
    reachable = levels[first : min(last, levels.size - 1) + 1]
    quietest = float(reachable.min())
    if quietest > silence_db:
        return None
    centres = (first + np.flatnonzero(reachable == quietest)) * frame + frame // 2
    nearest = centres[int(np.argmin(np.abs(centres - target)))]
    return int(min(max(nearest, low), high))  # a frame's middle can lie just outside the range


def _windows_from(
    boundaries: Sequence[int], *, n_samples: int, overlap: int
) -> list[tuple[int, int]]:
    """Turn boundary offsets into the window ranges ``transcribe_long`` walks."""
    windows = [(0, boundaries[0])]
    windows.extend((previous - overlap, current) for previous, current in pairwise(boundaries))
    windows.append((boundaries[-1] - overlap, n_samples))
    return windows


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
    silence_aware: bool = False,
    search_s: float = DEFAULT_SEARCH_S,
    silence_db: float = SILENCE_DB,
    decoder: Callable[[AudioInput], bytes] = decode_audio,
) -> Transcript:
    """Transcribe audio of any length with any backend that handles up to 30 seconds.

    Decodes to 16 kHz mono, windows it, sends each window to ``backend.transcribe`` as a
    WAV, and merges the results. ``Transcript.chunks`` carries our window boundaries.

    ``silence_aware=True`` moves each window boundary to the quietest nearby moment (see
    :func:`plan_windows_silence_aware`) instead of leaving it on the fixed grid. The
    default is the fixed grid, because listening to both is the only way to choose, and
    because a run's chunk boundaries are compared across runs.
    """
    pcm = decoder(audio)
    if silence_aware:
        windows = plan_windows_silence_aware(
            pcm,
            window_s=window_s,
            overlap_s=overlap_s,
            search_s=search_s,
            silence_db=silence_db,
        )
    else:
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


class LongAudioBackend:
    """Wrap any backend so ``transcribe`` accepts audio of any length.

    Everything else is delegated unchanged. Used by ``eval`` for ASR datasets, whose clips
    can exceed the models' 30-second limit.
    """

    def __init__(
        self,
        inner: Backend,
        *,
        window_s: float = DEFAULT_WINDOW_S,
        overlap_s: float = DEFAULT_OVERLAP_S,
        silence_aware: bool = False,
        search_s: float = DEFAULT_SEARCH_S,
        silence_db: float = SILENCE_DB,
        decoder: Callable[[AudioInput], bytes] = decode_audio,
    ) -> None:
        plan_windows(1, window_s=window_s, overlap_s=overlap_s)  # validate settings up front
        if search_s < 0:
            raise ConfigError("search_s must be >= 0.")
        self._inner = inner
        self._window_s = window_s
        self._overlap_s = overlap_s
        self._silence_aware = silence_aware
        self._search_s = search_s
        self._silence_db = silence_db
        self._decoder = decoder

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        return self._inner.generate(messages, params)

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        return transcribe_long(
            self._inner,
            audio,
            lang,
            window_s=self._window_s,
            overlap_s=self._overlap_s,
            silence_aware=self._silence_aware,
            search_s=self._search_s,
            silence_db=self._silence_db,
            decoder=self._decoder,
        )

    def info(self) -> BackendInfo:
        return self._inner.info()

    def close(self) -> None:
        self._inner.close()


__all__ = [
    "DEFAULT_OVERLAP_S",
    "DEFAULT_SEARCH_S",
    "DEFAULT_WINDOW_S",
    "FRAME_MS",
    "MODEL_LIMIT_S",
    "SILENCE_DB",
    "LongAudioBackend",
    "duration_s",
    "frame_levels",
    "merge_texts",
    "plan_windows",
    "plan_windows_silence_aware",
    "transcribe_long",
]
