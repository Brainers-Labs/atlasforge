"""Shared, framework-free data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypedDict, get_args

from atlasforge.errors import ConfigError

if TYPE_CHECKING:
    import os

Lang: TypeAlias = Literal["ha", "yo", "ig", "en"]
"""Languages covered by N-ATLaS: Hausa, Yoruba, Igbo, Nigerian-accented English."""

LANGS: Final[tuple[Lang, ...]] = get_args(Lang)

_LANG_ALIASES: Final[dict[str, Lang]] = {
    "hausa": "ha",
    "yoruba": "yo",
    "igbo": "ig",
    "english": "en",
    "eng": "en",
}


def parse_lang(value: str) -> Lang:
    """Return the canonical language code for ``value`` (``"ha"``, ``"Yoruba"``, ...)."""
    key = value.strip().lower()
    key = _LANG_ALIASES.get(key, key)
    if key in LANGS:
        return key
    raise ConfigError(
        f"Unsupported language {value!r}.",
        hint=f"Use one of: {', '.join(LANGS)} (or hausa, yoruba, igbo, english).",
    )


Role: TypeAlias = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """One chat turn."""

    role: Role
    content: str


AudioInput: TypeAlias = "str | os.PathLike[str] | bytes"
"""A path to an audio file, or the raw bytes of one."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GenParams:
    """Generation settings.

    Defaults follow the N-ATLaS model card (VERIFIED, see planning/21):
    ``temperature=0.1``, ``repetition_penalty=1.12``, ``max_new_tokens=1000``.
    """

    temperature: float = 0.1
    repetition_penalty: float = 1.12
    max_new_tokens: int = 1000
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ConfigError("temperature must be >= 0.")
        if self.repetition_penalty <= 0:
            raise ConfigError("repetition_penalty must be > 0.")
        if self.max_new_tokens < 1:
            raise ConfigError("max_new_tokens must be >= 1.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ConfigError("top_p must be in (0, 1].")


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Token accounting, when the backend reports it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Generation:
    """Result of a text generation call."""

    text: str
    latency_ms: float
    usage: Usage | None = None
    finish_reason: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    """A transcribed span.

    ``start_s`` / ``end_s`` are the boundaries of *our* audio chunking, not
    model-produced word timestamps.
    """

    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Transcript:
    """Result of an ASR call."""

    text: str
    lang: Lang
    chunks: tuple[Chunk, ...] = ()
    latency_ms: float = 0.0


Capability: TypeAlias = Literal["generate", "transcribe"]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendInfo:
    """What a backend is actually running. Recorded in every evaluation report."""

    backend: str
    model: str
    revision: str | None = None
    device: str | None = None
    dtype: str | None = None
    capabilities: frozenset[Capability] = frozenset()
