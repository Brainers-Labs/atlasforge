"""Text normalisation for Hausa, Yoruba, Igbo and Nigerian English.

Why this exists: the same Yoruba word can arrive as precomposed or combining
Unicode, with or without tone marks, and naive string comparison scores those as
different. Every text metric should therefore be reported under *both* views:

* **tone-aware**: tone marks kept (only Unicode form and case/punctuation unified)
* **tone-insensitive**: tone marks removed

Only *tone* marks are removed. Letters that merely look like accented forms are
kept: Yoruba/Igbo dot-below (e with underdot, o with underdot, s with underdot),
Igbo n with dot-above, and Hausa hooked letters (implosive b/d, ejective k, y-hook).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

from atlasforge.errors import ConfigError
from atlasforge.types import LANGS, Lang

Tones = Literal["keep", "strip"]
Punctuation = Literal["strip", "keep"]

# grave, acute, circumflex, macron, caron: the tone marks used across ha/yo/ig.
_TONE_MARKS: Final = re.compile("[̀́̂̄̌]")

# Typographic apostrophes and look-alikes, unified to ASCII '.
_APOSTROPHES: Final = str.maketrans(
    dict.fromkeys("‘’‛ʼʻʹ`´", "'")  # noqa: RUF001 - the look-alike characters are the point
)

# An apostrophe with no letter/digit on either side is noise in every language.
_ORPHAN_APOSTROPHE: Final = re.compile(r"(?<!\w)'(?!\w)")
# An apostrophe not between two word characters (a quote mark). Hausa keeps these:
# a leading glottal apostrophe is part of the word (e.g. 'yar).
_EDGE_APOSTROPHE: Final = re.compile(r"(?<!\w)'|'(?!\w)")

_APOSTROPHE_CODEPOINT: Final = 0x27


class _PunctuationTable(dict[int, int | str]):
    """Lazily built ``str.translate`` table: punctuation and symbols become a space.

    Characters are classified on first sight and cached, so translating stays a
    C-speed dictionary lookup per character after warm-up.
    """

    def __missing__(self, codepoint: int) -> int | str:
        is_noise = unicodedata.category(chr(codepoint))[0] in "PS"
        result: int | str = " " if is_noise and codepoint != _APOSTROPHE_CODEPOINT else codepoint
        self[codepoint] = result
        return result


_PUNCTUATION: Final = _PunctuationTable()


@dataclass(frozen=True, slots=True)
class NormalizeConfig:
    """How text is normalised before scoring.

    ``lang`` currently only changes apostrophe handling (Hausa keeps edge
    apostrophes); it is validated so future per-language rules cannot be
    silently mistyped. The config is written into every report.
    """

    lang: Lang | None = None
    tones: Tones = "keep"
    lowercase: bool = True
    punctuation: Punctuation = "strip"

    def __post_init__(self) -> None:
        if self.lang is not None and self.lang not in LANGS:
            raise ConfigError(f"Unsupported language {self.lang!r}.", hint=f"Use one of: {LANGS}.")
        if self.tones not in ("keep", "strip"):
            raise ConfigError(f"tones must be 'keep' or 'strip', got {self.tones!r}.")
        if self.punctuation not in ("keep", "strip"):
            raise ConfigError(f"punctuation must be 'keep' or 'strip', got {self.punctuation!r}.")


def tone_aware(lang: Lang | None = None) -> NormalizeConfig:
    """Config for the tone-aware view."""
    return NormalizeConfig(lang=lang, tones="keep")


def tone_insensitive(lang: Lang | None = None) -> NormalizeConfig:
    """Config for the tone-insensitive view."""
    return NormalizeConfig(lang=lang, tones="strip")


def strip_tones(text: str) -> str:
    """Remove tone marks, keeping dot-below, dot-above and hooked letters. Result is NFC."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", _TONE_MARKS.sub("", decomposed))


def normalize(text: str, config: NormalizeConfig | None = None) -> str:
    """Normalise ``text`` under ``config`` (tone-aware defaults when omitted).

    Steps: lowercase, Unicode NFC (after removing tone marks when asked),
    apostrophe unification, punctuation to space, whitespace collapse.
    The function is idempotent.
    """
    cfg = config or NormalizeConfig()

    if cfg.lowercase:
        text = text.lower()  # before NFC: lower() can emit combining sequences (e.g. U+0130)
    text = strip_tones(text) if cfg.tones == "strip" else unicodedata.normalize("NFC", text)

    if cfg.punctuation == "strip":
        text = text.translate(_APOSTROPHES).translate(_PUNCTUATION)
        pattern = _ORPHAN_APOSTROPHE if cfg.lang == "ha" else _EDGE_APOSTROPHE
        text = pattern.sub(" ", text)

    return " ".join(text.split())
