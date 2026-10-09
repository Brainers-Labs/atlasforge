"""Which official NCAIR1 ASR checkpoint serves which language.

There is one model per language and each is monolingual, so routing is a lookup
rather than a guess. AtlasForge does **no language detection**: the declared
``lang`` is trusted, and a wrong ``--lang`` produces confidently wrong text
rather than an error. See ``planning/21_NATLAS_DISCOVERY.md`` for the sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from atlasforge.types import Lang

#: The official checkpoint for each supported language. Verified against the
#: Hugging Face org ``NCAIR1``; changing one of these changes what users get.
ASR_MODELS: Final[dict[str, str]] = {
    "ha": "NCAIR1/Hausa-ASR",
    "yo": "NCAIR1/Yoruba-ASR",
    "ig": "NCAIR1/Igbo-ASR",
    "en": "NCAIR1/NigerianAccentedEnglish",
}


def asr_model_for(lang: Lang) -> str:
    """The official ASR checkpoint for ``lang``, e.g. ``NCAIR1/Hausa-ASR``."""
    return ASR_MODELS[lang]


def accept_licence_url(model: str) -> str:
    """The Hugging Face page where a gated model's licence is accepted."""
    return f"https://huggingface.co/{model}"
