"""Helpers that keep secrets out of output."""

from __future__ import annotations

_VISIBLE_TAIL = 4
_MAX_PREFIX = 6


def mask_secret(value: str) -> str:
    """Mask a token for display, revealing at most its prefix and last 4 characters.

    ``hf_abcdefghijkl`` -> ``hf_****ijkl``. Values too short to mask safely
    reveal nothing: ``abc`` -> ``****``.
    """
    if len(value) <= _VISIBLE_TAIL * 2:
        return "****"
    underscore = value.find("_", 0, _MAX_PREFIX)
    prefix = value[: underscore + 1] if underscore != -1 else ""
    return f"{prefix}****{value[-_VISIBLE_TAIL:]}"
