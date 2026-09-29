"""Number formatting shared by every report, so a metric always reads the same way.

Fraction metrics (exact match, accuracy, macro-F1, WER, CER) are shown as percentages
and their differences as "pts"; chrF is already on a 0-100 scale and is shown as is.
"""

from __future__ import annotations

from typing import Final

FRACTION_METRICS: Final = frozenset({"exact_match", "accuracy", "macro_f1", "wer", "cer"})


def is_fraction(name: str) -> bool:
    """True for metrics stored as a 0-1 fraction."""
    return name in FRACTION_METRICS


def fmt_value(name: str, x: float | None) -> str:
    """A metric value: ``71.4%`` for fractions, ``54.3`` for chrF, ``-`` for no data."""
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if is_fraction(name) else f"{x:.1f}"


def fmt_delta(name: str, x: float | None) -> str:
    """A signed difference: ``+8.5 pts`` for fractions, ``+3.2`` for chrF."""
    if x is None:
        return "-"
    return f"{x * 100:+.1f} pts" if is_fraction(name) else f"{x:+.1f}"


def fmt_bound(name: str, x: float | None) -> str:
    """A signed interval endpoint, without a unit."""
    if x is None:
        return "-"
    return f"{x * 100:+.1f}" if is_fraction(name) else f"{x:+.1f}"
