"""Paired statistics for comparing two models on the same examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Resample matrix is capped near this many integers so huge datasets stay in memory budget.
_MAX_MATRIX_CELLS: Final = 4_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class BootstrapResult:
    """Mean paired difference (candidate minus base) and its percentile confidence interval."""

    mean: float
    low: float
    high: float
    n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class McNemarResult:
    """Exact McNemar test on paired correct/incorrect outcomes."""

    base_only: int  # base right, candidate wrong
    candidate_only: int  # base wrong, candidate right
    p_value: float


def paired_bootstrap(
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Paired bootstrap of the mean difference ``candidate - base``.

    Resamples *examples* (not scores independently), so per-example difficulty cancels
    out. Deterministic for a given ``seed``. Identical inputs give a zero-width interval.
    """
    if len(base) != len(candidate):
        raise ValueError("base and candidate must be the same length")
    if not base:
        raise ValueError("no paired examples")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    diffs = np.asarray(candidate, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    n = diffs.size
    rng = np.random.default_rng(seed)
    rows = max(1, min(n_boot, _MAX_MATRIX_CELLS // n))
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, rows):
        stop = min(start + rows, n_boot)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = diffs[indices].mean(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return BootstrapResult(mean=float(diffs.mean()), low=float(low), high=float(high), n=int(n))


def mcnemar_exact(base: Sequence[float], candidate: Sequence[float]) -> McNemarResult:
    """Exact two-sided McNemar test. Inputs must be 0/1 correctness values."""
    if len(base) != len(candidate):
        raise ValueError("base and candidate must be the same length")
    if any(v not in (0.0, 1.0) for v in (*base, *candidate)):
        raise ValueError("McNemar's test needs 0/1 values")
    base_only = sum(b == 1.0 and c == 0.0 for b, c in zip(base, candidate, strict=True))
    candidate_only = sum(b == 0.0 and c == 1.0 for b, c in zip(base, candidate, strict=True))
    discordant = base_only + candidate_only
    if discordant == 0:
        return McNemarResult(base_only=0, candidate_only=0, p_value=1.0)
    smaller = min(base_only, candidate_only)
    tail = sum(comb(discordant, k) for k in range(smaller + 1))
    p_value = min(1.0, 2 * tail / 2**discordant)
    return McNemarResult(base_only=base_only, candidate_only=candidate_only, p_value=p_value)


def is_binary(values: Sequence[float]) -> bool:
    """True if every value is exactly 0 or 1 (a correct/incorrect outcome)."""
    return bool(values) and all(v in (0.0, 1.0) for v in values)
