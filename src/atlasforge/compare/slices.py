"""Slice analysis: where did the candidate improve, and where did it get worse?

A slice with fewer than ``MIN_SLICE_N`` examples is reported as *insufficient data*,
never as an improvement or a regression: a headline claim built on a dozen examples
is noise. Slice intervals are not adjusted for multiple comparisons; the report says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from atlasforge.compare.stats import paired_bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from atlasforge.eval.dataset import Example

MIN_SLICE_N: Final = 30
BUILTIN_FIELDS: Final = ("lang", "length", "has_number")
_SHORT_WORDS: Final = 5
_MEDIUM_WORDS: Final = 15
_DIGIT: Final = re.compile(r"\d")

Status = Literal["improved", "regressed", "no clear change", "insufficient data"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SliceResult:
    """One value of one slice field. Numbers are ``None`` when there is no data."""

    field: str
    value: str
    n: int
    base_mean: float | None
    candidate_mean: float | None
    delta: float | None
    low: float | None
    high: float | None
    status: Status


def slice_value(example: Example, field: str) -> str:
    """The slice bucket of ``example`` for ``field``.

    Built-in fields: ``lang``, ``length`` (short/medium/long by reference word count) and
    ``has_number`` (reference contains a digit). Any other name reads ``example.meta``.
    """
    if field == "lang":
        return example.lang or "(none)"
    if field == "length":
        words = len((example.reference or "").split())
        if words <= _SHORT_WORDS:
            return "short (<=5 words)"
        return "medium (6-15 words)" if words <= _MEDIUM_WORDS else "long (>15 words)"
    if field == "has_number":
        return "yes" if _DIGIT.search(example.reference or "") else "no"
    value = example.meta.get(field)
    return "(missing)" if value is None else str(value)


def analyse_slices(
    field: str,
    examples: Sequence[Example],
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool = True,
    min_n: int = MIN_SLICE_N,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[SliceResult]:
    """Compare the aligned ``base``/``candidate`` values within each bucket of ``field``.

    Results are ordered worst-first (largest regression, in the metric's own direction).
    """
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(slice_value(example, field), []).append(index)

    results = [
        _analyse_group(
            field,
            value,
            [base[i] for i in indices],
            [candidate[i] for i in indices],
            higher_is_better=higher_is_better,
            min_n=min_n,
            n_boot=n_boot,
            seed=seed,
        )
        for value, indices in groups.items()
    ]
    sign = 1 if higher_is_better else -1
    results.sort(key=lambda r: (r.delta is None, sign * (r.delta or 0.0)))
    return results


def _analyse_group(
    field: str,
    value: str,
    base: Sequence[float],
    candidate: Sequence[float],
    *,
    higher_is_better: bool,
    min_n: int,
    n_boot: int,
    seed: int,
) -> SliceResult:
    n = len(base)
    base_mean = sum(base) / n
    candidate_mean = sum(candidate) / n
    if n < min_n:
        return SliceResult(
            field=field,
            value=value,
            n=n,
            base_mean=base_mean,
            candidate_mean=candidate_mean,
            delta=None,
            low=None,
            high=None,
            status="insufficient data",
        )
    boot = paired_bootstrap(base, candidate, n_boot=n_boot, seed=seed)
    return SliceResult(
        field=field,
        value=value,
        n=n,
        base_mean=base_mean,
        candidate_mean=candidate_mean,
        delta=boot.mean,
        low=boot.low,
        high=boot.high,
        status=verdict(boot.low, boot.high, higher_is_better=higher_is_better),
    )


def verdict(low: float, high: float, *, higher_is_better: bool) -> Status:
    """``improved`` / ``regressed`` only when the whole interval is on one side of zero."""
    if low > 0:
        return "improved" if higher_is_better else "regressed"
    if high < 0:
        return "regressed" if higher_is_better else "improved"
    return "no clear change"


def group_counts(examples: Sequence[Example], field: str) -> Mapping[str, int]:
    """How many examples fall in each bucket of ``field`` (for previews and validation)."""
    counts: dict[str, int] = {}
    for example in examples:
        key = slice_value(example, field)
        counts[key] = counts.get(key, 0) + 1
    return counts
