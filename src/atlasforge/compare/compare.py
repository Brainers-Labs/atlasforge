"""Compare two scored runs on the same dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from atlasforge.compare.slices import (
    BUILTIN_FIELDS,
    MIN_SLICE_N,
    SliceResult,
    analyse_slices,
    verdict,
)
from atlasforge.compare.stats import McNemarResult, is_binary, mcnemar_exact, paired_bootstrap
from atlasforge.errors import ConfigError
from atlasforge.eval.runner import RESULTS_NAME, read_manifest, read_results
from atlasforge.eval.score import ScoreReport, score_run

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atlasforge.eval.dataset import Dataset

_EPS: Final = 1e-12
POOLED_ONLY: Final = "not tested (pooled metric)"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunInfo:
    """Identity and health of one run, taken from its manifest and score."""

    model: str | None
    revision: str | None
    backend: str | None
    n_failed: int
    n_missing: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricComparison:
    """One metric, paired over the examples both runs scored.

    ``delta`` is candidate minus base. The interval and verdict describe the mean
    per-example difference; ``*_corpus`` are the pooled figures (WER, chrF, macro-F1).
    """

    key: str
    name: str
    view: str
    higher_is_better: bool
    n: int
    base_mean: float | None
    candidate_mean: float | None
    delta: float | None
    low: float | None
    high: float | None
    verdict: str
    wins: int
    ties: int
    losses: int
    base_corpus: float | None
    candidate_corpus: float | None
    mcnemar: McNemarResult | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonReport:
    task: str
    dataset_sha256: str
    n_total: int
    base: RunInfo
    candidate: RunInfo
    primary: str | None
    metrics: tuple[MetricComparison, ...]
    slices: tuple[SliceResult, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def regressed_metrics(self) -> tuple[MetricComparison, ...]:
        return tuple(m for m in self.metrics if m.verdict == "regressed")

    @property
    def regressed_slices(self) -> tuple[SliceResult, ...]:
        return tuple(s for s in self.slices if s.status == "regressed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_runs(
    dataset: Dataset,
    base_dir: str | Path,
    candidate_dir: str | Path,
    *,
    metrics: Sequence[str] | None = None,
    slice_fields: Sequence[str] = BUILTIN_FIELDS,
    primary: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    min_slice_n: int = MIN_SLICE_N,
) -> ComparisonReport:
    """Score two run directories on ``dataset`` and compare them.

    Both runs must have been produced from exactly this dataset (checked by hash),
    otherwise the pairing would be meaningless.
    """
    scores: list[ScoreReport] = []
    infos: list[RunInfo] = []
    for directory in (base_dir, candidate_dir):
        manifest = read_manifest(directory)
        if manifest.get("dataset_sha256") != dataset.sha256 or manifest.get("task") != dataset.task:
            raise ConfigError(
                f"{Path(directory)} was not produced from this dataset and task.",
                hint="Re-run it on the same dataset file, or pass the dataset it was run on.",
            )
        report = score_run(dataset, read_results(Path(directory) / RESULTS_NAME), metrics=metrics)
        scores.append(report)
        infos.append(
            RunInfo(
                model=manifest.get("model"),
                revision=manifest.get("revision"),
                backend=manifest.get("backend"),
                n_failed=report.n_failed,
                n_missing=report.n_missing,
            )
        )
    return compare_scores(
        dataset,
        scores[0],
        scores[1],
        base_info=infos[0],
        candidate_info=infos[1],
        slice_fields=slice_fields,
        primary=primary,
        n_boot=n_boot,
        seed=seed,
        min_slice_n=min_slice_n,
    )


def compare_scores(
    dataset: Dataset,
    base: ScoreReport,
    candidate: ScoreReport,
    *,
    base_info: RunInfo | None = None,
    candidate_info: RunInfo | None = None,
    slice_fields: Sequence[str] = BUILTIN_FIELDS,
    primary: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    min_slice_n: int = MIN_SLICE_N,
) -> ComparisonReport:
    """Compare two already-scored runs."""
    if base.dataset_sha256 != candidate.dataset_sha256 or base.task != candidate.task:
        raise ConfigError(
            "The two runs were scored on different datasets or tasks.",
            hint="Compare runs of the same dataset file.",
        )
    candidate_by_key = {m.key: m for m in candidate.metrics}
    comparisons: list[MetricComparison] = []
    aligned: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for summary in base.metrics:
        other = candidate_by_key.get(summary.key)
        if other is None:
            continue
        indices, base_values, candidate_values = _pair(dataset, base, candidate, summary.key)
        aligned[summary.key] = (indices, base_values, candidate_values)
        comparisons.append(
            _compare_metric(
                summary.key,
                summary.name,
                summary.view,
                higher_is_better=summary.higher_is_better,
                values=(base_values, candidate_values),
                corpus=(summary.corpus, other.corpus),
                n_boot=n_boot,
                seed=seed,
            )
        )

    chosen = _choose_primary(primary, comparisons)
    slices: list[SliceResult] = []
    if chosen is not None and chosen in aligned:
        indices, base_values, candidate_values = aligned[chosen]
        higher = next(c.higher_is_better for c in comparisons if c.key == chosen)
        subset = [dataset.examples[i] for i in indices]
        for name in slice_fields:
            slices.extend(
                analyse_slices(
                    name,
                    subset,
                    base_values,
                    candidate_values,
                    higher_is_better=higher,
                    min_n=min_slice_n,
                    n_boot=n_boot,
                    seed=seed,
                )
            )

    return ComparisonReport(
        task=dataset.task,
        dataset_sha256=dataset.sha256,
        n_total=len(dataset),
        base=base_info or _info(base),
        candidate=candidate_info or _info(candidate),
        primary=chosen,
        metrics=tuple(comparisons),
        slices=tuple(slices),
        settings={
            "n_boot": n_boot,
            "seed": seed,
            "confidence": 0.95,
            "min_slice_n": min_slice_n,
            "slice_fields": list(slice_fields),
        },
    )


def _pair(
    dataset: Dataset, base: ScoreReport, candidate: ScoreReport, key: str
) -> tuple[list[int], list[float], list[float]]:
    """Per-example values present in both runs, in dataset order, with their positions."""
    indices: list[int] = []
    base_values: list[float] = []
    candidate_values: list[float] = []
    for position, example in enumerate(dataset.examples):
        b = base.per_example.get(example.id, {}).get(key)
        c = candidate.per_example.get(example.id, {}).get(key)
        if b is not None and c is not None:
            indices.append(position)
            base_values.append(b)
            candidate_values.append(c)
    return indices, base_values, candidate_values


def _compare_metric(
    key: str,
    name: str,
    view: str,
    *,
    higher_is_better: bool,
    values: tuple[list[float], list[float]],
    corpus: tuple[float | None, float | None],
    n_boot: int,
    seed: int,
) -> MetricComparison:
    base_values, candidate_values = values
    if not base_values:  # pooled-only metric (e.g. macro-F1) or no references
        return MetricComparison(
            key=key,
            name=name,
            view=view,
            higher_is_better=higher_is_better,
            n=0,
            base_mean=None,
            candidate_mean=None,
            delta=None,
            low=None,
            high=None,
            verdict=POOLED_ONLY,
            wins=0,
            ties=0,
            losses=0,
            base_corpus=corpus[0],
            candidate_corpus=corpus[1],
            mcnemar=None,
        )
    boot = paired_bootstrap(base_values, candidate_values, n_boot=n_boot, seed=seed)
    sign = 1.0 if higher_is_better else -1.0
    diffs = [sign * (c - b) for b, c in zip(base_values, candidate_values, strict=True)]
    n = len(base_values)
    return MetricComparison(
        key=key,
        name=name,
        view=view,
        higher_is_better=higher_is_better,
        n=n,
        base_mean=sum(base_values) / n,
        candidate_mean=sum(candidate_values) / n,
        delta=boot.mean,
        low=boot.low,
        high=boot.high,
        verdict=verdict(boot.low, boot.high, higher_is_better=higher_is_better),
        wins=sum(d > _EPS for d in diffs),
        ties=sum(abs(d) <= _EPS for d in diffs),
        losses=sum(d < -_EPS for d in diffs),
        base_corpus=corpus[0],
        candidate_corpus=corpus[1],
        mcnemar=(
            mcnemar_exact(base_values, candidate_values)
            if is_binary(base_values) and is_binary(candidate_values)
            else None
        ),
    )


def _choose_primary(requested: str | None, comparisons: Sequence[MetricComparison]) -> str | None:
    tested = [c for c in comparisons if c.n > 0]
    if requested is not None:
        if requested not in {c.key for c in tested}:
            raise ConfigError(
                f"Primary metric {requested!r} has no per-example values to compare.",
                hint=f"Choose one of: {', '.join(c.key for c in tested) or '(none)'}.",
            )
        return requested
    for comparison in tested:
        if comparison.view == "tone_aware":
            return comparison.key
    return tested[0].key if tested else None


def _info(report: ScoreReport) -> RunInfo:
    return RunInfo(
        model=None,
        revision=None,
        backend=None,
        n_failed=report.n_failed,
        n_missing=report.n_missing,
    )
