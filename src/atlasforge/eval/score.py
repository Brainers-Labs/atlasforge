"""Scoring: turn a finished run into metrics, under both tone views.

Two rules keep the numbers honest:

* **Failures count against the model.** A failed or missing prediction is scored as
  an empty answer (wrong for accuracy, all-words-deleted for WER, zero for chrF), and
  the report states how many there were. Dropping failures would inflate scores.
* **Every text metric is reported under both views**: ``tone_aware`` and
  ``tone_insensitive``. Neither is silently preferred.

Per-example values are kept so ``compare`` can run paired statistics later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from atlasforge.errors import ConfigError
from atlasforge.eval import metrics as m
from atlasforge.eval.normalize import NormalizeConfig, Tones, normalize

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from atlasforge.eval.dataset import Dataset, Example
    from atlasforge.eval.runner import Record
    from atlasforge.types import Lang

VIEWS: Final[dict[str, Tones]] = {"tone_aware": "keep", "tone_insensitive": "strip"}

TASK_DEFAULTS: Final = {
    "generation": ("exact_match", "chrf"),
    "classification": ("accuracy", "macro_f1"),
    "asr": ("wer", "cer"),
}
KNOWN_METRICS: Final = frozenset(
    {"exact_match", "chrf", "chrf++", "wer", "cer", "accuracy", "macro_f1"}
)
LOWER_IS_BETTER: Final = frozenset({"wer", "cer"})
_CLASSIFICATION_ONLY: Final = frozenset({"accuracy", "macro_f1"})


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricSummary:
    """One metric under one view. ``mean`` averages per-example values; ``corpus`` is the
    pooled figure where one exists (chrF, WER, CER, macro-F1). ``n`` counts scored examples."""

    key: str
    name: str
    view: str
    n: int
    mean: float | None
    corpus: float | None
    higher_is_better: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreReport:
    """Everything needed to render or compare a run."""

    task: str
    dataset_sha256: str
    n_total: int
    n_ok: int
    n_failed: int
    n_missing: int
    metrics: tuple[MetricSummary, ...]
    per_example: dict[str, dict[str, float]]
    latency_ms: dict[str, float | None]
    normalization: dict[str, dict[str, Any]]

    def metric(self, key: str) -> MetricSummary:
        """Look up a summary by ``name@view``."""
        for summary in self.metrics:
            if summary.key == key:
                return summary
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_run(
    dataset: Dataset,
    results: Mapping[str, Record],
    *,
    metrics: Sequence[str] | None = None,
) -> ScoreReport:
    """Score ``results`` (from ``read_results``) against ``dataset``."""
    names = _validate(dataset, metrics)
    examples = dataset.examples

    predictions: list[str] = []
    n_failed = n_missing = 0
    latencies: list[float] = []
    for example in examples:
        record = results.get(example.id)
        if record is None:
            n_missing += 1
            predictions.append("")
        elif not record.ok:
            n_failed += 1
            predictions.append("")
        else:
            predictions.append(record.prediction or "")
            if record.latency_ms is not None:
                latencies.append(record.latency_ms)

    per_example: dict[str, dict[str, float]] = {e.id: {} for e in examples}
    summaries: list[MetricSummary] = []
    for view, tones in VIEWS.items():
        summaries.extend(
            _score_view(
                view,
                tones,
                names=names,
                examples=examples,
                predictions=predictions,
                per_example=per_example,
            )
        )

    return ScoreReport(
        task=dataset.task,
        dataset_sha256=dataset.sha256,
        n_total=len(examples),
        n_ok=len(examples) - n_failed - n_missing,
        n_failed=n_failed,
        n_missing=n_missing,
        metrics=tuple(summaries),
        per_example=per_example,
        latency_ms={
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": m.percentile(latencies, 50),
            "p95": m.percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
        normalization={
            view: {"tones": tones, "lowercase": True, "punctuation": "strip"}
            for view, tones in VIEWS.items()
        },
    )


def write_report_json(report: ScoreReport, path: str | Path) -> None:
    """Write the report as UTF-8 JSON (non-ASCII kept readable)."""
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _validate(dataset: Dataset, metrics: Sequence[str] | None) -> tuple[str, ...]:
    names = tuple(metrics) if metrics else TASK_DEFAULTS[dataset.task]
    unknown = [n for n in names if n not in KNOWN_METRICS]
    if unknown:
        raise ConfigError(
            f"Unknown metric(s): {', '.join(unknown)}.",
            hint=f"Available: {', '.join(sorted(KNOWN_METRICS))}.",
        )
    if dataset.task != "classification" and _CLASSIFICATION_ONLY & set(names):
        raise ConfigError(
            "accuracy and macro_f1 need a classification dataset.",
            hint="Use task 'classification', or pick exact_match / chrf / wer / cer.",
        )
    return names


def _score_view(
    view: str,
    tones: Tones,
    *,
    names: Sequence[str],
    examples: Sequence[Example],
    predictions: Sequence[str],
    per_example: dict[str, dict[str, float]],
) -> list[MetricSummary]:
    configs: dict[Lang | None, NormalizeConfig] = {}

    def config_for(lang: Lang | None) -> NormalizeConfig:
        if lang not in configs:
            configs[lang] = NormalizeConfig(lang=lang, tones=tones)
        return configs[lang]

    ids: list[str] = []
    preds: list[str] = []
    refs: list[str] = []
    for example, prediction in zip(examples, predictions, strict=True):
        if example.reference is None:
            continue
        cfg = config_for(example.lang)
        ids.append(example.id)
        preds.append(normalize(prediction, cfg))
        refs.append(normalize(example.reference, cfg))
    labels = set(refs)

    summaries: list[MetricSummary] = []
    for name in names:
        per_item = _PER_EXAMPLE.get(name)
        values = (
            [per_item(p, r, labels) for p, r in zip(preds, refs, strict=True)] if per_item else []
        )
        key = f"{name}@{view}"
        for example_id, value in zip(ids, values, strict=False):
            per_example[example_id][key] = value
        pooled = _CORPUS.get(name)
        summaries.append(
            MetricSummary(
                key=key,
                name=name,
                view=view,
                n=len(refs),
                mean=sum(values) / len(values) if values else None,
                corpus=pooled(preds, refs, labels) if pooled and refs else None,
                higher_is_better=name not in LOWER_IS_BETTER,
            )
        )
    return summaries


# (prediction, reference, label set) -> value. macro_f1 has no per-example value.
_PER_EXAMPLE: Final[dict[str, Callable[[str, str, set[str]], float]]] = {
    "exact_match": lambda p, r, _labels: m.exact_match(p, r),
    "chrf": lambda p, r, _labels: m.chrf(p, r),
    "chrf++": lambda p, r, _labels: m.chrf(p, r, word_order=2),
    "wer": lambda p, r, _labels: m.wer(p, r),
    "cer": lambda p, r, _labels: m.cer(p, r),
    "accuracy": lambda p, r, labels: 1.0 if m.extract_label(p, labels) == r else 0.0,
}

# (predictions, references, label set) -> pooled value. Metrics absent here have none.
_CORPUS: Final[dict[str, Callable[[list[str], list[str], set[str]], float | None]]] = {
    "chrf": lambda p, r, _labels: m.corpus_chrf(p, r),
    "chrf++": lambda p, r, _labels: m.corpus_chrf(p, r, word_order=2),
    "wer": lambda p, r, _labels: m.corpus_wer(p, r),
    "cer": lambda p, r, _labels: m.corpus_cer(p, r),
    "macro_f1": lambda p, r, labels: m.macro_f1([m.extract_label(x, labels) for x in p], r),
}
