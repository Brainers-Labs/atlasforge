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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from atlasforge.errors import ConfigError
from atlasforge.eval import custom, flags
from atlasforge.eval import metrics as m
from atlasforge.eval.asr_analysis import AsrAnalysis, analyse
from atlasforge.eval.normalize import NormalizeConfig, Tones, normalize

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from atlasforge.eval.custom import MetricFn
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
    {
        "exact_match",
        "chrf",
        "chrf++",
        "wer",
        "cer",
        "accuracy",
        "macro_f1",
        "accuracy_strict",
        "macro_f1_strict",
    }
)
LOWER_IS_BETTER: Final = frozenset({"wer", "cer"})
_CLASSIFICATION_ONLY: Final = frozenset(
    {"accuracy", "macro_f1", "accuracy_strict", "macro_f1_strict"}
)


#: A metric as scoring sees it: a name, how to score one example, how to pool them.
#: ``per_example`` receives the (prediction, reference, label set, example).
@dataclass(frozen=True, slots=True)
class Metric:
    """One metric, either built in or supplied by the caller."""

    name: str
    per_example: Callable[[str, str, set[str], Example], float] | None
    corpus: Callable[[list[str], list[str], set[str]], float | None] | None
    higher_is_better: bool


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
    flags: dict[str, int] = field(default_factory=dict)
    n_flagged: int = 0
    per_example_flags: dict[str, list[str]] = field(default_factory=dict)
    asr: AsrAnalysis | None = None

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
    metrics: Sequence[str | MetricFn] | None = None,
) -> ScoreReport:
    """Score ``results`` (from ``read_results``) against ``dataset``.

    ``metrics`` takes built-in names and/or custom callables, and/or ``module:function``
    strings naming them; ``None`` means the task's defaults. See
    :mod:`atlasforge.eval.custom`.
    """
    chosen = _resolve(dataset, metrics)
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
                metrics=chosen,
                examples=examples,
                predictions=predictions,
                per_example=per_example,
            )
        )

    per_example_flags = (
        {} if dataset.task == "classification" else _flag_examples(examples, predictions, results)
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
        flags=flags.summarise(per_example_flags),
        n_flagged=sum(1 for raised in per_example_flags.values() if raised),
        per_example_flags={
            example_id: list(raised) for example_id, raised in per_example_flags.items() if raised
        },
        asr=_asr_analysis(dataset, examples, predictions),
    )


def write_report_json(report: ScoreReport, path: str | Path) -> None:
    """Write the report as UTF-8 JSON (non-ASCII kept readable)."""
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _resolve(dataset: Dataset, metrics: Sequence[str | MetricFn] | None) -> tuple[Metric, ...]:
    """Turn what the caller asked for into runnable metrics, or say what is wrong."""
    asked: Sequence[str | MetricFn] = tuple(metrics) if metrics else TASK_DEFAULTS[dataset.task]
    chosen: list[Metric] = []
    seen: set[str] = set()
    for item in asked:
        metric = _custom_metric(item) if callable(item) else _builtin_metric(item)
        if metric.name in seen:
            raise ConfigError(
                f"{metric.name!r} was given more than once.",
                hint="Two metrics cannot share a name; the report keys would collide.",
            )
        seen.add(metric.name)
        chosen.append(metric)
    misused = sorted(_CLASSIFICATION_ONLY & {m.name for m in chosen})
    if dataset.task != "classification" and misused:
        verb = "needs" if len(misused) == 1 else "need"
        raise ConfigError(
            f"{', '.join(misused)} {verb} a classification dataset.",
            hint="Use task 'classification', or pick exact_match / chrf / wer / cer.",
        )
    return tuple(chosen)


def _builtin_metric(name: str) -> Metric:
    if ":" in name:
        return _custom_metric(custom.resolve(name))
    if name not in KNOWN_METRICS:
        raise ConfigError(
            f"Unknown metric {name!r}.",
            hint=f"Available: {', '.join(sorted(KNOWN_METRICS))}, or a custom metric "
            "as module:function.",
        )
    return Metric(
        name=name,
        per_example=_PER_EXAMPLE.get(name),
        corpus=_CORPUS.get(name),
        higher_is_better=name not in LOWER_IS_BETTER,
    )


def _custom_metric(fn: MetricFn) -> Metric:
    """Adapt a caller's ``(prediction, reference, example)`` callable to our view loop."""
    name = custom.name_of(fn)
    if name in KNOWN_METRICS:
        raise ConfigError(
            f"A custom metric may not be called {name!r}: that name is taken by a built-in.",
            hint="Rename your function; the two would be indistinguishable in the report.",
        )

    def per_example(_pred: str, _ref: str, _labels: set[str], example: Example) -> float:
        return float(fn(_pred, _ref, example))

    return Metric(
        name=name,
        per_example=per_example,
        corpus=None,  # a custom metric has no pooled form we could invent
        higher_is_better=custom.declares_higher_is_better(fn),
    )


def _flag_examples(
    examples: Sequence[Example],
    predictions: Sequence[str],
    results: Mapping[str, Record],
) -> dict[str, tuple[str, ...]]:
    """Deterministic failure-mode flags, for the answers we actually received.

    A failed or missing call has no answer to flag — the report counts those on their own —
    and neither has an example with no reference to check the answer against.
    """
    return {
        example.id: flags.flags_for(example, prediction)
        for example, prediction in zip(examples, predictions, strict=True)
        if example.reference is not None and (record := results.get(example.id)) and record.ok
    }


def _asr_analysis(
    dataset: Dataset, examples: Sequence[Example], predictions: Sequence[str]
) -> AsrAnalysis | None:
    """Alignment-based error analysis, for speech runs only. Needs no audio: it reads text."""
    if dataset.task != "asr":
        return None
    pairs = []
    for example, prediction in zip(examples, predictions, strict=True):
        if not example.reference:
            continue
        config = NormalizeConfig(lang=example.lang, tones="keep")
        pairs.append(
            (
                example.id,
                normalize(example.reference, config),
                normalize(prediction, config),
            )
        )
    return analyse(pairs)


def _score_view(
    view: str,
    tones: Tones,
    *,
    metrics: Sequence[Metric],
    examples: Sequence[Example],
    predictions: Sequence[str],
    per_example: dict[str, dict[str, float]],
) -> list[MetricSummary]:
    configs: dict[Lang | None, NormalizeConfig] = {}

    def config_for(lang: Lang | None) -> NormalizeConfig:
        if lang not in configs:
            configs[lang] = NormalizeConfig(lang=lang, tones=tones)
        return configs[lang]

    scored: list[Example] = []
    ids: list[str] = []
    preds: list[str] = []
    refs: list[str] = []
    for example, prediction in zip(examples, predictions, strict=True):
        if example.reference is None:
            continue
        cfg = config_for(example.lang)
        scored.append(example)
        ids.append(example.id)
        preds.append(normalize(prediction, cfg))
        refs.append(normalize(example.reference, cfg))
    labels = set(refs)

    summaries: list[MetricSummary] = []
    for metric in metrics:
        per_item = metric.per_example
        values = (
            [
                per_item(p, r, labels, example)
                for p, r, example in zip(preds, refs, scored, strict=True)
            ]
            if per_item
            else []
        )
        key = f"{metric.name}@{view}"
        for example_id, value in zip(ids, values, strict=False):
            per_example[example_id][key] = value
        pooled = metric.corpus
        summaries.append(
            MetricSummary(
                key=key,
                name=metric.name,
                view=view,
                n=len(refs),
                mean=sum(values) / len(values) if values else None,
                corpus=pooled(preds, refs, labels) if pooled and refs else None,
                higher_is_better=metric.higher_is_better,
            )
        )
    return summaries


# (prediction, reference, label set, example) -> value. macro_f1 has no per-example value.
_PER_EXAMPLE: Final[dict[str, Callable[[str, str, set[str], Example], float]]] = {
    "exact_match": lambda p, r, _labels, _ex: m.exact_match(p, r),
    "chrf": lambda p, r, _labels, _ex: m.chrf(p, r),
    "chrf++": lambda p, r, _labels, _ex: m.chrf(p, r, word_order=2),
    "wer": lambda p, r, _labels, _ex: m.wer(p, r),
    "cer": lambda p, r, _labels, _ex: m.cer(p, r),
    "accuracy": lambda p, r, labels, _ex: 1.0 if m.extract_label(p, labels) == r else 0.0,
    "accuracy_strict": lambda p, r, labels, _ex: (
        1.0 if m.extract_label(p, labels, strict=True) == r else 0.0
    ),
}

# (predictions, references, label set) -> pooled value. Metrics absent here have none.
_CORPUS: Final[dict[str, Callable[[list[str], list[str], set[str]], float | None]]] = {
    "chrf": lambda p, r, _labels: m.corpus_chrf(p, r),
    "chrf++": lambda p, r, _labels: m.corpus_chrf(p, r, word_order=2),
    "wer": lambda p, r, _labels: m.corpus_wer(p, r),
    "cer": lambda p, r, _labels: m.corpus_cer(p, r),
    "macro_f1": lambda p, r, labels: m.macro_f1([m.extract_label(x, labels) for x in p], r),
    "macro_f1_strict": lambda p, r, labels: m.macro_f1(
        [m.extract_label(x, labels, strict=True) for x in p], r
    ),
}
