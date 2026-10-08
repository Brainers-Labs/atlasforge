"""Metric primitives.

Inputs are expected to be **already normalised** (see ``normalize.py``); these
functions only compare strings. Scales: exact match / accuracy / F1 / WER / CER
are fractions (0-1, error rates can exceed 1); chrF is 0-100 as in sacrebleu.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import jiwer
from sacrebleu.metrics.chrf import CHRF

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

_CHRF: Final = CHRF()
_CHRF_PLUS_PLUS: Final = CHRF(word_order=2)


def exact_match(pred: str, ref: str) -> float:
    """1.0 if the strings are identical, else 0.0."""
    return 1.0 if pred == ref else 0.0


def chrf(pred: str, ref: str, *, word_order: int = 0) -> float:
    """Sentence chrF (``word_order=2`` gives chrF++). Two empty strings score 100."""
    if not pred and not ref:
        return 100.0
    if not pred or not ref:
        return 0.0
    metric = _CHRF_PLUS_PLUS if word_order else _CHRF
    return float(metric.sentence_score(pred, [ref]).score)


def corpus_chrf(preds: Sequence[str], refs: Sequence[str], *, word_order: int = 0) -> float:
    """Corpus chrF over aligned predictions and references."""
    _check_aligned(preds, refs)
    metric = _CHRF_PLUS_PLUS if word_order else _CHRF
    return float(metric.corpus_score(list(preds), [list(refs)]).score)


def wer(pred: str, ref: str) -> float:
    """Word error rate for one utterance. An empty reference scores 0 if the prediction is empty, else 1."""
    return _rate(jiwer.wer, pred, ref)


def cer(pred: str, ref: str) -> float:
    """Character error rate for one utterance (same empty-reference rule as :func:`wer`)."""
    return _rate(jiwer.cer, pred, ref)


def corpus_wer(preds: Sequence[str], refs: Sequence[str]) -> float | None:
    """Pooled WER (total errors / total reference words), the standard ASR figure.

    Pairs with an empty reference are excluded. Returns ``None`` if none remain.
    """
    return _corpus_rate(jiwer.wer, preds, refs)


def corpus_cer(preds: Sequence[str], refs: Sequence[str]) -> float | None:
    """Pooled CER; see :func:`corpus_wer`."""
    return _corpus_rate(jiwer.cer, preds, refs)


def extract_label(pred: str, labels: Collection[str], *, strict: bool = False) -> str | None:
    """Find which label a free-text answer names.

    The default is *loose*: returns the label occurring earliest as a whole word
    sequence (longest wins on a tie), or ``None``. Known limitation: negation
    ("not positive") is not understood, so keep prompts asking for the label only.

    ``strict=True`` requires the whole answer to *be* a label, so an answer that
    merely mentions one is no longer a match: ``"not positive"`` finds nothing
    instead of ``"positive"``. Use it when the prompt asks for the label and
    nothing else, and the model has been known to editorialise. Like loose mode it
    compares the string as given, so callers wanting tolerant matching should
    normalise first -- :func:`atlasforge.eval.score.score_run` does.
    """
    if strict:
        answer = pred.strip()
        return answer if answer and answer in labels else None
    padded = f" {pred} "
    best: tuple[int, int, str] | None = None
    for label in labels:
        if not label:
            continue
        index = padded.find(f" {label} ")
        if index == -1:
            continue
        candidate = (index, -len(label), label)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]


def macro_f1(preds: Sequence[str | None], refs: Sequence[str]) -> float:
    """Macro-averaged F1 over the classes present in ``refs``.

    A ``None`` prediction (no label found, or a failed call) counts as a miss for
    its reference class and adds no false positive anywhere.
    """
    _check_aligned(preds, refs)
    classes = set(refs)
    scores: list[float] = []
    for cls in classes:
        tp = sum(p == cls and r == cls for p, r in zip(preds, refs, strict=True))
        fp = sum(p == cls and r != cls for p, r in zip(preds, refs, strict=True))
        fn = sum(p != cls and r == cls for p, r in zip(preds, refs, strict=True))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return mean(scores)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of ``values``, which must not be empty.

    ``math.fsum``, not the builtin ``sum``, and deliberately. CPython 3.12 gave ``sum``
    Neumaier compensated summation for floats (gh-100425), so the same list of values adds
    up to a different last digit there than it does on 3.11 and earlier. A mean computed
    here reaches a report, and the committed example reports are compared byte for byte, so
    the arithmetic has to agree across the whole 3.10-3.13 CI matrix rather than describe
    the interpreter that happened to run it. ``fsum`` is exactly rounded — it returns the
    correctly rounded sum of the values, whatever their order — which makes that agreement
    a property of this code instead of two versions coinciding.
    """
    if not values:
        raise ValueError("no values to average")
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0-100). ``None`` for no data."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _rate(fn: Callable[..., float], pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return float(fn(ref, pred))


def _corpus_rate(
    fn: Callable[..., float], preds: Sequence[str], refs: Sequence[str]
) -> float | None:
    _check_aligned(preds, refs)
    pairs = [(p, r) for p, r in zip(preds, refs, strict=True) if r]
    if not pairs:
        return None
    return float(fn([r for _, r in pairs], [p for p, _ in pairs]))


def _check_aligned(preds: Sequence[object], refs: Sequence[object]) -> None:
    if len(preds) != len(refs):
        msg = f"predictions ({len(preds)}) and references ({len(refs)}) differ in length"
        raise ValueError(msg)
    if not preds:
        raise ValueError("no examples to score")
