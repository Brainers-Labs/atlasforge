"""Dataset health checks: the problems that quietly ruin fine-tuning and evaluation.

Unlike ``load_dataset`` (which stops at the first error), this collects *everything*:
malformed lines, duplicates and conflicting labels, train/test leakage, broken Unicode,
stripped diacritics and class imbalance.

What it does **not** do: identify languages. There is no reliable off-the-shelf language
ID for Hausa/Yoruba/Igbo, so it reports the *declared* ``lang`` field plus diacritic
statistics, and says so.
"""

from __future__ import annotations

import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from atlasforge.eval.dataset import Example, ParsedFile, parse_file
from atlasforge.eval.normalize import normalize, tone_insensitive

if TYPE_CHECKING:
    from pathlib import Path

    from atlasforge.eval.dataset import Task

Severity = Literal["error", "warning", "info"]

MIN_DIACRITIC_TEXTS: Final = 20
IMBALANCE_MINORITY_SHARE: Final = 0.10
IMBALANCE_MAJORITY_SHARE: Final = 0.80
MAX_LISTED_ERRORS: Final = 50
MAX_LISTED_IDS: Final = 5

_TONE_MARKS: Final = frozenset("̀́̂̄̌")
_UNDERDOT: Final = "̣"
_HOOKED_LETTERS: Final = frozenset("ɓɗƙƴƁƊƘƳ")
_ALLOWED_CONTROLS: Final = frozenset("\n\r\t")


@dataclass(frozen=True, slots=True, kw_only=True)
class Issue:
    """One finding. ``line`` is set for line-level problems, ``count`` for aggregate ones."""

    severity: Severity
    code: str
    message: str
    line: int | None = None
    count: int | None = None
    hint: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    path: str
    task: str
    sha256: str
    n_lines: int
    n_examples: int
    issues: tuple[Issue, ...]
    stats: dict[str, Any]

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings and notes do not fail a dataset)."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dataset(
    path: str | Path,
    task: Task,
    *,
    against: str | Path | None = None,
) -> ValidationReport:
    """Validate ``path`` for ``task``; with ``against`` also check for train/test leakage."""
    parsed = parse_file(path, task)
    issues: list[Issue] = _line_errors(parsed)
    examples = [example for _, example in parsed.examples]
    if not examples:
        issues.append(Issue(severity="error", code="no-examples", message="No valid examples."))

    issues += _content_checks(examples, task)
    issues += _diacritic_checks(examples)
    issues += _label_checks(examples, task)
    if against is not None:
        issues += _leakage_checks(examples, task, against)

    return ValidationReport(
        path=str(parsed.path),
        task=task,
        sha256=parsed.sha256,
        n_lines=parsed.n_lines,
        n_examples=len(examples),
        issues=tuple(issues),
        stats=_stats(examples, task),
    )


def _line_errors(parsed: ParsedFile) -> list[Issue]:
    issues = [
        Issue(
            severity="error",
            code="line-error",
            message=e.message.removeprefix(f"line {e.line}: "),  # the line is its own field
            line=e.line,
            hint=e.hint,
        )
        for e in parsed.errors[:MAX_LISTED_ERRORS]
    ]
    extra = len(parsed.errors) - MAX_LISTED_ERRORS
    if extra > 0:
        issues.append(
            Issue(
                severity="error",
                code="line-error",
                message=f"...and {extra} more line errors",
                count=extra,
            )
        )
    return issues


def _texts(example: Example) -> list[str]:
    """Every free-text field of an example."""
    texts = [m["content"] for m in example.messages]
    if example.reference is not None:
        texts.append(example.reference)
    return texts


def _input_text(example: Example) -> str:
    return "\n".join(m["content"] for m in example.messages)


def _key(example: Example, text: str) -> str:
    return normalize(text, tone_insensitive(example.lang))


def _sample(ids: list[str]) -> str:
    shown = ", ".join(ids[:MAX_LISTED_IDS])
    return shown + (f" (+{len(ids) - MAX_LISTED_IDS} more)" if len(ids) > MAX_LISTED_IDS else "")


def _text_problems(text: str) -> set[str]:
    """Which text-health problems ``text`` has."""
    found: set[str] = set()
    if text != unicodedata.normalize("NFC", text):
        found.add("non_nfc")
    if "\N{REPLACEMENT CHARACTER}" in text:
        found.add("replacement")
    if any(unicodedata.category(c) == "Cc" and c not in _ALLOWED_CONTROLS for c in text):
        found.add("controls")
    if text != text.strip():
        found.add("padded")
    return found


def _scan(examples: list[Example], task: Task) -> dict[str, list[str]]:
    """Example ids affected by each text-health problem."""
    found: dict[str, list[str]] = {
        k: []
        for k in ("non_nfc", "replacement", "controls", "padded", "empty_inputs", "empty_refs")
    }
    for example in examples:
        for problem in set().union(*(_text_problems(t) for t in _texts(example))):
            found[problem].append(example.id)
        if task != "asr" and not _key(example, _input_text(example)):
            found["empty_inputs"].append(example.id)
        if example.reference is not None and not _key(example, example.reference):
            found["empty_refs"].append(example.id)
    return found


def _content_checks(examples: list[Example], task: Task) -> list[Issue]:
    return _text_health(examples, task) + _duplicate_checks(examples, task)


def _text_health(examples: list[Example], task: Task) -> list[Issue]:
    found = _scan(examples, task)
    issues: list[Issue] = []
    replacement, controls, non_nfc = found["replacement"], found["controls"], found["non_nfc"]
    padded, empty_inputs, empty_refs = found["padded"], found["empty_inputs"], found["empty_refs"]

    if replacement:
        issues.append(
            _issue(
                "warning",
                "replacement-character",
                "contain U+FFFD (encoding damage)",
                replacement,
                "Re-export the source data as UTF-8; the original characters are lost.",
            )
        )
    if controls:
        issues.append(
            _issue("warning", "control-characters", "contain control characters", controls)
        )
    if non_nfc:
        issues.append(
            _issue(
                "warning",
                "non-nfc",
                "have text not in Unicode NFC form",
                non_nfc,
                "Normalise with unicodedata.normalize('NFC', text) before training. "
                "Scoring normalises anyway, but training data should be consistent.",
            )
        )
    if padded:
        issues.append(
            _issue("info", "surrounding-whitespace", "have leading/trailing whitespace", padded)
        )
    if empty_inputs:
        issues.append(
            _issue(
                "warning",
                "empty-input",
                "have an input that is empty after normalisation",
                empty_inputs,
            )
        )
    if empty_refs:
        severity: Severity = "error" if task in ("classification", "asr") else "warning"
        issues.append(
            _issue(
                severity,
                "empty-reference",
                "have a reference that is empty after normalisation",
                empty_refs,
                "Such examples cannot be scored.",
            )
        )
    return issues


def _duplicate_checks(examples: list[Example], task: Task) -> list[Issue]:
    """Duplicates and conflicting labels (tone-insensitive, so near-copies are caught)."""
    issues: list[Issue] = []
    by_pair: dict[tuple[str, str], list[str]] = {}
    by_input: dict[str, dict[str, list[str]]] = {}
    for example in examples:
        if task == "asr" or example.reference is None:
            continue
        text_key = _key(example, _input_text(example))
        ref_key = _key(example, example.reference)
        by_pair.setdefault((text_key, ref_key), []).append(example.id)
        by_input.setdefault(text_key, {}).setdefault(ref_key, []).append(example.id)

    duplicate_ids = [i for ids in by_pair.values() if len(ids) > 1 for i in ids[1:]]
    if duplicate_ids:
        issues.append(
            _issue(
                "warning",
                "duplicate-content",
                "repeat an earlier example (same input and reference)",
                duplicate_ids,
                "Duplicates inflate scores and can leak between splits.",
            )
        )
    conflicting = [ids[0] for refs in by_input.values() if len(refs) > 1 for ids in refs.values()]
    if conflicting:
        issues.append(
            _issue(
                "warning",
                "conflicting-references",
                "share an input with a different reference",
                conflicting,
                "The same question has more than one 'correct' answer.",
            )
        )
    return issues


def _issue(
    severity: Severity, code: str, what: str, ids: list[str], hint: str | None = None
) -> Issue:
    return Issue(
        severity=severity,
        code=code,
        message=f"{len(ids)} example(s) {what}: {_sample(ids)}",
        count=len(ids),
        hint=hint,
    )


def _marks(text: str) -> tuple[bool, bool, bool]:
    """(has tone marks, has underdot, has Hausa hooked letter)."""
    decomposed = unicodedata.normalize("NFD", text)
    return (
        any(c in _TONE_MARKS for c in decomposed),
        _UNDERDOT in decomposed,
        any(c in _HOOKED_LETTERS for c in text),
    )


def _diacritic_stats(examples: list[Example]) -> dict[str, dict[str, float | int]]:
    per_lang: dict[str, list[tuple[bool, bool, bool]]] = {}
    for example in examples:
        if example.lang in ("ha", "yo", "ig"):
            per_lang.setdefault(example.lang, []).extend(_marks(t) for t in _texts(example))
    stats: dict[str, dict[str, float | int]] = {}
    for lang, marks in per_lang.items():
        n = len(marks)
        stats[lang] = {
            "texts": n,
            "tone_rate": sum(m[0] for m in marks) / n,
            "underdot_rate": sum(m[1] for m in marks) / n,
            "hooked_rate": sum(m[2] for m in marks) / n,
        }
    return stats


def _diacritic_checks(examples: list[Example]) -> list[Issue]:
    """Flag languages whose diacritics look stripped. Heuristic, so warnings only."""
    issues: list[Issue] = []
    expectations = {
        "yo": (("tone_rate", "tone marks"), ("underdot_rate", "underdotted letters (e, o, s)")),
        "ig": (("underdot_rate", "underdotted letters (i, o, u)"),),
        "ha": (("hooked_rate", "hooked letters (b, d, k, y)"),),
    }
    for lang, stats in _diacritic_stats(examples).items():
        if stats["texts"] < MIN_DIACRITIC_TEXTS:
            continue
        for field, label in expectations[lang]:
            if stats[field] == 0:
                issues.append(
                    Issue(
                        severity="warning",
                        code="diacritics-missing",
                        message=(
                            f"None of {stats['texts']} '{lang}' texts contain {label}. "
                            "The diacritics may have been stripped."
                        ),
                        hint="Stripped diacritics change meaning and hurt training and scoring.",
                    )
                )
    return issues


def _label_checks(examples: list[Example], task: Task) -> list[Issue]:
    if task != "classification":
        return []
    counts = Counter(_key(e, e.reference) for e in examples if e.reference is not None)
    if not counts:
        return []
    total = sum(counts.values())
    if len(counts) == 1:
        return [
            Issue(
                severity="warning",
                code="single-class",
                message=f"Only one class present: {next(iter(counts))!r}.",
                hint="Accuracy will look perfect for a model that always says that label.",
            )
        ]
    issues: list[Issue] = []
    (top, top_n), (low, low_n) = counts.most_common()[0], counts.most_common()[-1]
    if top_n / total > IMBALANCE_MAJORITY_SHARE:
        issues.append(
            Issue(
                severity="warning",
                code="class-imbalance",
                message=f"Class {top!r} is {top_n / total:.0%} of the data.",
                hint="Prefer macro-F1 over accuracy, or rebalance.",
            )
        )
    elif low_n / total < IMBALANCE_MINORITY_SHARE:
        issues.append(
            Issue(
                severity="warning",
                code="class-imbalance",
                message=f"Class {low!r} is only {low_n / total:.0%} of the data.",
                hint="Prefer macro-F1 over accuracy, or rebalance.",
            )
        )
    return issues


def _leakage_checks(examples: list[Example], task: Task, against: str | Path) -> list[Issue]:
    if task == "asr":
        return [
            Issue(
                severity="info",
                code="leakage-skipped",
                message="Train/test leakage check is not implemented for ASR datasets.",
            )
        ]
    other = parse_file(against, task)
    issues: list[Issue] = []
    if other.errors:
        issues.append(
            Issue(
                severity="warning",
                code="against-parse-errors",
                message=f"{len(other.errors)} line(s) in {other.path.name} could not be parsed and "
                "were skipped in the leakage check.",
                count=len(other.errors),
            )
        )
    exact = {_input_text(e) for _, e in other.examples}
    keyed = {_key(e, _input_text(e)) for _, e in other.examples}
    leaked = [e.id for e in examples if _key(e, _input_text(e)) in keyed]
    if leaked:
        exact_n = sum(1 for e in examples if _input_text(e) in exact)
        issues.append(
            Issue(
                severity="error",
                code="train-test-leakage",
                message=(
                    f"{len(leaked)} example(s) also appear in {other.path.name} "
                    f"({exact_n} identical, {len(leaked) - exact_n} differing only in case, "
                    f"punctuation or tone marks): {_sample(leaked)}"
                ),
                count=len(leaked),
                hint="Remove them from one side, or your evaluation measures memorisation.",
            )
        )
    return issues


def _stats(examples: list[Example], task: Task) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "langs": dict(Counter(e.lang or "(none)" for e in examples)),
        "diacritics": _diacritic_stats(examples),
    }
    for name, values in (
        ("input_words", [len(_input_text(e).split()) for e in examples if e.messages]),
        (
            "reference_words",
            [len(e.reference.split()) for e in examples if e.reference is not None],
        ),
    ):
        if values:
            stats[name] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
    if task == "classification":
        stats["labels"] = dict(
            Counter(_key(e, e.reference) for e in examples if e.reference is not None)
        )
    return stats
