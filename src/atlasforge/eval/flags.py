"""Failure-mode flags: deterministic per-example signals, not a "hallucination rate".

Decision D023: no judge model and no invented quality score. What is here are plain rules
over the answer and the reference, each of which you can read, reproduce and argue with.
They are *flags* — places to look — never a verdict on the run.

Seven of them:

``empty_output``
    The call succeeded but the model said nothing.
``repeated_output``
    A word dominates the answer, or a four-word phrase appears three times or more.
``truncated_output``
    The reference ends in a full stop and the answer is a long sentence that does not,
    which is what a cut-off generation looks like.
``format_noncompliant``
    The example asked for JSON (``meta.format == "json"``), or for the answer to *be* a
    number (``meta.format == "number"``), or the reference is JSON, and the answer is not.
``missing_required_terms``
    An entity or acronym from the reference is absent from the answer. With
    ``meta.require`` you can name the terms yourself, which replaces the heuristic.
``number_mismatch``
    The set of numbers in the answer differs from the set in the reference.
``script_mismatch``
    The answer is written in a non-Latin script. This is **script statistics, not language
    identification** (D024): Hausa, Yoruba, Igbo and English all use Latin script, so a
    fluent answer in the wrong one of those four is invisible to this check and is not
    claimed otherwise.

The rules are deliberately conservative: a flag needs evidence in the text, and missing
evidence means no flag. Flags are computed for answers we actually received, so a failed call
is not counted here — the report already counts those separately. Classification runs have no
flags: a label is not prose.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Final

from atlasforge.eval.normalize import NormalizeConfig, normalize

if TYPE_CHECKING:
    from atlasforge.eval.dataset import Example

#: Every flag this module can raise, in the order reports list them.
FLAG_NAMES: Final[tuple[str, ...]] = (
    "empty_output",
    "repeated_output",
    "truncated_output",
    "format_noncompliant",
    "missing_required_terms",
    "number_mismatch",
    "script_mismatch",
)

#: What each flag means, for the report and the docs. One line each.
DESCRIPTIONS: Final[dict[str, str]] = {
    "empty_output": "the call succeeded but the answer was empty",
    "repeated_output": "a word or four-word phrase repeats until it fills the answer",
    "truncated_output": "a long answer stops where the reference ends in a full stop",
    "format_noncompliant": "the answer is not in the requested format",
    "missing_required_terms": "an entity or acronym from the reference is missing",
    "number_mismatch": "the answer's numbers differ from the reference's",
    "script_mismatch": "the answer is in a non-Latin script (script statistics, not language ID)",
}

#: Formats ``meta.format`` may ask for. Anything else is ignored rather than guessed at.
FORMATS: Final[frozenset[str]] = frozenset({"json", "number"})

_FLAGS_VIEW: Final = NormalizeConfig(lang=None, tones="strip")
_TERMINAL: Final = ".!?" + chr(0x2026)  # . ! ? and the ellipsis
_TRAILING: Final = "\"'" + chr(0x201D) + chr(0x2019) + ")]}"  # closing quotes and brackets
_WORD: Final = re.compile(r"\w+")
_ENTITY: Final = re.compile(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b")
_SENTENCE_START: Final = re.compile(r"(?:^|[.!?]\s+)$")
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")
_MIN_WORDS_FOR_TRUNCATION: Final = 5
_DOMINANT_SHARE: Final = 0.5
_NGRAM: Final = 4
_NGRAM_REPEATS: Final = 3
_MIN_LETTERS_FOR_SCRIPT: Final = 10
_NON_LATIN_SHARE: Final = 0.2


def flags_for(example: Example, prediction: str) -> tuple[str, ...]:
    """Every flag ``prediction`` earns against ``example``, in :data:`FLAG_NAMES` order."""
    if not prediction.strip():
        return ("empty_output",)
    reference = example.reference or ""
    answer = normalize(prediction, _FLAGS_VIEW)
    raised = {
        "repeated_output": _repeats(answer),
        "truncated_output": _truncated(reference, prediction),
        "format_noncompliant": not _format_ok(example, reference, prediction),
        "missing_required_terms": bool(_missing_terms(example, reference, answer)),
        "number_mismatch": _numbers(reference) != _numbers(prediction),
        "script_mismatch": _non_latin(prediction),
    }
    return tuple(name for name in FLAG_NAMES if raised.get(name))


def missing_terms(example: Example, prediction: str) -> tuple[str, ...]:
    """The required terms ``prediction`` does not contain, for the report to name them."""
    return _missing_terms(example, example.reference or "", normalize(prediction, _FLAGS_VIEW))


def summarise(per_example: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """Count each flag over ``per_example``, omitting flags nobody raised."""
    counts = Counter(name for flags in per_example.values() for name in flags)
    return {name: counts[name] for name in FLAG_NAMES if counts[name]}


def _repeats(answer: str) -> bool:
    words = _WORD.findall(answer)
    if len(words) >= 8:
        _, most = Counter(words).most_common(1)[0]
        if most / len(words) >= _DOMINANT_SHARE:
            return True
    if len(words) < _NGRAM * _NGRAM_REPEATS:
        return False
    grams = Counter(tuple(words[i : i + _NGRAM]) for i in range(len(words) - _NGRAM + 1))
    return max(grams.values()) >= _NGRAM_REPEATS


def _truncated(reference: str, prediction: str) -> bool:
    """A long answer with no final punctuation, where the reference has one."""
    if not reference.rstrip().endswith(tuple(_TERMINAL)):
        return False
    answer = prediction.rstrip().rstrip(_TRAILING).rstrip()
    if not answer or answer.endswith(tuple(_TERMINAL)):
        return False
    return len(_WORD.findall(normalize(prediction, _FLAGS_VIEW))) >= _MIN_WORDS_FOR_TRUNCATION


def _format_ok(example: Example, reference: str, prediction: str) -> bool:
    """Is the answer in the format the example asked for (or the format the reference is in)?"""
    asked = example.meta.get("format")
    wanted = asked if isinstance(asked, str) and asked in FORMATS else None
    if wanted is None and _is_json(reference):
        wanted = "json"
    if wanted == "json":
        return _is_json(prediction)
    if wanted == "number":
        return _NUMBER.fullmatch(prediction.strip().rstrip(".")) is not None
    return True


def _is_json(text: str) -> bool:
    stripped = text.strip()
    if stripped[:1] not in {"{", "["}:
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def _missing_terms(example: Example, reference: str, answer: str) -> tuple[str, ...]:
    """Terms the reference requires and the answer lacks: ``meta.require``, or entities."""
    declared = example.meta.get("require")
    if isinstance(declared, list) and all(isinstance(t, str) for t in declared):
        return tuple(t for t in declared if normalize(t, _FLAGS_VIEW) not in answer)
    missing = [e for e in _entities(reference) if normalize(e, _FLAGS_VIEW) not in answer]
    return tuple(dict.fromkeys(missing))


def _entities(reference: str) -> list[str]:
    """Names and acronyms in the reference: capitalised words that are not just a first word.

    ``The capital of Nigeria is Abuja.`` names Nigeria and Abuja. The initial ``The`` is
    capitalised because sentences are, so it is skipped — unless it is all caps, because
    ``NECO was founded ...`` is a name starting a sentence.
    """
    found: list[str] = []
    for match in _ENTITY.finditer(reference):
        word = match.group(0)
        starts_a_sentence = _SENTENCE_START.search(reference[: match.start()]) is not None
        if starts_a_sentence and not word.isupper():
            continue
        found.append(word)
    return found


def _numbers(text: str) -> set[str]:
    """The numbers in ``text``, ignoring thousands separators, as text (so 1.0 != 1)."""
    return set(_NUMBER.findall(re.sub(r"(?<=\d),(?=\d)", "", text)))


def _non_latin(text: str) -> bool:
    """Is a substantial share of the answer in a script that is not Latin?"""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < _MIN_LETTERS_FOR_SCRIPT:
        return False
    foreign = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, "LATIN"))
    return foreign / len(letters) >= _NON_LATIN_SHARE
