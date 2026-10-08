"""ASR error analysis: what the model heard instead, and where it goes wrong.

A WER number says how much was wrong. This says *what*: the word pairs it substituted, the words
it dropped, the words it invented, and how the rate moves with utterance length.

The alignment is jiwer's, over the same normalised text the WER metric is computed on, so the
counts here add up to the rates in the report. Substitutions are marked ``tone_only`` when the two
words differ only in tone marks — which for Hausa, Yoruba and Igbo is the difference between a
mis-hearing and a *typing* convention, and the whole reason this tool scores both views.

Nothing here calls a model or opens an audio file: it is arithmetic over transcripts we already
have, which is why it lives in ``eval`` rather than in ``atlasforge.asr`` — importing that package
pulls in numpy and the audio stack, and scoring a report should not need either.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import jiwer
from jiwer.transforms import Compose, ReduceToListOfListOfWords

from atlasforge.eval.normalize import NormalizeConfig, normalize

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Identity transform: the text arrives already normalised, so jiwer must not touch it again.
_WORDS: Final = Compose([ReduceToListOfListOfWords()])

#: Tone marks only, for deciding whether a substitution was a tone slipping.
_NO_TONES: Final = NormalizeConfig(lang=None, tones="strip")

#: How many entries each top-errors list keeps, and how many examples each entry names.
TOP_N: Final = 10
EXAMPLES_PER_ERROR: Final = 3

#: Reference-length buckets, in words. Utterance length is the honest proxy we have offline:
#: decoding audio to measure seconds would drag ffmpeg into every `report` run.
LENGTH_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("1-5", 1, 5),
    ("6-15", 6, 15),
    ("16-30", 16, 30),
    ("31+", 31, 1 << 30),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TopError:
    """One repeated mistake. ``reference`` is ``None`` for an insertion, ``hypothesis`` for a deletion."""

    reference: str | None
    hypothesis: str | None
    count: int
    tone_only: bool
    examples: tuple[str, ...]

    def describe(self) -> str:
        """The mistake as a reader sees it, e.g. ``káàárọ̀ → kaaro``."""
        if self.reference is None:
            return f"(nothing) → {self.hypothesis}"
        if self.hypothesis is None:
            return f"{self.reference} → (dropped)"
        return f"{self.reference} → {self.hypothesis}"


@dataclass(frozen=True, slots=True, kw_only=True)
class LengthBucket:
    """Pooled WER for utterances whose reference is this long. ``wer`` is ``None`` if empty."""

    label: str
    n: int
    wer: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class AsrAnalysis:
    """Alignment-based error analysis over the utterances that had a reference."""

    n: int
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    tone_only_substitutions: int
    top_substitutions: tuple[TopError, ...]
    top_deletions: tuple[TopError, ...]
    top_insertions: tuple[TopError, ...]
    by_length: tuple[LengthBucket, ...]

    @property
    def errors(self) -> int:
        """Substitutions + deletions + insertions — the numerator of a pooled WER."""
        return self.substitutions + self.deletions + self.insertions


def analyse(pairs: Sequence[tuple[str, str, str]]) -> AsrAnalysis | None:
    """Analyse ``(example id, reference, hypothesis)`` triples. ``None`` if none can be analysed.

    Utterances whose reference has no words are skipped, exactly as the pooled WER skips them:
    there is no length to divide by, so a rate would be a fiction.
    """
    usable = [(id_, ref, hyp) for id_, ref, hyp in pairs if ref.split()]
    if not usable:
        return None
    ids = [id_ for id_, _, _ in usable]
    references = [ref for _, ref, _ in usable]
    hypotheses = [hyp for _, _, hyp in usable]
    output = jiwer.process_words(
        references, hypotheses, reference_transform=_WORDS, hypothesis_transform=_WORDS
    )

    substitutions: dict[tuple[str, str], list[str]] = defaultdict(list)
    deletions: dict[str, list[str]] = defaultdict(list)
    insertions: dict[str, list[str]] = defaultdict(list)
    tone_only_total = 0
    bucket_errors: dict[str, int] = defaultdict(int)
    bucket_words: dict[str, int] = defaultdict(int)
    bucket_n: dict[str, int] = defaultdict(int)
    totals = {"sub": 0, "del": 0, "ins": 0}

    for index, chunks in enumerate(output.alignments):
        example_id = ids[index]
        reference_words = output.references[index]
        hypothesis_words = output.hypotheses[index]
        bucket = _bucket_for(len(reference_words))
        bucket_n[bucket] += 1
        bucket_words[bucket] += len(reference_words)
        for chunk in chunks:
            ref_span = reference_words[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_span = hypothesis_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
            if chunk.type == "substitute":
                for ref_word, hyp_word in zip(ref_span, hyp_span, strict=False):
                    substitutions[(ref_word, hyp_word)].append(example_id)
                    tone_only_total += _tone_only(ref_word, hyp_word)
                    totals["sub"] += 1
            elif chunk.type == "delete":
                for ref_word in ref_span:
                    deletions[ref_word].append(example_id)
                    totals["del"] += 1
            elif chunk.type == "insert":
                for hyp_word in hyp_span:
                    insertions[hyp_word].append(example_id)
                    totals["ins"] += 1
        bucket_errors[bucket] += _chunk_errors(chunks)

    return AsrAnalysis(
        n=len(usable),
        hits=output.hits,
        substitutions=totals["sub"],
        deletions=totals["del"],
        insertions=totals["ins"],
        tone_only_substitutions=tone_only_total,
        top_substitutions=_top_pairs(substitutions),
        top_deletions=_top_singles(deletions, side="reference"),
        top_insertions=_top_singles(insertions, side="hypothesis"),
        by_length=tuple(
            LengthBucket(
                label=label,
                n=bucket_n[label],
                wer=(bucket_errors[label] / bucket_words[label]) if bucket_words[label] else None,
            )
            for label, _, _ in LENGTH_BUCKETS
            if bucket_n[label]
        ),
    )


def _chunk_errors(chunks: Sequence[jiwer.AlignmentChunk]) -> int:
    """Errors in one utterance: every aligned word that is not a match."""
    errors = 0
    for chunk in chunks:
        if chunk.type == "equal":
            continue
        ref_span = chunk.ref_end_idx - chunk.ref_start_idx
        hyp_span = chunk.hyp_end_idx - chunk.hyp_start_idx
        errors += max(ref_span, hyp_span)  # a 1-to-2 substitution is two errors
    return errors


def _tone_only(reference_word: str, hypothesis_word: str) -> int:
    """1 when the two words differ only in tone marks — the same word, heard differently."""
    if reference_word == hypothesis_word:
        return 0
    return int(normalize(reference_word, _NO_TONES) == normalize(hypothesis_word, _NO_TONES))


def _bucket_for(words: int) -> str:
    for label, low, high in LENGTH_BUCKETS:
        if low <= words <= high:
            return label
    return LENGTH_BUCKETS[-1][0]


def _top_pairs(counts: dict[tuple[str, str], list[str]]) -> tuple[TopError, ...]:
    ranked = sorted(counts.items(), key=lambda item: (-len(item[1]), item[0]))
    return tuple(
        TopError(
            reference=ref,
            hypothesis=hyp,
            count=len(ids),
            tone_only=bool(_tone_only(ref, hyp)),
            examples=tuple(dict.fromkeys(ids))[:EXAMPLES_PER_ERROR],
        )
        for (ref, hyp), ids in ranked[:TOP_N]
    )


def _top_singles(counts: dict[str, list[str]], *, side: str) -> tuple[TopError, ...]:
    ranked = sorted(counts.items(), key=lambda item: (-len(item[1]), item[0]))
    return tuple(
        TopError(
            reference=word if side == "reference" else None,
            hypothesis=word if side == "hypothesis" else None,
            count=len(ids),
            tone_only=False,
            examples=tuple(dict.fromkeys(ids))[:EXAMPLES_PER_ERROR],
        )
        for word, ids in ranked[:TOP_N]
    )
