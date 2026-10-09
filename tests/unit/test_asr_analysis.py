"""ASR error analysis: the counts must agree with the WER the report prints."""

import json
import wave
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from atlasforge.eval.asr_analysis import LENGTH_BUCKETS, TOP_N, analyse
from atlasforge.eval.dataset import Dataset, Example, load_dataset
from atlasforge.eval.report import to_markdown
from atlasforge.eval.runner import Record
from atlasforge.eval.score import score_run


def silent_wav(path: Path) -> Path:
    """The smallest real audio file the dataset loader will accept."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)
    return path


def asr_dataset(tmp_path: Path, rows: Sequence[Mapping[str, object]]) -> Dataset:
    lines = []
    for index, row in enumerate(rows):
        audio = silent_wav(tmp_path / f"a{index}.wav")
        lines.append(json.dumps({**row, "audio": audio.name}, ensure_ascii=False))
    path = tmp_path / "asr.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_dataset(path, "asr")


def results(**predictions: str | None) -> dict[str, Record]:
    return {
        i: Record(id=i, prediction=p, latency_ms=5.0)
        if p is not None
        else Record(id=i, error="BackendTimeout: x")
        for i, p in predictions.items()
    }


class TestAlignment:
    def test_an_exact_transcript_has_no_errors(self) -> None:
        analysis = analyse([("a", "ina kwana", "ina kwana")])
        assert analysis is not None
        assert (analysis.hits, analysis.errors) == (2, 0)
        assert analysis.top_substitutions == ()

    def test_a_substituted_word_is_named_on_both_sides(self) -> None:
        analysis = analyse([("a", "ina kwana", "ina kwanda")])
        assert analysis is not None
        assert analysis.substitutions == 1
        assert analysis.top_substitutions[0].reference == "kwana"
        assert analysis.top_substitutions[0].hypothesis == "kwanda"
        assert analysis.top_substitutions[0].count == 1

    def test_a_dropped_word_is_a_deletion_and_an_extra_word_is_an_insertion(self) -> None:
        # Both pairs are one unambiguous edit apart. Swapping both words ("kwana sosai") would
        # tie with two substitutions, and jiwer breaks that tie by substituting.
        analysis = analyse([("a", "ina kwana", "kwana"), ("b", "ina kwana", "ina kwana sosai")])
        assert analysis is not None
        assert (analysis.substitutions, analysis.deletions, analysis.insertions) == (0, 1, 1)
        assert analysis.top_deletions[0].reference == "ina"
        assert analysis.top_deletions[0].hypothesis is None
        assert analysis.top_insertions[0].hypothesis == "sosai"
        assert analysis.top_insertions[0].reference is None

    def test_the_counts_add_up_to_the_pooled_word_error_rate(self) -> None:
        """The section is only useful if its arithmetic is the WER's arithmetic. The inserted
        word is the case that separates the two: it is an error, but not a reference word."""
        pairs = [
            ("a", "ina kwana", "ina kwanda"),  # substituted
            ("b", "sannu da zuwa", "sannu zuwa"),  # dropped
            ("c", "yaya kake", "yaya kake sosai"),  # inserted
            ("d", "ina kwana", "ina kwana"),  # correct
        ]
        analysis = analyse(pairs)
        assert analysis is not None
        report = score_run(asr_dataset_for(pairs), results(**{i: h for i, _, h in pairs}))
        pooled = report.metric("wer@tone_aware").corpus
        reference_words = analysis.hits + analysis.substitutions + analysis.deletions
        assert pooled == pytest.approx(analysis.errors / reference_words)
        assert analysis.insertions == 1
        assert f"{pooled:.1%}" in to_markdown(report)

    def test_a_repeated_mistake_is_counted_and_names_where_it_happened(self) -> None:
        pairs = [("a", "ina kwana", "ina kwanda"), ("b", "ina kwana sosai", "ina kwanda sosai")]
        analysis = analyse(pairs)
        assert analysis is not None
        top = analysis.top_substitutions[0]
        assert (top.count, top.examples) == (2, ("a", "b"))

    def test_only_the_top_errors_are_kept(self) -> None:
        pairs = [(f"e{i}", f"word{i} x", "y x") for i in range(TOP_N + 5)]
        analysis = analyse(pairs)
        assert analysis is not None
        assert len(analysis.top_substitutions) == TOP_N

    def test_an_empty_hypothesis_is_all_deletions(self) -> None:
        analysis = analyse([("a", "ina kwana", "")])
        assert analysis is not None
        assert (analysis.hits, analysis.deletions) == (0, 2)

    def test_nothing_to_analyse_is_none_rather_than_an_empty_section(self) -> None:
        assert analyse([]) is None
        assert analyse([("a", "", "anything")]) is None

    def test_the_same_pairs_always_give_the_same_order(self) -> None:
        pairs = [("a", "one two", "x y"), ("b", "three four", "x y")]
        first, second = analyse(pairs), analyse(pairs)
        assert first is not None
        assert second is not None
        assert first.top_substitutions == second.top_substitutions


class TestToneOnlySubstitutions:
    def test_a_tone_slip_is_marked_as_tone_only(self) -> None:
        analysis = analyse([("a", "iná kwana", "ina kwana")])
        assert analysis is not None
        assert analysis.substitutions == 1
        assert analysis.tone_only_substitutions == 1
        assert analysis.top_substitutions[0].tone_only is True

    def test_a_real_mishearing_is_not_marked_tone_only(self) -> None:
        analysis = analyse([("a", "ina kwana", "ina kwanda")])
        assert analysis is not None
        assert analysis.tone_only_substitutions == 0
        assert analysis.top_substitutions[0].tone_only is False

    def test_an_underdot_is_not_a_tone_slip_even_beside_one(self) -> None:
        """The rule the metrics use: tones may go, ẹ/ọ/ṣ may not — only one of these two counts."""
        analysis = analyse([("a", "Ẹ ṣé", "E ṣe")])
        assert analysis is not None
        assert analysis.substitutions == 2
        assert analysis.tone_only_substitutions == 1
        assert {e.describe(): e.tone_only for e in analysis.top_substitutions} == {
            "ṣé → ṣe": True,
            "Ẹ → E": False,
        }


class TestLengthBuckets:
    def test_every_utterance_lands_in_a_bucket(self) -> None:
        pairs = [
            ("a", "one", "one"),
            ("b", " ".join(f"w{i}" for i in range(8)), "x"),
            ("c", " ".join(f"w{i}" for i in range(20)), "x"),
            ("d", " ".join(f"w{i}" for i in range(40)), "x"),
        ]
        analysis = analyse(pairs)
        assert analysis is not None
        assert [b.label for b in analysis.by_length] == [label for label, _, _ in LENGTH_BUCKETS]
        assert [b.n for b in analysis.by_length] == [1, 1, 1, 1]

    def test_empty_buckets_are_left_out(self) -> None:
        analysis = analyse([("a", "one two", "one two")])
        assert analysis is not None
        assert [b.label for b in analysis.by_length] == ["1-5"]

    def test_the_bucket_rate_is_pooled_within_the_bucket(self) -> None:
        pairs = [("a", "one two three four five", "one two three four five"), ("b", "six", "bad")]
        analysis = analyse(pairs)
        assert analysis is not None
        bucket = analysis.by_length[0]
        assert bucket.n == 2
        assert bucket.wer == pytest.approx(1 / 6)

    def test_word_error_rate_rises_with_length_here_by_construction(self) -> None:
        pairs = [("a", "one two three four five", "one two three four five"), ("b", "x " * 40, "y")]
        analysis = analyse(pairs)
        assert analysis is not None
        short = next(b for b in analysis.by_length if b.label == "1-5")
        long = next(b for b in analysis.by_length if b.label == "31+")
        assert short.wer == 0.0
        assert long.wer == 1.0


class TestInTheReport:
    def test_an_asr_run_gets_an_error_analysis_section(self, tmp_path: Path) -> None:
        ds = asr_dataset(
            tmp_path,
            [{"id": "a", "reference": "iná kwana"}, {"id": "b", "reference": "ina kwana"}],
        )
        report = score_run(ds, results(a="ina kwana", b="ina kwanda"))
        assert report.asr is not None
        text = to_markdown(report)
        assert "## ASR error analysis" in text
        assert "Most substituted" in text
        assert "differ only in tone marks" in text
        assert "| iná → ina | 1 | yes | `a` |" in text
        assert "Word error rate by reference length" in text

    def test_a_generation_run_gets_no_asr_section(self, tmp_path: Path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text(
            json.dumps({"id": "a", "input": "q", "reference": "x"}) + "\n", encoding="utf-8"
        )
        report = score_run(load_dataset(path, "generation"), results(a="y"))
        assert report.asr is None
        assert "ASR error analysis" not in to_markdown(report)

    def test_the_report_json_carries_the_analysis(self, tmp_path: Path) -> None:
        ds = asr_dataset(tmp_path, [{"id": "a", "reference": "ina kwana"}])
        written = json.loads(json.dumps(score_run(ds, results(a="ina kwanda")).to_dict()))
        assert written["asr"]["substitutions"] == 1
        assert written["asr"]["top_substitutions"][0]["reference"] == "kwana"
        assert written["asr"]["by_length"][0]["label"] == "1-5"

    def test_a_substitution_containing_a_pipe_does_not_break_the_table(
        self, tmp_path: Path
    ) -> None:
        """Normalisation turns `|` into a space, so no scored run can carry one into the table.
        The analysis is built directly to hold the escape to its job anyway."""
        ds = asr_dataset(tmp_path, [{"id": "a", "reference": "x"}])
        analysis = analyse([("a", "x|y", "z")])
        assert analysis is not None
        text = to_markdown(replace(score_run(ds, results(a="x")), asr=analysis))
        assert "x\\|y" in text

    def test_a_reference_with_no_words_is_left_out_like_the_pooled_wer_does(
        self, tmp_path: Path
    ) -> None:
        """A punctuation-only reference normalises to nothing: no words, so no rate."""
        ds = asr_dataset(tmp_path, [{"id": "a", "reference": "..."}])
        report = score_run(ds, results(a="..."))
        assert report.asr is None
        assert "ASR error analysis" not in to_markdown(report)


def asr_dataset_for(pairs: Sequence[tuple[str, str, str]]) -> Dataset:
    """A tiny in-memory ASR dataset, for the one test that checks against the pooled WER."""
    examples = tuple(
        Example(id=id_, reference=reference, audio=Path("a.wav"), lang="ha")
        for id_, reference, _ in pairs
    )
    return Dataset(task="asr", path=Path("asr.jsonl"), sha256="0" * 64, examples=examples)
