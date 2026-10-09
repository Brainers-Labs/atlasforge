"""Failure-mode flags: deterministic rules over the answer, and where they surface."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from atlasforge.compare.compare import compare_scores
from atlasforge.compare.report import to_markdown as comparison_markdown
from atlasforge.eval.dataset import Dataset, Example, load_dataset
from atlasforge.eval.flags import (
    DESCRIPTIONS,
    FLAG_NAMES,
    flags_for,
    missing_terms,
    summarise,
)
from atlasforge.eval.report import to_markdown
from atlasforge.eval.runner import Record
from atlasforge.eval.score import score_run
from atlasforge.types import Lang

NO_FLAGS: tuple[str, ...] = ()


def example(
    reference: str | None,
    *,
    meta: Mapping[str, object] | None = None,
    lang: Lang | None = "en",
    id_: str = "a",
) -> Example:
    return Example(
        id=id_,
        messages=(),
        reference=reference,
        lang=lang,
        meta=meta if meta is not None else {},
    )


def dataset(tmp_path: Path, rows: Sequence[Mapping[str, object]]) -> Dataset:
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return load_dataset(path, "generation")


def results(**predictions: str | None) -> dict[str, Record]:
    """Build results; a value of None means the call failed."""
    return {
        i: Record(id=i, prediction=p, latency_ms=10.0)
        if p is not None
        else Record(id=i, error="BackendTimeout: x")
        for i, p in predictions.items()
    }


class TestEmptyOutput:
    def test_blank_answers_are_flagged(self) -> None:
        assert flags_for(example("anything"), "") == ("empty_output",)
        assert flags_for(example("anything"), "   \n ") == ("empty_output",)

    def test_an_empty_answer_is_flagged_once_not_seven_times(self) -> None:
        """The one certain flag, and no guessing about what else is missing."""
        assert flags_for(example("The capital of Nigeria is Abuja."), "") == ("empty_output",)

    def test_a_whitespace_only_answer_is_scored_under_both_views_too(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        report = score_run(ds, results(a="  "))
        assert report.flags == {"empty_output": 1}
        assert report.metric("exact_match@tone_aware").mean == 0.0


class TestRepeatedOutput:
    def test_one_word_filling_the_answer_is_flagged(self) -> None:
        assert "repeated_output" in flags_for(example("x"), "yes yes yes yes yes yes yes yes")

    def test_a_phrase_repeated_three_times_is_flagged(self) -> None:
        answer = " ".join(["the answer is that"] * 3) + " so we move on to other things"
        assert "repeated_output" in flags_for(example("x"), answer)

    def test_ordinary_prose_is_not_flagged(self) -> None:
        answer = "Ina kwana, yaya kake? I am well, thank you for asking about my family today."
        assert "repeated_output" not in flags_for(example("x"), answer)

    def test_a_word_repeated_twice_in_a_short_answer_is_not_flagged(self) -> None:
        assert "repeated_output" not in flags_for(example("x"), "very very good")

    def test_four_words_repeated_twice_is_not_enough(self) -> None:
        assert "repeated_output" not in flags_for(
            example("x"), "one two three four one two three four"
        )


class TestTruncatedOutput:
    def test_a_long_answer_without_a_final_stop_is_flagged(self) -> None:
        reference = "The capital of Nigeria is Abuja."
        answer = "The capital of Nigeria is Abuja and it is a large"
        assert "truncated_output" in flags_for(example(reference), answer)

    def test_a_finished_answer_is_not_flagged(self) -> None:
        reference = "The capital of Nigeria is Abuja."
        assert "truncated_output" not in flags_for(example(reference), reference)

    def test_a_reference_without_a_final_stop_proves_nothing(self) -> None:
        answer = "i think the answer is probably abuja in the federal capital territory"
        assert "truncated_output" not in flags_for(example("abuja"), answer)

    def test_a_short_answer_is_not_called_truncated(self) -> None:
        assert "truncated_output" not in flags_for(example("47."), "forty seven")


class TestFormatCompliance:
    def test_json_asked_for_and_not_given_is_flagged(self) -> None:
        ex = example("anything", meta={"format": "json"})
        assert "format_noncompliant" in flags_for(ex, "sure, here you go")
        assert "format_noncompliant" not in flags_for(ex, '{"answer": "ok"}')

    def test_a_number_asked_for_and_not_given_is_flagged(self) -> None:
        ex = example("anything", meta={"format": "number"})
        assert "format_noncompliant" in flags_for(ex, "The answer is 47 words long")
        assert "format_noncompliant" not in flags_for(ex, " 47 ")
        assert "format_noncompliant" not in flags_for(ex, "47.")

    def test_a_json_reference_sets_the_expectation_by_itself(self) -> None:
        ex = example('{"sentiment": "positive"}')
        assert "format_noncompliant" in flags_for(ex, "positive")
        assert "format_noncompliant" not in flags_for(ex, '{"sentiment": "positive"}')

    def test_no_format_asked_for_means_no_format_flag(self) -> None:
        assert "format_noncompliant" not in flags_for(example("positive"), "negative")

    def test_an_unknown_format_is_ignored_rather_than_guessed_at(self) -> None:
        ex = example("x", meta={"format": "yaml-ish-whatever"})
        assert "format_noncompliant" not in flags_for(ex, "x")


class TestMissingRequiredTerms:
    def test_an_entity_from_the_reference_that_is_absent_flags(self) -> None:
        ex = example("The capital of Nigeria is Abuja.")
        assert "missing_required_terms" in flags_for(ex, "Abuja.")
        assert "missing_required_terms" not in flags_for(ex, "the capital of nigeria is abuja")

    def test_the_missing_terms_are_named_for_the_reader(self) -> None:
        ex = example("The capital of Nigeria is Abuja.")
        assert missing_terms(ex, "Abuja.") == ("Nigeria",)

    def test_meta_require_names_the_terms_yourself(self) -> None:
        ex = example("take the medicine", meta={"require": ["mg", "dosage"]})
        assert missing_terms(ex, "take 5 mg twice a day") == ("dosage",)
        assert missing_terms(ex, "take 5 mg, dosage as prescribed") == ()

    def test_declared_terms_are_matched_the_tone_insensitive_way(self) -> None:
        """A Yoruba term typed without tone marks is the same term, not a missing one."""
        ex = example("Ẹ ṣé", meta={"require": ["Ẹ ṣé"]})
        assert missing_terms(ex, "Ẹ ṣe") == ()

    def test_underdots_are_not_forgiven_in_a_required_term(self) -> None:
        """The same rule the metrics follow: tones may be dropped, ẹ/ọ/ṣ may not."""
        ex = example("Ẹ ṣé", meta={"require": ["Ẹ ṣé"]})
        assert missing_terms(ex, "E se") == ("Ẹ ṣé",)

    def test_lower_case_prose_has_no_entities_to_require(self) -> None:
        ex = example("the answer is a fruit")
        assert "missing_required_terms" not in flags_for(ex, "vegetable")

    def test_a_capitalised_first_word_alone_is_not_an_entity(self) -> None:
        ex = example("Mango is a fruit.")
        assert "missing_required_terms" not in flags_for(ex, "a fruit")


class TestNumberMismatch:
    def test_a_different_number_is_flagged(self) -> None:
        ex = example("47")
        assert "number_mismatch" in flags_for(ex, "The answer is 48")
        assert "number_mismatch" not in flags_for(ex, "The answer is 47")

    def test_thousands_separators_are_not_a_difference(self) -> None:
        assert "number_mismatch" not in flags_for(example("1000 naira"), "1,000 naira")

    def test_a_decimal_is_a_different_number_from_its_truncation(self) -> None:
        assert "number_mismatch" in flags_for(example("3.5 kg"), "3 kg")

    def test_an_answer_with_no_numbers_is_flagged_against_a_number_reference(self) -> None:
        assert "number_mismatch" in flags_for(example("47"), "forty seven")

    def test_two_numbers_in_either_order_are_the_same_set(self) -> None:
        assert "number_mismatch" not in flags_for(example("2 + 3"), "3 and 2")


class TestScriptMismatch:
    def test_an_answer_in_another_script_is_flagged(self) -> None:
        cyrillic = "Это ответ на русском языке, и он довольно длинный."
        assert "script_mismatch" in flags_for(example("yes"), cyrillic)

    def test_latin_script_answers_are_never_flagged(self) -> None:
        yoruba = "Ẹ káàárọ̀, ṣé dáadáa ni? Mo wà dáadáa, ẹ ṣé púpọ̀."
        assert "script_mismatch" not in flags_for(example(yoruba, lang="yo"), yoruba)

    def test_a_stray_foreign_character_in_a_long_answer_is_not_flagged(self) -> None:
        answer = "The quick brown fox jumps over the lazy dog and then says 犬 to everyone."
        assert "script_mismatch" not in flags_for(example("x"), answer)

    def test_a_short_answer_is_not_judged_on_script(self) -> None:
        assert "script_mismatch" not in flags_for(example("x"), "да")

    def test_the_flag_does_not_claim_language_identification(self) -> None:
        """D024: no language-ID claims. The wording has to say what the check really is."""
        assert "language ID" in DESCRIPTIONS["script_mismatch"]
        assert "script" in DESCRIPTIONS["script_mismatch"]


class TestShape:
    def test_flags_are_always_in_the_documented_order(self) -> None:
        ex = example("The capital of Nigeria is 47.")
        raised = flags_for(ex, "да")
        assert raised == tuple(name for name in FLAG_NAMES if name in raised)

    def test_every_flag_has_a_description(self) -> None:
        assert set(DESCRIPTIONS) == set(FLAG_NAMES)

    def test_the_same_input_always_gives_the_same_flags(self) -> None:
        ex = example("The capital of Nigeria is Abuja.")
        assert flags_for(ex, "Abuja in 1960.") == flags_for(ex, "Abuja in 1960.")

    def test_summarise_counts_each_flag_and_hides_the_zeros(self) -> None:
        per_example = {
            "a": ("number_mismatch",),
            "b": ("number_mismatch", "empty_output"),
            "c": NO_FLAGS,
        }
        assert summarise(per_example) == {"empty_output": 1, "number_mismatch": 2}

    def test_summarise_of_nothing_is_empty(self) -> None:
        assert summarise({}) == {}


class TestInTheScoreReport:
    def test_flags_are_counted_and_attributed(self, tmp_path: Path) -> None:
        ds = dataset(
            tmp_path,
            [
                {"id": "a", "input": "q", "reference": "47"},
                {"id": "b", "input": "q", "reference": "47"},
                {"id": "c", "input": "q", "reference": "x"},
            ],
        )
        report = score_run(ds, results(a="48", b="47", c=""))
        assert report.flags == {"empty_output": 1, "number_mismatch": 1}
        assert report.n_flagged == 2
        assert report.per_example_flags == {"a": ["number_mismatch"], "c": ["empty_output"]}

    def test_a_failed_call_is_counted_as_a_failure_not_as_a_flag(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        report = score_run(ds, results(a=None))
        assert report.n_failed == 1
        assert report.flags == {}
        assert report.n_flagged == 0
        assert report.per_example_flags == {}

    def test_an_example_with_no_reference_is_not_flagged(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q"}])
        report = score_run(ds, results(a="anything at all"))
        assert report.flags == {}

    def test_classification_labels_are_not_prose_and_get_no_flags(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text(
            json.dumps({"id": "a", "input": "q", "reference": "fruit"}) + "\n", encoding="utf-8"
        )
        report = score_run(load_dataset(path, "classification"), results(a="vegetable"))
        assert report.flags == {}
        assert report.n_flagged == 0

    def test_the_report_json_carries_the_per_example_flags(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "47"}])
        report = score_run(ds, results(a="48"))
        written = json.loads(json.dumps(report.to_dict()))
        assert written["flags"] == {"number_mismatch": 1}
        assert written["per_example_flags"] == {"a": ["number_mismatch"]}

    def test_the_markdown_says_a_flag_is_not_a_quality_score(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "47"}])
        text = to_markdown(score_run(ds, results(a="48")))
        assert "## Failure-mode flags" in text
        assert "not a hallucination rate" in text
        assert "| `number_mismatch` | 1 |" in text

    def test_a_clean_run_says_so_explicitly(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "47"}])
        text = to_markdown(score_run(ds, results(a="47")))
        assert "No flags on any of the 1 answer(s)" in text

    def test_a_classification_report_has_no_flag_section(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text(
            json.dumps({"id": "a", "input": "q", "reference": "fruit"}) + "\n", encoding="utf-8"
        )
        text = to_markdown(score_run(load_dataset(path, "classification"), results(a="fruit")))
        assert "Failure-mode flags" not in text


class TestInTheComparison:
    def two_runs(self, tmp_path: Path) -> tuple[Dataset, dict[str, Record], dict[str, Record]]:
        ds = dataset(
            tmp_path,
            [
                {"id": "a", "input": "q", "reference": "47"},
                {"id": "b", "input": "q", "reference": "the capital of Nigeria is Abuja."},
            ],
        )
        base = results(a="48", b="the capital of Nigeria is Abuja.")
        candidate = results(a="47", b="Abuja.")
        return ds, base, candidate

    def test_the_comparison_counts_each_flag_in_both_runs(self, tmp_path: Path) -> None:
        ds, base, candidate = self.two_runs(tmp_path)
        report = compare_scores(ds, score_run(ds, base), score_run(ds, candidate))
        assert [(f.name, f.base, f.candidate, f.delta) for f in report.flags] == [
            ("missing_required_terms", 0, 1, 1),
            ("number_mismatch", 1, 0, -1),
        ]

    def test_the_run_table_carries_the_flagged_count(self, tmp_path: Path) -> None:
        ds, base, candidate = self.two_runs(tmp_path)
        report = compare_scores(ds, score_run(ds, base), score_run(ds, candidate))
        assert report.base.n_flagged == 1
        assert report.candidate.n_flagged == 1

    def test_a_new_flag_names_where_to_look_next(self, tmp_path: Path) -> None:
        ds, base, candidate = self.two_runs(tmp_path)
        report = compare_scores(ds, score_run(ds, base), score_run(ds, candidate))
        assert [f.name for f in report.new_flags] == ["missing_required_terms"]

    def test_the_markdown_shows_the_flag_table(self, tmp_path: Path) -> None:
        ds, base, candidate = self.two_runs(tmp_path)
        text = comparison_markdown(
            compare_scores(ds, score_run(ds, base), score_run(ds, candidate))
        )
        assert "## Failure-mode flags" in text
        assert "| `number_mismatch` | 1 | 0 | -1 |" in text
        assert "not a hallucination rate" in text

    def test_no_flags_in_either_run_means_no_section(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "47"}])
        report = compare_scores(ds, score_run(ds, results(a="47")), score_run(ds, results(a="47")))
        assert report.flags == ()
        assert "Failure-mode flags" not in comparison_markdown(report)

    def test_the_comparison_json_carries_the_flags(self, tmp_path: Path) -> None:
        ds, base, candidate = self.two_runs(tmp_path)
        report = compare_scores(ds, score_run(ds, base), score_run(ds, candidate))
        written = json.loads(json.dumps(report.to_dict()))
        assert written["flags"][0]["name"] == "missing_required_terms"
        assert written["base"]["n_flagged"] == 1
