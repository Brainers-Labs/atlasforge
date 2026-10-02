import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from atlasforge.errors import DatasetError
from atlasforge.eval.validate import ValidationReport, validate_dataset

Rows = Sequence[Mapping[str, object] | str]


def write(tmp_path: Path, rows: Rows, name: str = "d.jsonl") -> Path:
    path = tmp_path / name
    body = "\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def codes(report: ValidationReport) -> set[str]:
    return {i.code for i in report.issues}


def gen(
    i: int, text: str | None = None, ref: str | None = None, **extra: object
) -> dict[str, object]:
    row: dict[str, object] = {"id": f"e{i}", "input": text or f"question number {i}", **extra}
    row["reference"] = ref or f"answer number {i}"
    return row


class TestCleanDataset:
    def test_no_issues(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(i) for i in range(10)]), "generation")
        assert report.ok
        assert report.issues == ()
        assert (report.n_lines, report.n_examples) == (10, 10)
        assert len(report.sha256) == 64

    def test_stats(self, tmp_path: Path) -> None:
        rows = [
            gen(0, "a b c", "x", lang="ha"),
            gen(1, "a b c d e", "x y", lang="ha"),
            gen(2, "a", "x y z", lang="yo"),
        ]
        report = validate_dataset(write(tmp_path, rows), "generation")
        assert report.stats["langs"] == {"ha": 2, "yo": 1}
        assert report.stats["input_words"] == {"min": 1, "median": 3, "max": 5}
        assert report.stats["reference_words"] == {"min": 1, "median": 2, "max": 3}

    def test_report_is_json_serialisable(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(0), gen(0)]), "generation")
        assert json.loads(json.dumps(report.to_dict()))["n_examples"] == 1


class TestLineErrors:
    def test_all_errors_collected_with_line_numbers_and_good_rows_still_counted(
        self, tmp_path: Path
    ) -> None:
        path = write(
            tmp_path, [gen(0), "{broken", gen(1), {"input": "x", "refrence": "typo"}, gen(2), "[1]"]
        )
        report = validate_dataset(path, "generation")
        errors = [i for i in report.issues if i.code == "line-error"]
        assert sorted(i.line or 0 for i in errors) == [2, 4, 6]
        assert report.n_examples == 3
        assert report.n_lines == 6
        assert not report.ok

    def test_line_number_is_not_repeated_in_the_message(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(0), "{broken"]), "generation")
        (error,) = [i for i in report.issues if i.code == "line-error"]
        assert error.line == 2
        assert not error.message.startswith("line ")

    def test_duplicate_id_reported(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(0), gen(0)]), "generation")
        assert any("duplicate id" in i.message for i in report.errors)

    def test_error_listing_is_capped(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, ["{bad"] * 80), "generation")
        line_errors = [i for i in report.issues if i.code == "line-error"]
        assert len(line_errors) == 51
        assert line_errors[-1].count == 30

    def test_empty_file(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [""]), "generation")
        assert "no-examples" in codes(report)
        assert not report.ok


class TestDuplicates:
    def test_exact_and_near_duplicates_flagged(self, tmp_path: Path) -> None:
        rows = [
            gen(0, "Ina kwana?", "Lafiya"),
            gen(1, "ina kwana", "lafiya!"),
            gen(2, "different", "other"),
        ]
        report = validate_dataset(write(tmp_path, rows), "generation")
        issue = next(i for i in report.issues if i.code == "duplicate-content")
        assert issue.count == 1
        assert "e1" in issue.message

    def test_tone_only_difference_counts_as_duplicate(self, tmp_path: Path) -> None:
        rows = [gen(0, "wà", "a"), gen(1, "wa", "a")]
        assert "duplicate-content" in codes(validate_dataset(write(tmp_path, rows), "generation"))

    def test_conflicting_references_flagged(self, tmp_path: Path) -> None:
        rows = [gen(0, "same question", "yes"), gen(1, "same question", "no")]
        report = validate_dataset(write(tmp_path, rows), "generation")
        assert "conflicting-references" in codes(report)
        assert "duplicate-content" not in codes(report)

    def test_asr_skips_duplicate_text_checks(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        rows = [
            {"id": str(i), "audio": "a.wav", "reference": "same words", "lang": "ha"}
            for i in range(3)
        ]
        assert "duplicate-content" not in codes(validate_dataset(write(tmp_path, rows), "asr"))


class TestUnicodeHealth:
    def test_non_nfc_flagged(self, tmp_path: Path) -> None:
        rows = [gen(0, "ẹ́ decomposed"), gen(1, "ẹ́ precomposed")]
        issue = next(
            i
            for i in validate_dataset(write(tmp_path, rows), "generation").issues
            if i.code == "non-nfc"
        )
        assert issue.count == 1
        assert "e0" in issue.message

    def test_replacement_character_flagged(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(0, "bad � char")]), "generation")
        assert "replacement-character" in codes(report)

    def test_control_character_flagged_but_tab_and_newline_allowed(self, tmp_path: Path) -> None:
        flagged = validate_dataset(write(tmp_path, [gen(0, "bell \x07 char")]), "generation")
        allowed = validate_dataset(
            write(tmp_path, [gen(0, "tab\tand\nnewline ok")], "b.jsonl"), "generation"
        )
        assert "control-characters" in codes(flagged)
        assert "control-characters" not in codes(allowed)

    def test_surrounding_whitespace_is_only_a_note(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(0, "  padded  ")]), "generation")
        issue = next(i for i in report.issues if i.code == "surrounding-whitespace")
        assert issue.severity == "info"
        assert report.ok

    def test_empty_after_normalisation(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, [gen(0, "!!!", "ok")]), "generation")
        assert "empty-input" in codes(report)

    def test_empty_reference_is_an_error_for_classification_only(self, tmp_path: Path) -> None:
        rows = [gen(0, "question", "...")]
        assert not validate_dataset(write(tmp_path, rows), "classification").ok
        assert validate_dataset(write(tmp_path, rows, "g.jsonl"), "generation").ok


class TestDiacriticStatistics:
    def texts(self, n: int, template: str, lang: str) -> list[dict[str, object]]:
        return [gen(i, template.format(i), template.format(i), lang=lang) for i in range(n)]

    def test_stripped_yoruba_is_flagged(self, tmp_path: Path) -> None:
        report = validate_dataset(
            write(tmp_path, self.texts(25, "bawo ni o se wa {}", "yo")), "generation"
        )
        messages = [i.message for i in report.issues if i.code == "diacritics-missing"]
        assert len(messages) == 2
        assert any("tone marks" in m for m in messages)
        assert any("underdotted" in m for m in messages)
        assert report.ok  # a heuristic: warning, never an error

    def test_diacritised_yoruba_is_clean(self, tmp_path: Path) -> None:
        text = "Ṣé o wà dáadáa ọjọ́ {}"
        report = validate_dataset(write(tmp_path, self.texts(25, text, "yo")), "generation")
        assert "diacritics-missing" not in codes(report)
        assert report.stats["diacritics"]["yo"]["tone_rate"] == 1.0
        assert report.stats["diacritics"]["yo"]["underdot_rate"] == 1.0

    def test_hausa_without_hooked_letters_flagged_and_with_them_clean(self, tmp_path: Path) -> None:
        stripped = validate_dataset(
            write(tmp_path, self.texts(25, "kasa dan bara {}", "ha")), "generation"
        )
        hooked = validate_dataset(
            write(tmp_path, self.texts(25, "ƙasa ɗan {}", "ha"), "b.jsonl"), "generation"
        )
        assert "diacritics-missing" in codes(stripped)
        assert "diacritics-missing" not in codes(hooked)

    def test_igbo_underdot(self, tmp_path: Path) -> None:
        stripped = validate_dataset(
            write(
                tmp_path, self.texts(25, "onye ọbịa {}".replace("ọ", "o").replace("ị", "i"), "ig")
            ),
            "generation",
        )
        assert "diacritics-missing" in codes(stripped)

    def test_too_few_texts_to_judge(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, self.texts(5, "bawo ni {}", "yo")), "generation")
        assert "diacritics-missing" not in codes(report)

    def test_english_and_undeclared_languages_are_ignored(self, tmp_path: Path) -> None:
        report = validate_dataset(
            write(tmp_path, self.texts(30, "plain english {}", "en")), "generation"
        )
        assert "diacritics-missing" not in codes(report)
        assert report.stats["diacritics"] == {}


class TestClassBalance:
    def rows(self, spec: dict[str, int]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for label, n in spec.items():
            rows += [
                {"id": f"{label}{i}", "input": f"{label} question {i}", "reference": label}
                for i in range(n)
            ]
        return rows

    def test_balanced_is_clean(self, tmp_path: Path) -> None:
        report = validate_dataset(
            write(tmp_path, self.rows({"pos": 20, "neg": 20})), "classification"
        )
        assert report.issues == ()
        assert report.stats["labels"] == {"pos": 20, "neg": 20}

    def test_majority_class_warning(self, tmp_path: Path) -> None:
        report = validate_dataset(
            write(tmp_path, self.rows({"pos": 90, "neg": 10})), "classification"
        )
        assert "class-imbalance" in codes(report)
        assert report.ok

    def test_tiny_minority_warning(self, tmp_path: Path) -> None:
        report = validate_dataset(
            write(tmp_path, self.rows({"a": 48, "b": 47, "c": 5})), "classification"
        )
        issue = next(i for i in report.issues if i.code == "class-imbalance")
        assert "'c'" in issue.message

    def test_single_class_warning(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, self.rows({"only": 10})), "classification")
        assert "single-class" in codes(report)

    def test_not_checked_for_generation(self, tmp_path: Path) -> None:
        report = validate_dataset(write(tmp_path, self.rows({"pos": 90, "neg": 10})), "generation")
        assert "class-imbalance" not in codes(report)


class TestLeakage:
    def test_overlap_between_train_and_test_is_an_error(self, tmp_path: Path) -> None:
        train = write(
            tmp_path,
            [
                gen(0, "Ina kwana?", "a"),
                gen(1, "shared question", "b"),
                gen(2, "unique train", "c"),
            ],
            "train.jsonl",
        )
        test = write(
            tmp_path,
            [
                gen(10, "shared question", "b"),
                gen(11, "INA KWANA", "a"),
                gen(12, "unique test", "d"),
            ],
            "test.jsonl",
        )
        report = validate_dataset(test, "generation", against=train)
        issue = next(i for i in report.errors if i.code == "train-test-leakage")
        assert issue.count == 2
        assert "1 identical, 1 differing" in issue.message
        assert "e10" in issue.message
        assert not report.ok

    def test_tone_only_difference_counts_as_leakage(self, tmp_path: Path) -> None:
        train = write(tmp_path, [gen(0, "wà")], "train.jsonl")
        test = write(tmp_path, [gen(1, "wa")], "test.jsonl")
        assert "train-test-leakage" in codes(validate_dataset(test, "generation", against=train))

    def test_disjoint_splits_are_clean(self, tmp_path: Path) -> None:
        train = write(tmp_path, [gen(0, "alpha")], "train.jsonl")
        test = write(tmp_path, [gen(1, "beta")], "test.jsonl")
        assert validate_dataset(test, "generation", against=train).ok

    def test_unparseable_lines_in_the_other_file_are_reported_not_fatal(
        self, tmp_path: Path
    ) -> None:
        train = write(tmp_path, [gen(0, "alpha"), "{bad"], "train.jsonl")
        test = write(tmp_path, [gen(1, "beta")], "test.jsonl")
        report = validate_dataset(test, "generation", against=train)
        assert "against-parse-errors" in codes(report)
        assert report.ok

    def test_asr_leakage_check_is_declared_unimplemented(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        row = {"id": "1", "audio": "a.wav", "reference": "x", "lang": "ha"}
        a, b = write(tmp_path, [row], "a.jsonl"), write(tmp_path, [row], "b.jsonl")
        report = validate_dataset(a, "asr", against=b)
        assert "leakage-skipped" in codes(report)
        assert report.ok


def test_ok_ignores_warnings_and_notes(tmp_path: Path) -> None:
    rows = [gen(0, "  padded  ", "x"), gen(1, "ẹ́", "y")]
    report = validate_dataset(write(tmp_path, rows), "generation")
    assert report.issues
    assert report.ok
    assert {i.severity for i in report.issues} <= {"warning", "info"}


def test_file_level_problems_still_raise(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read"):
        validate_dataset(tmp_path / "missing.jsonl", "generation")
