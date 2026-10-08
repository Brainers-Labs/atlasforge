import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from atlasforge.errors import ConfigError
from atlasforge.eval.dataset import Dataset, load_dataset
from atlasforge.eval.runner import Record
from atlasforge.eval.score import score_run, write_report_json


def dataset(
    tmp_path: Path, rows: Sequence[Mapping[str, object]], task: str = "generation"
) -> Dataset:
    path = tmp_path / "d.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return load_dataset(path, task)  # type: ignore[arg-type]


def results(**predictions: str | None) -> dict[str, Record]:
    """Build results; a value of None means the call failed."""
    return {
        i: Record(id=i, prediction=p, latency_ms=10.0)
        if p is not None
        else Record(id=i, error="BackendTimeout: x")
        for i, p in predictions.items()
    }


class TestToneViews:
    def test_tone_only_difference_splits_the_two_views(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "wà"}])
        report = score_run(ds, results(a="wa"), metrics=["exact_match"])
        assert report.metric("exact_match@tone_aware").mean == 0.0
        assert report.metric("exact_match@tone_insensitive").mean == 1.0

    def test_underdot_difference_is_never_forgiven(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "ṣe"}])
        report = score_run(ds, results(a="se"), metrics=["exact_match"])
        assert report.metric("exact_match@tone_aware").mean == 0.0
        assert report.metric("exact_match@tone_insensitive").mean == 0.0

    def test_unicode_form_difference_is_forgiven_in_both_views(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "ẹ́"}])
        report = score_run(ds, results(a="ẹ́"), metrics=["exact_match"])
        assert report.metric("exact_match@tone_aware").mean == 1.0

    def test_case_and_punctuation_ignored(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "Ina kwana!"}])
        report = score_run(ds, results(a="ina, kwana"), metrics=["exact_match"])
        assert report.metric("exact_match@tone_aware").mean == 1.0

    def test_both_views_reported_for_every_metric(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        report = score_run(ds, results(a="x"))
        keys = {s.key for s in report.metrics}
        assert keys == {
            "exact_match@tone_aware",
            "exact_match@tone_insensitive",
            "chrf@tone_aware",
            "chrf@tone_insensitive",
        }
        assert set(report.normalization) == {"tone_aware", "tone_insensitive"}


class TestFailuresCountAgainstTheModel:
    def test_failed_example_scores_as_wrong(self, tmp_path: Path) -> None:
        ds = dataset(
            tmp_path,
            [
                {"id": "a", "input": "q", "reference": "x"},
                {"id": "b", "input": "q", "reference": "y"},
            ],
        )
        report = score_run(ds, results(a="x", b=None), metrics=["exact_match"])
        assert (report.n_ok, report.n_failed, report.n_missing) == (1, 1, 0)
        assert report.metric("exact_match@tone_aware").mean == 0.5

    def test_missing_example_scores_as_wrong_and_is_counted(self, tmp_path: Path) -> None:
        ds = dataset(
            tmp_path,
            [
                {"id": "a", "input": "q", "reference": "x"},
                {"id": "b", "input": "q", "reference": "y"},
            ],
        )
        report = score_run(ds, results(a="x"), metrics=["exact_match"])
        assert (report.n_ok, report.n_failed, report.n_missing) == (1, 0, 1)
        assert report.metric("exact_match@tone_aware").mean == 0.5

    def test_failed_asr_example_is_all_deletions(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        ds = dataset(
            tmp_path, [{"id": "a", "audio": "a.wav", "reference": "ina kwana", "lang": "ha"}], "asr"
        )
        report = score_run(ds, results(a=None))
        assert report.metric("wer@tone_aware").corpus == 1.0
        assert report.metric("wer@tone_aware").mean == 1.0

    def test_all_failed_does_not_crash_chrf(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x y"}])
        report = score_run(ds, results(a=None), metrics=["chrf"])
        assert report.metric("chrf@tone_aware").corpus == 0.0


class TestTasks:
    def test_generation_defaults(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "ina kwana"}])
        report = score_run(ds, results(a="ina kwana"))
        assert report.metric("exact_match@tone_aware").mean == 1.0
        assert report.metric("chrf@tone_aware").mean == pytest.approx(100.0)
        assert report.metric("chrf@tone_aware").corpus == pytest.approx(100.0)
        assert report.metric("exact_match@tone_aware").corpus is None

    def test_classification_accuracy_and_macro_f1(self, tmp_path: Path) -> None:
        rows = [
            {"id": "1", "input": "q", "reference": "positive"},
            {"id": "2", "input": "q", "reference": "positive"},
            {"id": "3", "input": "q", "reference": "negative"},
            {"id": "4", "input": "q", "reference": "negative"},
        ]
        ds = dataset(tmp_path, rows, "classification")
        report = score_run(
            ds,
            results(**{"1": "Positive.", "2": "It is negative", "3": "negative", "4": "negative"}),
        )
        assert report.metric("accuracy@tone_aware").mean == 0.75
        # positive: tp1 fp0 fn1 -> 2/3.  negative: tp2 fp1 fn0 -> 4/5
        assert report.metric("macro_f1@tone_aware").corpus == pytest.approx((2 / 3 + 4 / 5) / 2)
        assert report.metric("macro_f1@tone_aware").mean is None

    def test_strict_labels_see_an_answer_that_contradicts_itself(self, tmp_path: Path) -> None:
        """G17: the loose match reads "not positive" as naming positive."""
        rows = [
            {"id": "1", "input": "q", "reference": "positive"},
            {"id": "2", "input": "q", "reference": "positive"},
        ]
        ds = dataset(tmp_path, rows, "classification")
        report = score_run(
            ds,
            results(**{"1": "Positive.", "2": "not positive"}),
            metrics=["accuracy", "accuracy_strict"],
        )
        # Loose: "not positive" contains the label the reference asked for, so it is a hit.
        assert report.metric("accuracy@tone_aware").mean == 1.0
        # Strict: the whole answer is not a label, so it is a miss.
        assert report.metric("accuracy_strict@tone_aware").mean == 0.5
        # The two are separate keys, so nothing can average them together by accident.
        assert report.metric("accuracy_strict@tone_insensitive").mean == 0.5
        assert report.metric("accuracy_strict@tone_aware").name == "accuracy_strict"

    def test_strict_macro_f1_is_pooled_only(self, tmp_path: Path) -> None:
        ds = dataset(
            tmp_path,
            [
                {"id": "1", "input": "q", "reference": "positive"},
                {"id": "2", "input": "q", "reference": "negative"},
            ],
            "classification",
        )
        report = score_run(
            ds,
            results(**{"1": "positive", "2": "It is negative"}),
            metrics=["macro_f1_strict"],
        )
        summary = report.metric("macro_f1_strict@tone_aware")
        assert summary.mean is None  # pooled only, as macro_f1
        assert summary.corpus == pytest.approx(0.5)

    def test_strict_metrics_need_classification_and_say_so(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        with pytest.raises(ConfigError, match="accuracy_strict needs a classification dataset"):
            score_run(ds, results(a="x"), metrics=["accuracy_strict"])

    def test_asr_wer_cer_pooled(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        rows = [
            {"id": "1", "audio": "a.wav", "reference": "ina kwana", "lang": "ha"},
            {"id": "2", "audio": "a.wav", "reference": "sannu da zuwa", "lang": "ha"},
        ]
        ds = dataset(tmp_path, rows, "asr")
        report = score_run(ds, results(**{"1": "ina kwana", "2": "sannu da zuwa yau"}))
        # 5 reference words, 1 insertion -> 0.2 pooled
        assert report.metric("wer@tone_aware").corpus == pytest.approx(0.2)
        assert report.metric("cer@tone_aware").corpus is not None

    def test_lower_is_better_flag(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        ds = dataset(
            tmp_path, [{"id": "1", "audio": "a.wav", "reference": "x", "lang": "ha"}], "asr"
        )
        report = score_run(ds, results(**{"1": "x"}))
        assert report.metric("wer@tone_aware").higher_is_better is False

    def test_examples_without_reference_are_not_scored(self, tmp_path: Path) -> None:
        ds = dataset(
            tmp_path, [{"id": "a", "input": "q"}, {"id": "b", "input": "q", "reference": "x"}]
        )
        report = score_run(ds, results(a="anything", b="x"), metrics=["exact_match"])
        summary = report.metric("exact_match@tone_aware")
        assert summary.n == 1
        assert summary.mean == 1.0
        assert "exact_match@tone_aware" not in report.per_example["a"]

    def test_no_references_at_all_gives_empty_metrics(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q"}])
        report = score_run(ds, results(a="x"), metrics=["exact_match"])
        summary = report.metric("exact_match@tone_aware")
        assert (summary.n, summary.mean, summary.corpus) == (0, None, None)


class TestValidation:
    def test_unknown_metric(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        with pytest.raises(ConfigError, match="Unknown metric") as info:
            score_run(ds, results(a="x"), metrics=["bleu9000"])
        assert info.value.hint is not None

    def test_accuracy_needs_classification(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        with pytest.raises(ConfigError, match="classification"):
            score_run(ds, results(a="x"), metrics=["accuracy"])


class TestLatencyAndPerExample:
    def test_latency_stats_use_successful_calls_only(self, tmp_path: Path) -> None:
        rows = [{"id": str(i), "input": "q", "reference": "x"} for i in range(4)]
        ds = dataset(tmp_path, rows)
        recs = {
            "0": Record(id="0", prediction="x", latency_ms=10.0),
            "1": Record(id="1", prediction="x", latency_ms=20.0),
            "2": Record(id="2", prediction="x", latency_ms=30.0),
            "3": Record(id="3", error="boom"),
        }
        report = score_run(ds, recs)
        assert report.latency_ms["mean"] == pytest.approx(20.0)
        assert report.latency_ms["p50"] == pytest.approx(20.0)
        assert report.latency_ms["max"] == 30.0

    def test_no_latency_data(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        report = score_run(ds, {"a": Record(id="a", error="boom")})
        assert report.latency_ms == {"mean": None, "p50": None, "p95": None, "max": None}

    def test_per_example_values_kept_for_paired_statistics(self, tmp_path: Path) -> None:
        ds = dataset(
            tmp_path,
            [
                {"id": "a", "input": "q", "reference": "x"},
                {"id": "b", "input": "q", "reference": "y"},
            ],
        )
        report = score_run(ds, results(a="x", b="z"), metrics=["exact_match"])
        assert report.per_example["a"]["exact_match@tone_aware"] == 1.0
        assert report.per_example["b"]["exact_match@tone_aware"] == 0.0

    def test_metric_lookup_miss(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "a", "input": "q", "reference": "x"}])
        with pytest.raises(KeyError):
            score_run(ds, results(a="x")).metric("nope@tone_aware")


class TestReportFile:
    def test_json_roundtrip_keeps_unicode_readable(self, tmp_path: Path) -> None:
        ds = dataset(tmp_path, [{"id": "ṣe", "input": "q", "reference": "x"}])
        report = score_run(ds, results(ṣe="x"))
        out = tmp_path / "report.json"
        write_report_json(report, out)
        raw = out.read_text(encoding="utf-8")
        assert "ṣe" in raw
        data = json.loads(raw)
        assert data["task"] == "generation"
        assert data["n_total"] == 1
        assert data["dataset_sha256"] == ds.sha256
