import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from atlasforge.compare import compare_runs, compare_scores, to_markdown
from atlasforge.compare.compare import POOLED_ONLY
from atlasforge.errors import ConfigError
from atlasforge.eval.dataset import Dataset, load_dataset
from atlasforge.eval.runner import Record, run
from atlasforge.eval.score import score_run
from atlasforge.types import (
    AudioInput,
    BackendInfo,
    Generation,
    GenParams,
    Lang,
    Message,
    Transcript,
)


class MapBackend:
    """Answers each prompt from a fixed mapping; unknown prompts get 'x'."""

    def __init__(self, answers: Mapping[str, str], *, model: str = "fake/model") -> None:
        self.answers = answers
        self.model = model

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        return Generation(text=self.answers.get(messages[-1]["content"], "x"), latency_ms=1.0)

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        return Transcript(text="", lang=lang)

    def info(self) -> BackendInfo:
        return BackendInfo(backend="fake", model=self.model, revision="rev1234567890")

    def close(self) -> None:
        return None


Rows = Sequence[Mapping[str, object]]


def write_dataset(tmp_path: Path, rows: Rows, task: str = "generation") -> Dataset:
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return load_dataset(path, task)  # type: ignore[arg-type]


def records(pairs: Mapping[str, str | None]) -> dict[str, Record]:
    return {
        i: Record(id=i, prediction=p, latency_ms=1.0)
        if p is not None
        else Record(id=i, error="boom")
        for i, p in pairs.items()
    }


def domain_dataset(tmp_path: Path) -> Dataset:
    rows = [
        {
            "id": f"e{i}",
            "input": f"q{i}",
            "reference": f"ans{i}",
            "meta": {"domain": "agri" if i < 40 else "legal"},
        }
        for i in range(80)
    ]
    return write_dataset(tmp_path, rows)


class TestImprovement:
    def test_clear_improvement_in_one_domain(self, tmp_path: Path) -> None:
        ds = domain_dataset(tmp_path)
        base = score_run(ds, records({f"e{i}": "x" for i in range(80)}), metrics=["exact_match"])
        cand = score_run(
            ds,
            records({f"e{i}": (f"ans{i}" if i < 40 else "x") for i in range(80)}),
            metrics=["exact_match"],
        )
        report = compare_scores(ds, base, cand, slice_fields=["domain"], n_boot=500)

        overall = next(m for m in report.metrics if m.key == "exact_match@tone_aware")
        assert overall.base_mean == 0.0
        assert overall.candidate_mean == 0.5
        assert overall.delta == pytest.approx(0.5)
        assert overall.verdict == "improved"
        assert (overall.wins, overall.ties, overall.losses) == (40, 40, 0)
        assert overall.mcnemar is not None
        assert overall.mcnemar.candidate_only == 40
        assert overall.mcnemar.p_value < 1e-6

        by_value = {s.value: s for s in report.slices}
        assert by_value["agri"].status == "improved"
        assert (
            by_value["legal"].status == "no clear change"
        )  # identical outputs: zero-width CI at 0
        assert report.regressed_metrics == ()
        assert report.regressed_slices == ()


class TestRegressionDetection:
    def numbers_dataset(self, tmp_path: Path) -> Dataset:
        rows = [{"id": f"n{i}", "input": f"qn{i}", "reference": "cost 50"} for i in range(40)]
        rows += [{"id": f"t{i}", "input": f"qt{i}", "reference": "hello world"} for i in range(40)]
        return write_dataset(tmp_path, rows)

    def test_hidden_regression_on_numbers_is_surfaced(self, tmp_path: Path) -> None:
        ds = self.numbers_dataset(tmp_path)
        base = score_run(
            ds, records({e.id: e.reference or "" for e in ds.examples}), metrics=["exact_match"]
        )
        cand_preds = {
            e.id: ("x" if e.id.startswith("n") else e.reference or "") for e in ds.examples
        }
        cand = score_run(ds, records(cand_preds), metrics=["exact_match"])

        report = compare_scores(ds, base, cand, slice_fields=["has_number"], n_boot=500)
        regressed = report.regressed_slices
        assert [(s.field, s.value) for s in regressed] == [("has_number", "yes")]
        assert regressed[0].delta == pytest.approx(-1.0)
        assert report.regressed_metrics  # and the overall number went down too

        text = to_markdown(report)
        assert "## Regressions" in text
        assert "has_number = yes" in text


class TestSmallSlicesAreNotOverClaimed:
    def test_tiny_slice_is_flagged_insufficient_in_the_report(self, tmp_path: Path) -> None:
        rows = [
            {"id": f"a{i}", "input": f"qa{i}", "reference": "r", "meta": {"g": "big"}}
            for i in range(40)
        ]
        rows += [
            {"id": f"b{i}", "input": f"qb{i}", "reference": "r", "meta": {"g": "tiny"}}
            for i in range(5)
        ]
        ds = write_dataset(tmp_path, rows)
        base = score_run(ds, records({e.id: "x" for e in ds.examples}), metrics=["exact_match"])
        cand = score_run(ds, records({e.id: "r" for e in ds.examples}), metrics=["exact_match"])
        report = compare_scores(ds, base, cand, slice_fields=["g"], n_boot=200)
        tiny = next(s for s in report.slices if s.value == "tiny")
        assert tiny.status == "insufficient data"
        assert "insufficient data" in to_markdown(report)


class TestDirection:
    def test_lower_wer_counts_as_improvement(self, tmp_path: Path) -> None:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        rows = [
            {"id": f"u{i}", "audio": "a.wav", "reference": "ina kwana yau", "lang": "ha"}
            for i in range(40)
        ]
        ds = write_dataset(tmp_path, rows, "asr")
        base = score_run(ds, records({e.id: "ina" for e in ds.examples}))
        cand = score_run(ds, records({e.id: "ina kwana yau" for e in ds.examples}))
        report = compare_scores(ds, base, cand, n_boot=200)
        wer = next(m for m in report.metrics if m.key == "wer@tone_aware")
        assert wer.higher_is_better is False
        assert wer.delta is not None
        assert wer.delta < 0
        assert wer.verdict == "improved"
        assert wer.base_corpus == pytest.approx(2 / 3)
        assert wer.candidate_corpus == 0.0


class TestPooledMetrics:
    def test_macro_f1_is_reported_but_not_tested(self, tmp_path: Path) -> None:
        rows = [
            {"id": f"c{i}", "input": f"q{i}", "reference": "pos" if i % 2 else "neg"}
            for i in range(40)
        ]
        ds = write_dataset(tmp_path, rows, "classification")
        base = score_run(ds, records({e.id: "pos" for e in ds.examples}))
        cand = score_run(ds, records({e.id: e.reference or "" for e in ds.examples}))
        report = compare_scores(ds, base, cand, n_boot=200)
        f1 = next(m for m in report.metrics if m.key == "macro_f1@tone_aware")
        assert f1.verdict == POOLED_ONLY
        assert f1.candidate_corpus == 1.0
        assert f1.base_corpus is not None
        assert f1.base_corpus < 1.0
        assert "pooled metric" in to_markdown(report)
        # accuracy is per-example, so it is tested and is the primary metric
        assert report.primary == "accuracy@tone_aware"


class TestRunDirectories:
    def make_runs(self, tmp_path: Path) -> tuple[Dataset, Path, Path]:
        ds = domain_dataset(tmp_path)
        gold = {f"q{i}": f"ans{i}" for i in range(80)}
        half = {k: v for k, v in gold.items() if int(k[1:]) < 40}
        run(MapBackend({}, model="base/m"), ds, tmp_path / "base")
        run(MapBackend(half, model="cand/m"), ds, tmp_path / "cand")
        return ds, tmp_path / "base", tmp_path / "cand"

    def test_end_to_end_from_directories(self, tmp_path: Path) -> None:
        ds, base, cand = self.make_runs(tmp_path)
        report = compare_runs(ds, base, cand, slice_fields=["domain"], n_boot=300)
        assert report.base.model == "base/m"
        assert report.candidate.model == "cand/m"
        assert report.candidate.revision == "rev1234567890"
        assert report.primary == "exact_match@tone_aware"
        text = to_markdown(report)
        assert "base/m" in text
        assert "cand/m" in text
        assert "`rev1234567` " in text or "`rev1234567`" in text
        assert "## Method" in text

    def test_refuses_runs_from_a_different_dataset(self, tmp_path: Path) -> None:
        _, base, cand = self.make_runs(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        other = write_dataset(elsewhere, [{"id": "z", "input": "z", "reference": "z"}])
        with pytest.raises(ConfigError, match="not produced from this dataset"):
            compare_runs(other, base, cand)

    def test_not_a_run_directory(self, tmp_path: Path) -> None:
        ds = domain_dataset(tmp_path)
        with pytest.raises(ConfigError, match="not a run directory"):
            compare_runs(ds, tmp_path / "nope", tmp_path / "nope2")

    def test_scores_from_different_datasets_rejected(self, tmp_path: Path) -> None:
        a = domain_dataset(tmp_path)
        b = write_dataset(tmp_path, [{"id": "z", "input": "z", "reference": "z"}])
        sa = score_run(a, {}, metrics=["exact_match"])
        sb = score_run(b, {}, metrics=["exact_match"])
        with pytest.raises(ConfigError, match="different datasets"):
            compare_scores(a, sa, sb)


class TestPrimary:
    def test_explicit_primary_is_respected(self, tmp_path: Path) -> None:
        ds = domain_dataset(tmp_path)
        s = score_run(ds, records({e.id: "x" for e in ds.examples}))
        report = compare_scores(
            ds, s, s, primary="chrf@tone_insensitive", slice_fields=[], n_boot=100
        )
        assert report.primary == "chrf@tone_insensitive"

    def test_unknown_primary_is_rejected_with_choices(self, tmp_path: Path) -> None:
        ds = domain_dataset(tmp_path)
        s = score_run(ds, records({e.id: "x" for e in ds.examples}))
        with pytest.raises(ConfigError, match="no per-example values") as info:
            compare_scores(ds, s, s, primary="bleu@tone_aware", n_boot=100)
        assert info.value.hint is not None
        assert "exact_match@tone_aware" in info.value.hint

    def test_no_references_means_nothing_to_compare(self, tmp_path: Path) -> None:
        ds = write_dataset(tmp_path, [{"id": "a", "input": "q"}])
        s = score_run(ds, records({"a": "x"}), metrics=["exact_match"])
        report = compare_scores(ds, s, s, n_boot=100)
        assert report.primary is None
        assert report.slices == ()


class TestMarkdown:
    def test_report_has_all_sections_and_both_views(self, tmp_path: Path) -> None:
        ds = domain_dataset(tmp_path)
        base = score_run(
            ds, records({e.id: "x" for e in ds.examples}), metrics=["exact_match", "chrf"]
        )
        cand = score_run(
            ds,
            records({e.id: (e.reference or "") for e in ds.examples}),
            metrics=["exact_match", "chrf"],
        )
        report = compare_scores(ds, base, cand, slice_fields=["domain", "lang"], n_boot=200)
        text = to_markdown(report)
        for heading in (
            "# AtlasForge comparison report",
            "## Overall",
            "## Regressions",
            "## Slices",
            "## Method",
        ):
            assert heading in text
        assert "tone-aware" in text
        assert "tone-insensitive" in text
        assert "+100.0 pts" in text  # exact match 0% -> 100%
        assert "No statistically clear regressions found." in text
        assert "McNemar" in text
        assert text.endswith("\n")

    def test_json_dict_is_serialisable(self, tmp_path: Path) -> None:
        ds = domain_dataset(tmp_path)
        s = score_run(ds, records({e.id: "x" for e in ds.examples}), metrics=["exact_match"])
        report = compare_scores(ds, s, s, slice_fields=["domain"], n_boot=100)
        data = json.loads(json.dumps(report.to_dict()))
        assert data["task"] == "generation"
        assert data["settings"]["n_boot"] == 100
