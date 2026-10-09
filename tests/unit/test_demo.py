"""The demo is documentation made executable, so it is tested like a feature."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atlasforge.cli import app
from atlasforge.compare import compare_runs, to_markdown
from atlasforge.demo import DATASET_NAME, build_demo, build_items
from atlasforge.errors import ConfigError
from atlasforge.eval.dataset import load_dataset
from atlasforge.eval.runner import read_results
from atlasforge.eval.score import score_run
from atlasforge.eval.validate import validate_dataset

runner = CliRunner()


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_demo(tmp_path_factory.mktemp("demo") / "d")


def comparison(demo_dir: Path) -> object:
    dataset = load_dataset(demo_dir / DATASET_NAME, "generation")
    return compare_runs(
        dataset,
        demo_dir / "runs" / "base",
        demo_dir / "runs" / "tuned",
        metrics=["exact_match"],
        slice_fields=["domain", "lang", "has_number", "length"],
        n_boot=500,
    )


class TestShape:
    def test_item_counts(self) -> None:
        items = build_items()
        assert len(items) == 92
        by_domain = {
            d: sum(i.domain == d for i in items) for d in ("agri", "numeracy", "greetings")
        }
        assert by_domain == {"agri": 40, "numeracy": 40, "greetings": 12}

    def test_every_prompt_and_id_is_unique(self) -> None:
        items = build_items()
        assert len({i.prompt for i in items}) == len(items)
        assert len({i.id for i in items}) == len(items)

    def test_the_dataset_passes_validation_without_warnings(self, demo_dir: Path) -> None:
        report = validate_dataset(demo_dir / DATASET_NAME, "generation")
        assert report.ok
        assert report.issues == ()

    def test_runs_are_marked_synthetic(self, demo_dir: Path) -> None:
        for which in ("base", "tuned"):
            manifest = json.loads(
                (demo_dir / "runs" / which / "run.json").read_text(encoding="utf-8")
            )
            assert manifest["model"] == f"synthetic-demo/{which}"
            assert manifest["backend"] == "synthetic"
            assert manifest["atlasforge_version"] == "sample-data"
            assert manifest["started_at"].startswith("2026-01-01")


class TestItDemonstratesEveryFeature:
    def test_the_base_run_contains_counted_failures(self, demo_dir: Path) -> None:
        dataset = load_dataset(demo_dir / DATASET_NAME, "generation")
        report = score_run(dataset, read_results(demo_dir / "runs" / "base" / "results.jsonl"))
        assert report.n_failed == 2
        assert (
            score_run(dataset, read_results(demo_dir / "runs" / "tuned" / "results.jsonl")).n_failed
            == 0
        )

    def test_tone_views_disagree_for_the_base_model_on_yoruba(self, demo_dir: Path) -> None:
        dataset = load_dataset(demo_dir / DATASET_NAME, "generation")
        report = score_run(
            dataset,
            read_results(demo_dir / "runs" / "base" / "results.jsonl"),
            metrics=["exact_match"],
        )
        yoruba = [e.id for e in dataset.examples if e.lang == "yo"]
        aware = [report.per_example[i]["exact_match@tone_aware"] for i in yoruba]
        insensitive = [report.per_example[i]["exact_match@tone_insensitive"] for i in yoruba]
        assert sum(aware) == 0
        assert sum(insensitive) == 12

    def test_overall_improvement_hidden_regression_and_a_slice_too_small_to_judge(
        self, demo_dir: Path
    ) -> None:
        result = comparison(demo_dir)
        overall = next(m for m in result.metrics if m.key == "exact_match@tone_aware")  # type: ignore[attr-defined]
        assert overall.verdict == "improved"
        status = {(s.field, s.value): s.status for s in result.slices}  # type: ignore[attr-defined]
        assert status[("domain", "agri")] == "improved"
        assert status[("domain", "numeracy")] == "regressed"
        assert status[("has_number", "yes")] == "regressed"
        assert status[("lang", "yo")] == "insufficient data"

    def test_the_markdown_report_tells_the_story(self, demo_dir: Path) -> None:
        text = to_markdown(comparison(demo_dir))  # type: ignore[arg-type]
        assert "## Regressions" in text
        assert "domain = numeracy" in text
        assert "insufficient data" in text
        assert "synthetic-demo/base" in text


class TestDeterminism:
    def test_two_builds_are_byte_identical(self, tmp_path: Path) -> None:
        a, b = build_demo(tmp_path / "a"), build_demo(tmp_path / "b")
        for relative in (
            DATASET_NAME,
            "runs/base/results.jsonl",
            "runs/tuned/results.jsonl",
            "runs/base/run.json",
        ):
            assert (a / relative).read_bytes() == (b / relative).read_bytes(), relative

    def test_refuses_a_non_empty_folder(self, tmp_path: Path) -> None:
        (tmp_path / "keep.txt").write_text("mine", encoding="utf-8")
        with pytest.raises(ConfigError, match="not empty"):
            build_demo(tmp_path)

    def test_an_existing_empty_folder_is_fine(self, tmp_path: Path) -> None:
        assert (build_demo(tmp_path) / DATASET_NAME).is_file()


class TestCliWorkflow:
    """The exact commands the quickstart tells a reader to run."""

    def test_demo_then_the_three_commands(self, tmp_path: Path) -> None:
        target = tmp_path / "try-it"
        made = runner.invoke(app, ["demo", str(target)])
        assert made.exit_code == 0, made.output
        assert "NOT real N-ATLaS output" in made.output
        data = target / DATASET_NAME
        base, tuned = target / "runs" / "base", target / "runs" / "tuned"

        assert runner.invoke(app, ["dataset", "validate", str(data)]).exit_code == 0
        report = runner.invoke(app, ["report", str(base), "--dataset", str(data)])
        assert report.exit_code == 0, report.output
        assert (base / "report.md").is_file()

        out = tmp_path / "cmp"
        cmp = runner.invoke(
            app,
            [
                "compare",
                str(data),
                "--base",
                str(base),
                "--candidate",
                str(tuned),
                "--slice",
                "domain",
                "--out",
                str(out),
            ],
        )
        assert cmp.exit_code == 0, cmp.output
        assert "Regressions" in cmp.output
        assert "domain = numeracy" in cmp.output
        assert (out / "comparison.md").is_file()

    def test_demo_default_folder_name_and_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert runner.invoke(app, ["demo"]).exit_code == 0
        assert (tmp_path / "atlasforge-demo" / DATASET_NAME).is_file()
        again = runner.invoke(app, ["demo"])
        assert isinstance(again.exception, ConfigError)
