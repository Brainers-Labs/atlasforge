"""The AfroBench wrapper, exercised against stand-ins for the harness and its output.

Nothing here needs ``lm-evaluation-harness``, a GPU or the gated weights: every test injects
the harness's task list, its exit status and the file it "wrote". That is the point of the
module's shape, and it is the only way this path can be tested at all in CI — see
``docs/help/status.md``.

The figures in these fixtures are made up. Nothing here asserts a benchmark *result*; the
tests pin the wiring that turns whatever the harness reports into an AtlasForge report.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from atlasforge import bench
from atlasforge.cli import app
from atlasforge.errors import ConfigError, ResourceError

if TYPE_CHECKING:
    from collections.abc import Sequence

runner = CliRunner()

#: Shaped like the harness's own file: numbers, strings mixed in, aggregation suffixes.
PAYLOAD: dict[str, Any] = {
    "lm_eval_version": "0.4.13",
    "results": {
        "belebele_hau": {"acc,none": 0.42, "acc_stderr,none": 0.02, "alias": "belebele_hau"},
        "afrixnli_yo": {"acc_norm,none": 0.33, "acc,none": 0.31},
        "afrimgsm_ha": {"exact_match,none": 0.05},
        "mmlu": {"acc,none": 0.6},
    },
}


def payload_text(**overrides: Any) -> str:
    return json.dumps({**PAYLOAD, **overrides})


def write_results(command: Sequence[str], *, at: Path | None = None, status: int = 0) -> int:
    """Stand-in for the harness: writes a results file for the tasks it was asked about.

    Only the requested tasks are reported, because that is what the real harness does — a
    stand-in that reported a fixed set would let a task-selection bug pass unnoticed.
    """
    if status != 0:
        return status
    argv = list(command)
    target = at or Path(argv[argv.index("--output_path") + 1])
    asked = argv[argv.index("--tasks") + 1].split(",")
    block = PAYLOAD["results"]
    assert isinstance(block, dict)
    payload = {
        "lm_eval_version": PAYLOAD["lm_eval_version"],
        "results": {name: block[name] for name in asked if name in block},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return 0


class TestDiscoverTasks:
    def test_finds_every_language_variant_of_a_family(self) -> None:
        available = ("afrixnli_yo", "afrixnli_ha", "belebele_hau", "mmlu", "hellaswag")
        found = bench.discover_tasks(available)
        assert "afrixnli_yo" in found
        assert "afrixnli_ha" in found
        assert "belebele_hau" in found

    def test_leaves_out_everything_outside_the_families(self) -> None:
        assert bench.discover_tasks(("mmlu", "hellaswag", "gsm8k")) == ()

    def test_is_sorted_so_the_report_does_not_depend_on_harness_order(self) -> None:
        one = bench.discover_tasks(("belebele_hau", "afrixnli_yo"))
        two = bench.discover_tasks(("afrixnli_yo", "belebele_hau"))
        assert one == two == ("afrixnli_yo", "belebele_hau")

    def test_duplicates_collapse(self) -> None:
        assert bench.discover_tasks(("flores_200_eng", "flores_200_eng")) == ("flores_200_eng",)

    def test_matching_ignores_case_and_punctuation_style(self) -> None:
        assert bench.discover_tasks(("AfriXNLI-YO",)) == ("AfriXNLI-YO",)

    def test_a_custom_family_list_is_honoured(self) -> None:
        found = bench.discover_tasks(("mmlu", "gsm8k"), families=("gsm8k",))
        assert found == ("gsm8k",)


class TestMissingFamilies:
    def test_names_the_families_with_no_match(self) -> None:
        missing = bench.missing_families(("afrixnli_yo", "belebele_hau"))
        assert "afrimmlu" in missing
        assert "afrixnli" not in missing

    def test_is_empty_when_every_family_is_covered(self) -> None:
        found = bench.discover_tasks(("afrixnli_yo", "belebele_hau", "afrimmlu_ha", "flores_200"))
        found += ("sib_200", "injongo_xn", "afrimgsm_ha")
        assert bench.missing_families(found) == ()

    def test_a_gap_is_absent_not_zero(self) -> None:
        """The whole point: a family nobody ran must never read as a family that scored 0."""
        report = bench.BenchReport(model="m", revision=None, tasks=(), missing_families=("sib",))
        assert "not zero" in bench.report_markdown(report)


class TestBuildCommand:
    def base(self, **overrides: Any) -> list[str]:
        values: dict[str, Any] = {
            "model": "NCAIR1/N-ATLaS",
            "tasks": ["afrixnli_yo"],
            "output_path": Path("out/harness/results.json"),
        }
        values.update(overrides)
        return bench.build_command(**values)

    def test_runs_the_module_rather_than_the_console_script(self) -> None:
        command = self.base()
        assert command[1:3] == ["-m", "lm_eval"]
        assert command[0] == sys.executable

    def test_names_the_model_backend_and_the_task_list(self) -> None:
        command = self.base(tasks=["afrixnli_yo", "belebele_hau"])
        assert command[command.index("--model") + 1] == "hf"
        assert command[command.index("--tasks") + 1] == "afrixnli_yo,belebele_hau"

    def test_model_args_carry_the_repo_and_trust_remote_code(self) -> None:
        args = self.base()[self.base().index("--model_args") + 1]
        assert "pretrained=NCAIR1/N-ATLaS" in args
        assert "trust_remote_code=True" in args

    def test_a_revision_is_pinned_only_when_asked_for(self) -> None:
        assert "revision=" not in self.base()[self.base().index("--model_args") + 1]
        args = self.base(revision="abc123")[self.base().index("--model_args") + 1]
        assert "revision=abc123" in args

    def test_optional_switches_are_absent_by_default(self) -> None:
        command = self.base()
        assert "--num_fewshot" not in command
        assert "--device" not in command
        assert "--apply_chat_template" not in command

    def test_optional_switches_appear_when_given(self) -> None:
        command = self.base(few_shot=5, device="cuda:0", apply_chat_template=True)
        assert command[command.index("--num_fewshot") + 1] == "5"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--apply_chat_template" in command

    def test_trust_remote_code_can_be_turned_off(self) -> None:
        args = self.base(trust_remote_code=False)[self.base().index("--model_args") + 1]
        assert "trust_remote_code" not in args

    def test_no_tasks_is_refused_with_a_hint(self) -> None:
        with pytest.raises(ConfigError, match="No benchmark tasks") as caught:
            self.base(tasks=[])
        assert "--tasks" in (caught.value.hint or "")

    def test_a_blank_revision_is_refused_rather_than_silently_dropped(self) -> None:
        with pytest.raises(ConfigError, match="revision"):
            self.base(revision="   ")


class TestParseResults:
    def test_reads_every_task_the_harness_reported(self) -> None:
        report = bench.parse_results(payload_text(), model="m")
        assert [t.task for t in report.tasks] == [
            "afrimgsm_ha",
            "afrixnli_yo",
            "belebele_hau",
            "mmlu",
        ]

    def test_keeps_only_numbers(self) -> None:
        report = bench.parse_results(payload_text(), model="m")
        task = next(t for t in report.tasks if t.task == "belebele_hau")
        assert task.metrics == {"acc,none": 0.42, "acc_stderr,none": 0.02}

    def test_prefers_acc_norm_over_acc(self) -> None:
        report = bench.parse_results(payload_text(), model="m")
        task = next(t for t in report.tasks if t.task == "afrixnli_yo")
        assert task.primary_metric == "acc_norm,none"
        assert task.primary == 0.33

    def test_the_aggregation_suffix_does_not_hide_a_preferred_metric(self) -> None:
        report = bench.parse_results(payload_text(), model="m")
        task = next(t for t in report.tasks if t.task == "mmlu")
        assert task.primary_metric == "acc,none"

    def test_falls_back_to_the_first_metric_it_does_not_know(self) -> None:
        text = json.dumps({"results": {"t": {"zed": 0.2, "alpha": 0.9}}})
        task = bench.parse_results(text, model="m").tasks[0]
        assert (task.primary_metric, task.primary) == ("alpha", 0.9)

    def test_a_task_with_no_numbers_gets_a_dash_not_a_zero(self) -> None:
        text = json.dumps({"results": {"t": {"alias": "t"}}})
        task = bench.parse_results(text, model="m").tasks[0]
        assert (task.primary_metric, task.primary, task.metrics) == (None, None, {})

    def test_booleans_are_not_numbers(self) -> None:
        text = json.dumps({"results": {"t": {"acc": True}}})
        assert bench.parse_results(text, model="m").tasks[0].metrics == {}

    def test_the_harness_version_is_recorded(self) -> None:
        assert bench.parse_results(payload_text(), model="m").harness_version == "0.4.13"

    def test_a_missing_version_is_none_rather_than_invented(self) -> None:
        text = json.dumps({"results": {}})
        assert bench.parse_results(text, model="m").harness_version is None

    def test_the_model_and_revision_travel_with_the_report(self) -> None:
        report = bench.parse_results(payload_text(), model="NCAIR1/N-ATLaS", revision="abc")
        assert (report.model, report.revision, report.suite) == (
            "NCAIR1/N-ATLaS",
            "abc",
            bench.SUITE,
        )

    def test_output_that_is_not_json_says_so(self) -> None:
        with pytest.raises(ConfigError, match="not JSON"):
            bench.parse_results("Traceback (most recent call last)", model="m")

    def test_json_without_results_says_so(self) -> None:
        with pytest.raises(ConfigError, match="no 'results' object") as caught:
            bench.parse_results('{"config": {}}', model="m")
        # The hint must name something real: an earlier draft said `--results`, and no such flag
        # exists on this CLI, so following it produced a second error instead of a fix.
        assert "results.json" in (caught.value.hint or "")
        assert "--results" not in (caught.value.hint or "")

    def test_a_block_that_is_not_an_object_is_skipped_not_crashed(self) -> None:
        text = json.dumps({"results": {"t": "oops"}})
        assert bench.parse_results(text, model="m").tasks[0].metrics == {}


class TestFmtMetric:
    def test_no_value_is_a_dash(self) -> None:
        assert bench.fmt_metric("acc,none", None) == "-"
        assert bench.fmt_metric(None, None) == "-"

    @pytest.mark.parametrize(
        "metric", ["acc,none", "acc_norm,none", "exact_match,none", "f1,none", "accuracy,none"]
    )
    def test_accuracy_like_metrics_read_as_percentages(self, metric: str) -> None:
        assert bench.fmt_metric(metric, 0.5) == "50.00"

    @pytest.mark.parametrize("metric", ["chrf,none", "chrf++", "bleu,none"])
    def test_metrics_already_on_a_0_100_scale_are_left_alone(self, metric: str) -> None:
        """chrF 38.2 must not print as 3820.00 — the bug the first real run produced."""
        assert bench.fmt_metric(metric, 38.2) == "38.20"

    def test_the_lookup_ignores_case_and_the_aggregation_suffix(self) -> None:
        assert bench.fmt_metric("ACC,NONE", 0.5) == "50.00"


class TestReportMarkdown:
    def report(self) -> bench.BenchReport:
        task = bench.TaskResult(
            task="afrixnli_yo",
            metrics={"acc,none": 0.5, "acc_stderr,none": 0.02},
            primary_metric="acc,none",
            primary=0.5,
        )
        return bench.BenchReport(
            model="NCAIR1/N-ATLaS",
            revision="abc",
            tasks=(task,),
            harness_version="0.4.13",
            missing_families=("sib",),
        )

    def test_names_the_suite_model_revision_and_harness(self) -> None:
        text = bench.report_markdown(self.report())
        assert text.startswith(f"# {bench.SUITE}\n")
        assert "`NCAIR1/N-ATLaS`" in text
        assert "`abc`" in text
        assert "lm_evaluation_harness 0.4.13" in text

    def test_points_at_the_raw_harness_output_beside_it(self) -> None:
        assert f"`{bench.RAW_NAME}`" in bench.report_markdown(self.report())

    def test_names_the_file_the_harness_actually_wrote(self) -> None:
        """The report must not name a path it only asked for -- the harness may not honour it."""
        report = bench.BenchReport(
            model="m",
            revision=None,
            tasks=(),
            raw_path="harness/results_2026-10-07.json",
        )
        assert "`harness/results_2026-10-07.json`" in bench.report_markdown(report)

    def test_an_unpinned_revision_reads_as_default(self) -> None:
        report = bench.BenchReport(model="m", revision=None, tasks=())
        assert "Revision: `default`" in bench.report_markdown(report)

    def test_an_unknown_harness_version_says_unknown(self) -> None:
        report = bench.BenchReport(model="m", revision=None, tasks=())
        assert "lm_evaluation_harness unknown" in bench.report_markdown(report)

    def test_the_headline_value_is_a_percentage(self) -> None:
        assert "| `afrixnli_yo` | `acc,none` | 50.00 |" in bench.report_markdown(self.report())

    def test_the_other_metrics_are_listed_separately(self) -> None:
        text = bench.report_markdown(self.report())
        assert "## Every metric the harness reported" in text
        assert "| `afrixnli_yo` | `acc_stderr,none` | 0.02 |" in text

    def test_the_headline_metric_is_not_repeated_in_the_appendix(self) -> None:
        appendix = bench.report_markdown(self.report()).split("## Every metric", 1)[1]
        assert "acc,none" not in appendix

    def test_a_report_with_no_tasks_says_so_instead_of_empty_table(self) -> None:
        report = bench.BenchReport(model="m", revision=None, tasks=())
        text = bench.report_markdown(report)
        assert "No task reported a result." in text
        assert "| Task |" not in text

    def test_a_task_without_a_figure_gets_a_dash(self) -> None:
        task = bench.TaskResult(task="t", metrics={}, primary_metric=None, primary=None)
        report = bench.BenchReport(model="m", revision=None, tasks=(task,))
        assert "| `t` | `-` | - |" in bench.report_markdown(report)

    def test_a_translation_score_is_not_scaled_into_a_percentage(self) -> None:
        """One table holds both a multiple-choice and a translation task; only one scales."""
        task = bench.TaskResult(
            task="flores_200_eng",
            metrics={"chrf,none": 38.2},
            primary_metric="chrf,none",
            primary=38.2,
        )
        report = bench.BenchReport(model="m", revision=None, tasks=(task,))
        text = bench.report_markdown(report)
        assert "| `flores_200_eng` | `chrf,none` | 38.20 |" in text
        assert "3820" not in text

    def test_the_footnote_names_both_scales(self) -> None:
        assert "0-100 scale" in bench.report_markdown(self.report())

    def test_a_few_shot_count_is_reported_as_given(self) -> None:
        report = replace(self.report(), few_shot=5)
        assert "Few-shot: `5`" in bench.report_markdown(report)

    def test_an_unset_few_shot_is_not_reported_as_zero(self) -> None:
        """Not passing the flag leaves each task on the harness's own default, which may not be 0."""
        text = bench.report_markdown(self.report())
        assert "Few-shot: `0`" not in text
        assert "not overridden" in text


class TestWriteBench:
    def test_writes_both_halves_into_a_new_directory(self, tmp_path: Path) -> None:
        report = bench.parse_results(payload_text(), model="m")
        path = bench.write_bench(report, tmp_path / "nested" / "bench")
        assert path == tmp_path / "nested" / "bench" / bench.REPORT_NAME
        assert (path.parent / bench.RESULTS_NAME).is_file()

    def test_the_json_round_trips_through_to_dict(self, tmp_path: Path) -> None:
        report = bench.parse_results(payload_text(), model="m", revision="abc")
        bench.write_bench(report, tmp_path)
        written = json.loads((tmp_path / bench.RESULTS_NAME).read_text(encoding="utf-8"))
        assert written == report.to_dict()
        assert written["suite"] == bench.SUITE
        assert written["revision"] == "abc"

    def test_the_markdown_is_the_renderer_output(self, tmp_path: Path) -> None:
        report = bench.parse_results(payload_text(), model="m")
        bench.write_bench(report, tmp_path)
        assert (tmp_path / bench.REPORT_NAME).read_text(encoding="utf-8") == (
            bench.report_markdown(report)
        )


class TestFindResults:
    def test_prefers_the_exact_path_it_asked_for(self, tmp_path: Path) -> None:
        wanted = tmp_path / bench.RAW_NAME
        wanted.parent.mkdir(parents=True)
        wanted.write_text("{}", encoding="utf-8")
        assert bench.find_results(tmp_path, preferred=wanted) == wanted

    def test_searches_when_the_harness_invented_its_own_name(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "2026-10-07" / "results_2026.json"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_text("{}", encoding="utf-8")
        found = bench.find_results(tmp_path, preferred=tmp_path / bench.RAW_NAME)
        assert found == elsewhere

    def test_nothing_found_is_none_rather_than_a_guess(self, tmp_path: Path) -> None:
        assert bench.find_results(tmp_path, preferred=tmp_path / bench.RAW_NAME) is None


class TestRunAfrobench:
    def test_happy_path_writes_the_report_and_returns_it(self, tmp_path: Path) -> None:
        report = bench.run_afrobench(
            tmp_path,
            available=lambda: ("afrixnli_yo", "belebele_hau"),
            run=write_results,
            version=lambda: "0.4.13",
        )
        assert [t.task for t in report.tasks] == ["afrixnli_yo", "belebele_hau"]
        assert report.harness_version == "0.4.13"
        assert (tmp_path / bench.RESULTS_NAME).is_file()
        assert (tmp_path / bench.REPORT_NAME).is_file()

    def test_the_harness_version_comes_from_the_file_when_it_is_there(self, tmp_path: Path) -> None:
        report = bench.run_afrobench(
            tmp_path,
            available=lambda: ("afrixnli_yo",),
            run=write_results,
            version=lambda: "9.9.9",
        )
        assert report.harness_version == "0.4.13"

    def test_the_harness_version_falls_back_to_the_installed_one(self, tmp_path: Path) -> None:
        """A harness that does not record its own version is still identified, or is None."""
        without = json.dumps({"results": {"afrixnli_yo": {"acc,none": 0.1}}})

        def writes_without_version(command: Sequence[str]) -> int:
            target = Path(list(command)[list(command).index("--output_path") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(without, encoding="utf-8")
            return 0

        report = bench.run_afrobench(
            tmp_path,
            available=lambda: ("afrixnli_yo",),
            run=writes_without_version,
            version=lambda: "9.9.9",
        )
        assert report.harness_version == "9.9.9"

    def test_explicit_tasks_override_discovery_but_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="does not have"):
            bench.run_afrobench(
                tmp_path,
                tasks=["afrixnli_yo", "not_a_task"],
                available=lambda: ("afrixnli_yo",),
                run=write_results,
            )

    def test_explicit_tasks_are_passed_through_in_order(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        def recording(command: Sequence[str]) -> int:
            seen["tasks"] = command[list(command).index("--tasks") + 1]
            return write_results(command)

        bench.run_afrobench(
            tmp_path,
            tasks=["mmlu", "afrixnli_yo"],
            available=lambda: ("mmlu", "afrixnli_yo"),
            run=recording,
        )
        assert seen["tasks"] == "mmlu,afrixnli_yo"

    def test_no_matching_task_is_refused_rather_than_run_empty(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceError, match="no task matching") as caught:
            bench.run_afrobench(
                tmp_path, available=lambda: ("mmlu",), run=write_results, version=lambda: "0.4.13"
            )
        assert bench.INSTALL_HINT in (caught.value.hint or "")

    def test_a_failing_harness_writes_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceError, match="status 1") as caught:
            bench.run_afrobench(
                tmp_path,
                available=lambda: ("afrixnli_yo",),
                run=lambda command: write_results(command, status=1),
            )
        assert "nothing was written" in (caught.value.hint or "")
        assert not (tmp_path / bench.RESULTS_NAME).exists()

    def test_a_clean_exit_with_no_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceError, match="wrote no results"):
            bench.run_afrobench(tmp_path, available=lambda: ("afrixnli_yo",), run=lambda command: 0)

    def test_the_harness_file_is_found_wherever_it_landed(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "logs" / "results.json"

        def writes_elsewhere(command: Sequence[str]) -> int:
            elsewhere.parent.mkdir(parents=True, exist_ok=True)
            elsewhere.write_text(payload_text(), encoding="utf-8")
            return 0

        report = bench.run_afrobench(
            tmp_path, available=lambda: ("afrixnli_yo",), run=writes_elsewhere
        )
        assert report.tasks

    def test_the_report_names_the_file_the_harness_really_wrote(self, tmp_path: Path) -> None:
        """``--output_path`` is a hint, not a promise: the report records what it found.

        A version that treats the flag as a directory invents a name underneath, and a report
        that printed the requested path would point at a file that is not there.
        """

        def invents_a_name(command: Sequence[str]) -> int:
            wanted = Path(list(command)[list(command).index("--output_path") + 1])
            wanted.parent.mkdir(parents=True, exist_ok=True)
            (wanted.parent / "results_2026-10-07.json").write_text(payload_text(), encoding="utf-8")
            return 0

        report = bench.run_afrobench(
            tmp_path, available=lambda: ("afrixnli_yo",), run=invents_a_name
        )
        assert report.raw_path == "harness/results_2026-10-07.json"
        assert report.raw_path != bench.RAW_NAME
        assert "`harness/results_2026-10-07.json`" in bench.report_markdown(report)

    def test_the_raw_path_is_in_the_machine_readable_half(self, tmp_path: Path) -> None:
        report = bench.run_afrobench(
            tmp_path, available=lambda: ("afrixnli_yo",), run=write_results
        )
        assert report.to_dict()["raw_path"] == report.raw_path
        written = json.loads((tmp_path / bench.RESULTS_NAME).read_text(encoding="utf-8"))
        assert written["raw_path"] == report.raw_path

    def test_the_few_shot_setting_survives_into_the_report(self, tmp_path: Path) -> None:
        """A figure whose few-shot count is unrecoverable is a figure that cannot be quoted."""
        report = bench.run_afrobench(
            tmp_path, available=lambda: ("afrixnli_yo",), run=write_results, few_shot=5
        )
        assert report.few_shot == 5
        written = json.loads((tmp_path / bench.RESULTS_NAME).read_text(encoding="utf-8"))
        assert written["few_shot"] == 5
        assert "Few-shot: `5`" in (tmp_path / bench.REPORT_NAME).read_text(encoding="utf-8")

    def test_not_passing_few_shot_is_recorded_as_absent(self, tmp_path: Path) -> None:
        report = bench.run_afrobench(
            tmp_path, available=lambda: ("afrixnli_yo",), run=write_results
        )
        assert report.few_shot is None
        assert report.to_dict()["few_shot"] is None

    def test_families_the_harness_lacks_are_recorded_as_gaps(self, tmp_path: Path) -> None:
        report = bench.run_afrobench(
            tmp_path, available=lambda: ("afrixnli_yo",), run=write_results
        )
        assert "belebele" in report.missing_families
        assert "afrixnli" not in report.missing_families

    def test_the_command_it_builds_is_the_one_it_runs(self, tmp_path: Path) -> None:
        seen: dict[str, Any] = {}

        def recording(command: Sequence[str]) -> int:
            seen["command"] = list(command)
            return write_results(command)

        bench.run_afrobench(
            tmp_path,
            model="NCAIR1/N-ATLaS",
            revision="abc",
            available=lambda: ("afrixnli_yo",),
            run=recording,
            batch_size="4",
            few_shot=3,
            device="cpu",
        )
        assert seen["command"] == bench.build_command(
            model="NCAIR1/N-ATLaS",
            tasks=["afrixnli_yo"],
            output_path=tmp_path / bench.RAW_NAME,
            revision="abc",
            batch_size="4",
            few_shot=3,
            device="cpu",
        )


class TestCli:
    def test_list_prints_the_harness_tasks_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bench, "harness_tasks", lambda: ("afrixnli_yo", "mmlu"))
        result = runner.invoke(app, ["bench", "afrobench", "--list"])
        assert result.exit_code == 0
        assert "afrixnli_yo" in result.stdout
        assert "mmlu" not in result.stdout

    def test_list_reports_the_families_nothing_matched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bench, "harness_tasks", lambda: ("afrixnli_yo",))
        result = runner.invoke(app, ["bench", "afrobench", "--list"])
        assert "no harness task matches the 'belebele' family" in result.output

    def test_list_exits_nonzero_when_nothing_matched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bench, "harness_tasks", lambda: ("mmlu",))
        assert runner.invoke(app, ["bench", "afrobench", "--list"]).exit_code == 1

    def test_a_missing_harness_is_a_clean_error_not_an_import_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing() -> tuple[str, ...]:
            raise ImportError("No module named 'lm_eval'")

        monkeypatch.setattr(bench, "harness_tasks", missing)
        result = runner.invoke(app, ["bench", "afrobench", "--list"])
        assert isinstance(result.exception, ResourceError)
        assert bench.INSTALL_HINT in (result.exception.hint or "")

    def test_dry_run_prints_the_command_and_runs_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("--dry-run must not run the harness")

        monkeypatch.setattr(bench, "run_afrobench", explode)
        result = runner.invoke(
            app,
            [
                "bench",
                "afrobench",
                "--dry-run",
                "--tasks",
                "afrixnli_yo",
                "--out",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "-m lm_eval" in result.stdout
        assert "Dry run: nothing was run." in result.stdout

    def test_no_discovered_task_is_a_resource_error_with_a_way_forward(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(bench, "harness_tasks", lambda: ("mmlu",))
        result = runner.invoke(app, ["bench", "afrobench", "--out", str(tmp_path)])
        assert isinstance(result.exception, ResourceError)
        assert "--tasks" in (result.exception.hint or "")

    def test_a_real_run_reports_each_task_and_where_the_files_went(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(bench, "harness_tasks", lambda: ("afrixnli_yo", "belebele_hau"))
        monkeypatch.setattr(bench, "run_afrobench", _fake_run_afrobench)
        result = runner.invoke(app, ["bench", "afrobench", "--out", str(tmp_path)])
        assert result.exit_code == 0
        assert "afrixnli_yo: 33.00 (acc_norm,none)" in result.stdout
        assert bench.RESULTS_NAME in result.stdout
        assert "warning: no harness task matched 'flores'" in result.output


def _fake_run_afrobench(out_dir: str | Path, **kwargs: Any) -> bench.BenchReport:
    """A CLI-level stand-in: real discovery, but no subprocess and no harness."""
    report = bench.parse_results(payload_text(), model=kwargs.get("model", "m"))
    return bench.BenchReport(
        model=report.model,
        revision=report.revision,
        tasks=report.tasks,
        harness_version="0.4.13",
        missing_families=bench.missing_families([t.task for t in report.tasks]),
    )
