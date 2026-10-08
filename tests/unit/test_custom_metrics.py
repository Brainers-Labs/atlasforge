"""Custom metrics, through both doors: a callable in the library, ``module:function`` on the CLI."""

import importlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atlasforge.cli import app
from atlasforge.errors import ConfigError
from atlasforge.eval.custom import declares_higher_is_better, name_of, resolve
from atlasforge.eval.dataset import Dataset, Example, load_dataset
from atlasforge.eval.runner import Record
from atlasforge.eval.runner import run as run_dataset
from atlasforge.eval.score import KNOWN_METRICS, score_run
from atlasforge.types import (
    AudioInput,
    BackendInfo,
    Generation,
    GenParams,
    Lang,
    Message,
    Transcript,
)

runner = CliRunner()

MODULE_SOURCE = '''
"""Metric plug-ins for the test suite."""

SEEN = []


def mentions_dosage(prediction, reference, example):
    SEEN.append((prediction, reference, example.id))
    return float("mg" in prediction)


def word_count(prediction, reference, example):
    return float(len(prediction.split()))


word_count.higher_is_better = False

not_callable = 3
'''


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write ``my_metrics.py`` where ``import`` will find it, as a user would."""
    path = tmp_path / "my_metrics.py"
    path.write_text(MODULE_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "my_metrics"


def make_dataset(tmp_path: Path, rows: Sequence[dict[str, object]]) -> Dataset:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return load_dataset(path, "generation")


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return make_dataset(
        tmp_path,
        [
            {"id": "a", "input": "a", "reference": "5 mg"},
            {"id": "b", "input": "b", "reference": "no dose"},
        ],
    )


def results_echoing(**by_id: str) -> dict[str, Record]:
    return {key: Record(id=key, prediction=value, latency_ms=1.0) for key, value in by_id.items()}


class TestResolve:
    def test_loads_a_named_function(self, plugin: str) -> None:
        fn = resolve(f"{plugin}:mentions_dosage")
        assert callable(fn)
        assert name_of(fn) == "mentions_dosage"

    @pytest.mark.parametrize(
        "spec", ["mentions_dosage", "my_metrics:", ":mentions_dosage", " my_metrics : "]
    )
    def test_a_spec_without_both_halves_is_refused(self, plugin: str, spec: str) -> None:
        with pytest.raises(ConfigError):
            resolve(spec)

    def test_a_missing_module_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="Could not import"):
            resolve("no_such_module_anywhere:fn")

    def test_a_missing_attribute_is_refused(self, plugin: str) -> None:
        with pytest.raises(ConfigError, match="has no"):
            resolve(f"{plugin}:not_there")

    def test_an_attribute_that_is_not_callable_is_refused(self, plugin: str) -> None:
        with pytest.raises(ConfigError, match="not callable"):
            resolve(f"{plugin}:not_callable")

    def test_the_hint_names_the_expected_shape(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolve("nonsense")
        assert caught.value.hint is not None
        assert "prediction, reference, example" in caught.value.hint


class TestNameAndDirection:
    def test_a_callable_object_without_a_name_is_refused(self) -> None:
        class Anonymous:
            def __call__(self, prediction: str, reference: str, example: Example) -> float:
                return 0.0

        with pytest.raises(ConfigError, match="__name__"):
            name_of(Anonymous())

    def test_higher_is_better_defaults_to_true(self) -> None:
        def metric(prediction: str, reference: str, example: Example) -> float:
            return 0.0

        assert declares_higher_is_better(metric) is True

    def test_a_metric_can_declare_a_lower_value_is_better(self, plugin: str) -> None:
        assert declares_higher_is_better(resolve(f"{plugin}:word_count")) is False


class TestScoring:
    def test_a_callable_in_the_metrics_list_is_scored(self, dataset: Dataset) -> None:
        def mentions_dosage(prediction: str, reference: str, example: Example) -> float:
            return float("mg" in prediction)

        report = score_run(
            dataset,
            results_echoing(a="5 mg", b="no dose"),
            metrics=["exact_match", mentions_dosage],
        )
        assert report.metric("mentions_dosage@tone_aware").mean == 0.5
        assert report.metric("mentions_dosage@tone_aware").n == 2

    def test_it_is_reported_under_both_views(self, dataset: Dataset) -> None:
        def always(prediction: str, reference: str, example: Example) -> float:
            return 1.0

        report = score_run(dataset, results_echoing(a="x", b="y"), metrics=[always])
        assert {m.key for m in report.metrics} == {"always@tone_aware", "always@tone_insensitive"}

    def test_a_module_function_string_is_scored(self, plugin: str, dataset: Dataset) -> None:
        """The string form is what `--metric module:function` goes through."""
        report = score_run(
            dataset, results_echoing(a="took 5 mg", b="none"), metrics=[f"{plugin}:mentions_dosage"]
        )
        assert report.metric("mentions_dosage@tone_aware").mean == 0.5

    def test_the_metric_sees_normalised_strings_and_the_example(
        self, plugin: str, dataset: Dataset
    ) -> None:
        # By name rather than `import my_metrics`: the module only exists at run time, so
        # a real import statement is a static error mypy cannot resolve. This is the same
        # path `custom.resolve` takes, so it is the module the metric actually ran from.
        my_metrics = importlib.import_module(plugin)

        my_metrics.SEEN.clear()
        score_run(
            dataset,
            results_echoing(a="  5  MG!  ", b="none"),
            metrics=[f"{plugin}:mentions_dosage"],
        )
        seen = {(example_id, prediction) for prediction, _ref, example_id in my_metrics.SEEN}
        assert ("a", "5 mg") in seen  # lower-cased and de-punctuated, tones kept

    def test_a_lower_is_better_metric_says_so(self, plugin: str, dataset: Dataset) -> None:
        report = score_run(
            dataset, results_echoing(a="one two", b="x"), metrics=[f"{plugin}:word_count"]
        )
        assert report.metric("word_count@tone_aware").higher_is_better is False

    def test_two_metrics_may_not_share_a_name(self, plugin: str, dataset: Dataset) -> None:
        with pytest.raises(ConfigError, match="more than once"):
            score_run(
                dataset,
                results_echoing(a="x", b="y"),
                metrics=[f"{plugin}:word_count", f"{plugin}:word_count"],
            )

    def test_a_custom_metric_may_not_shadow_a_built_in(self, dataset: Dataset) -> None:
        def exact_match(prediction: str, reference: str, example: Example) -> float:
            return 0.0

        with pytest.raises(ConfigError, match="taken by a built-in"):
            score_run(dataset, results_echoing(a="x", b="y"), metrics=[exact_match])

    def test_an_unknown_name_still_lists_the_built_ins(self, dataset: Dataset) -> None:
        with pytest.raises(ConfigError) as caught:
            score_run(dataset, results_echoing(a="x", b="y"), metrics=["nonsense"])
        assert caught.value.hint is not None
        assert "exact_match" in caught.value.hint

    def test_built_in_names_are_unchanged(self) -> None:
        """Adding a name here is a decision, not a detail: these strings are report keys."""
        assert {
            "exact_match",
            "chrf",
            "chrf++",
            "wer",
            "cer",
            "accuracy",
            "macro_f1",
            "accuracy_strict",
            "macro_f1_strict",
        } == KNOWN_METRICS

    def test_a_metric_that_raises_is_not_swallowed(self, dataset: Dataset) -> None:
        """The traceback should point at the caller's own function, not be hidden."""

        def broken(prediction: str, reference: str, example: Example) -> float:
            raise ZeroDivisionError("mine")

        with pytest.raises(ZeroDivisionError, match="mine"):
            score_run(dataset, results_echoing(a="x", b="y"), metrics=[broken])

    def test_custom_metrics_reach_compare(self, plugin: str, tmp_path: Path) -> None:
        from atlasforge.api import compare_runs, evaluate  # noqa: PLC0415

        dataset = make_dataset(
            tmp_path,
            [
                {"id": "a", "input": "a", "reference": "5 mg"},
                {"id": "b", "input": "b", "reference": "none"},
            ],
        )
        base = evaluate(dataset, out_dir=tmp_path / "base", backend=_Echo())
        tuned = evaluate(dataset, out_dir=tmp_path / "tuned", backend=_Echo("5 mg"))
        report = compare_runs(
            dataset, base.out_dir, tuned.out_dir, metrics=[f"{plugin}:mentions_dosage"]
        )
        assert {m.name for m in report.metrics} == {"mentions_dosage"}
        assert {m.key for m in report.metrics} == {
            "mentions_dosage@tone_aware",
            "mentions_dosage@tone_insensitive",
        }


class _Echo:
    """The smallest backend that can drive a run."""

    def __init__(self, text: str = "none") -> None:
        self.text = text

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        return Generation(text=self.text, latency_ms=1.0)

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        raise NotImplementedError

    def info(self) -> BackendInfo:
        return BackendInfo(backend="echo", model="echo")

    def close(self) -> None:
        return None


class TestCommandLine:
    def test_metric_accepts_a_module_function(self, plugin: str, tmp_path: Path) -> None:
        dataset = make_dataset(
            tmp_path,
            [
                {"id": "a", "input": "a", "reference": "5 mg"},
                {"id": "b", "input": "b", "reference": "no"},
            ],
        )
        run_dataset(_Echo("took 5 mg"), dataset, tmp_path / "run")

        result = runner.invoke(
            app,
            [
                "report",
                str(tmp_path / "run"),
                "--dataset",
                str(dataset.path),
                "--metric",
                f"{plugin}:mentions_dosage",
            ],
        )
        assert result.exit_code == 0, result.output
        written = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
        keys = {m["key"] for m in written["metrics"]}
        assert keys == {"mentions_dosage@tone_aware", "mentions_dosage@tone_insensitive"}

    def test_a_bad_spec_explains_itself(self, plugin: str, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path, [{"id": "a", "input": "a", "reference": "r"}])
        run_dataset(_Echo("x"), dataset, tmp_path / "run")

        result = runner.invoke(
            app,
            [
                "report",
                str(tmp_path / "run"),
                "--dataset",
                str(dataset.path),
                "--metric",
                "no_such_module_anywhere:fn",
            ],
        )
        assert result.exit_code != 0
        # Invoked through `app` rather than `main()`, so the ConfigError is the exception
        # main() would turn into `error: ...` output. See the other CLI error tests.
        failure = result.exception
        assert isinstance(failure, ConfigError)
        assert "no_such_module_anywhere" in str(failure)
        assert failure.hint is not None
        assert "module:function" in failure.hint

    def test_metric_names_still_work_on_the_command_line(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path, [{"id": "a", "input": "a", "reference": "x"}])
        run_dataset(_Echo("x"), dataset, tmp_path / "run")

        result = runner.invoke(
            app,
            ["report", str(tmp_path / "run"), "--dataset", str(dataset.path), "-m", "exact_match"],
        )
        assert result.exit_code == 0, result.output
        written = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
        assert {m["key"] for m in written["metrics"]} == {
            "exact_match@tone_aware",
            "exact_match@tone_insensitive",
        }
