"""The high-level library API: ``atlasforge.evaluate`` and ``atlasforge.compare``.

A fake backend stands in for a model, so everything here runs offline with no weights.
"""

import json
import subprocess
import sys
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import atlasforge
from atlasforge import api
from atlasforge.api import Evaluation
from atlasforge.errors import ConfigError
from atlasforge.eval.dataset import Dataset, load_dataset
from atlasforge.eval.score import ScoreReport
from atlasforge.types import (
    AudioInput,
    BackendInfo,
    Generation,
    GenParams,
    Lang,
    Message,
    Transcript,
)


class FakeBackend:
    """Echoes the prompt, so every prediction is known in advance."""

    def __init__(self, *, model: str = "fake/model", prefix: str = "echo:") -> None:
        self.model = model
        self.prefix = prefix
        self.calls: list[str] = []
        self.closed = False

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        return Generation(text=f"{self.prefix}{prompt}", latency_ms=1.0)

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        self.calls.append(f"{Path(str(audio)).name}|{lang}")
        return Transcript(text=self.prefix, lang=lang, latency_ms=2.0)

    def info(self) -> BackendInfo:
        return BackendInfo(backend="fake", model=self.model)

    def close(self) -> None:
        self.closed = True


def write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def make_dataset(tmp_path: Path, ids: Sequence[str] = ("a", "b", "c")) -> Dataset:
    path = write_jsonl(
        tmp_path / "data.jsonl",
        [{"id": i, "input": i, "reference": f"echo:{i}"} for i in ids],
    )
    return load_dataset(path, "generation")


def silent_wav(path: Path, seconds: float = 0.2) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return path


class TestEvaluate:
    def test_returns_run_scores_and_reports(self, tmp_path: Path) -> None:
        result = api.evaluate(
            make_dataset(tmp_path), out_dir=tmp_path / "run", backend=FakeBackend()
        )
        assert isinstance(result, Evaluation)
        assert (result.run.total, result.run.ok) == (3, 3)
        assert result.out_dir == tmp_path / "run"
        assert (result.out_dir / api.REPORT_JSON).is_file()
        assert (result.out_dir / api.REPORT_MD).is_file()
        assert (result.out_dir / api.REPORT_HTML).is_file()

    def test_predictions_are_scored_not_just_written(self, tmp_path: Path) -> None:
        result = api.evaluate(
            make_dataset(tmp_path), out_dir=tmp_path / "run", backend=FakeBackend()
        )
        # The fake echoes the reference exactly, so a perfect score is the honest answer.
        assert result.metric("exact_match@tone_aware").mean == 1.0

    def test_accepts_a_dataset_path_and_a_task(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        result = api.evaluate(
            dataset.path, out_dir=tmp_path / "run", task="generation", backend=FakeBackend()
        )
        assert result.scores.n_total == 3

    def test_write_report_false_leaves_no_report_files(self, tmp_path: Path) -> None:
        result = api.evaluate(
            make_dataset(tmp_path),
            out_dir=tmp_path / "run",
            backend=FakeBackend(),
            write_report=False,
        )
        assert not (result.out_dir / api.REPORT_JSON).exists()
        assert result.scores.n_ok == 3  # still scored, just not written

    def test_metrics_can_be_narrowed(self, tmp_path: Path) -> None:
        result = api.evaluate(
            make_dataset(tmp_path),
            out_dir=tmp_path / "run",
            backend=FakeBackend(),
            metrics=["exact_match"],
        )
        assert {m.name for m in result.scores.metrics} == {"exact_match"}

    def test_a_passed_backend_is_not_closed(self, tmp_path: Path) -> None:
        """The caller owns a backend it built; closing it here would be a surprise."""
        backend = FakeBackend()
        api.evaluate(make_dataset(tmp_path), out_dir=tmp_path / "run", backend=backend)
        assert backend.closed is False

    def test_a_named_backend_is_created_and_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = FakeBackend()

        def fake_build(name: str, **options: object) -> FakeBackend:
            assert name == "openai"
            assert options["base_url"] == "http://localhost:8000/v1"
            return created

        monkeypatch.setattr("atlasforge.backends.factory.build_backend", fake_build)
        api.evaluate(
            make_dataset(tmp_path),
            out_dir=tmp_path / "run",
            backend="openai",
            base_url="http://localhost:8000/v1",
        )
        assert created.closed is True

    def test_unknown_backend_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            api.evaluate(make_dataset(tmp_path), out_dir=tmp_path / "run", backend="nope")

    def test_speech_datasets_get_the_long_audio_splitter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        used: list[FakeBackend] = []
        options: dict[str, Any] = {}

        class Recording:
            """Stands in for the splitter, forwarding to the backend it wraps."""

            def __init__(self, inner: FakeBackend, **kwargs: Any) -> None:
                used.append(inner)
                options.update(kwargs)

            def generate(
                self, messages: Sequence[Message], params: GenParams | None = None
            ) -> Generation:
                return used[0].generate(messages, params)

            def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
                return used[0].transcribe(audio, lang)

            def info(self) -> BackendInfo:
                return used[0].info()

            def close(self) -> None:
                used[0].close()

        monkeypatch.setattr("atlasforge.asr.chunking.LongAudioBackend", Recording)
        dataset = load_dataset(
            write_jsonl(
                tmp_path / "asr.jsonl",
                [
                    {
                        "id": "a",
                        "audio": silent_wav(tmp_path / "a.wav").name,
                        "lang": "ha",
                        "reference": "echo:",
                    }
                ],
            ),
            "asr",
        )
        result = api.evaluate(dataset, out_dir=tmp_path / "run", backend=FakeBackend())
        assert used, "the ASR path must chunk long audio"
        assert result.scores.n_total == 1
        assert options["silence_aware"] is False, "the fixed grid stays the default"

        # The keyword is not decoration: it is forwarded to the constructor that reads it.
        api.evaluate(
            dataset,
            out_dir=tmp_path / "run",
            backend=FakeBackend(),
            silence_aware=True,
        )
        assert options["silence_aware"] is True

    def test_resuming_skips_finished_examples(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        api.evaluate(dataset, out_dir=tmp_path / "run", backend=FakeBackend())
        second = api.evaluate(dataset, out_dir=tmp_path / "run", backend=FakeBackend())
        assert second.run.skipped == 3
        assert second.run.ok == 3


class TestCompare:
    def test_two_runs_are_compared(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        base = api.evaluate(dataset, out_dir=tmp_path / "base", backend=FakeBackend())
        tuned = api.evaluate(
            dataset, out_dir=tmp_path / "tuned", backend=FakeBackend(prefix="echo:")
        )
        report = api.compare_runs(dataset, base.out_dir, tuned.out_dir)
        assert report.n_total == 3
        assert {m.name for m in report.metrics} == {"exact_match", "chrf"}

    def test_out_dir_writes_both_reports(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        base = api.evaluate(dataset, out_dir=tmp_path / "base", backend=FakeBackend())
        tuned = api.evaluate(dataset, out_dir=tmp_path / "tuned", backend=FakeBackend())
        api.compare_runs(dataset, base.out_dir, tuned.out_dir, out_dir=tmp_path / "cmp")
        written = json.loads((tmp_path / "cmp" / api.COMPARISON_JSON).read_text(encoding="utf-8"))
        assert written["n_total"] == 3
        assert (tmp_path / "cmp" / api.COMPARISON_MD).is_file()
        assert (tmp_path / "cmp" / api.COMPARISON_HTML).is_file()

    def test_no_out_dir_writes_nothing(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path)
        api.evaluate(dataset, out_dir=tmp_path / "base", backend=FakeBackend())
        api.evaluate(dataset, out_dir=tmp_path / "tuned", backend=FakeBackend())
        api.compare_runs(dataset, tmp_path / "base", tmp_path / "tuned")
        assert not (tmp_path / "comparison").exists()

    def test_a_run_from_another_dataset_is_refused(self, tmp_path: Path) -> None:
        base = api.evaluate(
            make_dataset(tmp_path / "one"), out_dir=tmp_path / "base", backend=FakeBackend()
        )
        other = make_dataset(tmp_path / "two", ids=("x", "y"))
        with pytest.raises(ConfigError):
            api.compare_runs(other, base.out_dir, base.out_dir)


class TestRescoring:
    def test_score_finished_run_reads_results_back(self, tmp_path: Path) -> None:
        result = api.evaluate(
            make_dataset(tmp_path), out_dir=tmp_path / "run", backend=FakeBackend()
        )
        rescored = api.score_finished_run(tmp_path / "run", result.dataset)
        assert isinstance(rescored, ScoreReport)
        assert rescored.metric("exact_match@tone_aware").mean == 1.0

    def test_write_reports_redoes_the_report_in_place(self, tmp_path: Path) -> None:
        result = api.evaluate(
            make_dataset(tmp_path), out_dir=tmp_path / "run", backend=FakeBackend()
        )
        (tmp_path / "run" / api.REPORT_MD).unlink()
        report = api.write_reports(tmp_path / "run", result.dataset)
        assert report.n_ok == 3
        assert (tmp_path / "run" / api.REPORT_MD).is_file()
        assert (tmp_path / "run" / api.REPORT_HTML).is_file()


class TestPackageSurface:
    def test_the_top_level_names_are_the_api_functions(self) -> None:
        assert atlasforge.evaluate is api.evaluate
        assert atlasforge.compare_runs is api.compare_runs

    def test_all_lists_exactly_what_is_exported(self) -> None:
        assert set(atlasforge.__all__) == {
            "Evaluation",
            "__version__",
            "compare_runs",
            "evaluate",
            "score_finished_run",
            "write_comparison",
            "write_reports",
        }
        for name in atlasforge.__all__:
            assert hasattr(atlasforge, name), name

    def test_the_name_compare_belongs_to_the_subpackage(self) -> None:
        """The entry point is ``compare_runs`` because ``atlasforge.compare`` is taken.

        If this ever stops being the subpackage, ``atlasforge.compare(...)`` and
        ``atlasforge.compare.compare_runs`` would start overwriting each other depending
        on import order — the reason the wrapper is named the way it is.
        """
        import atlasforge.compare as subpackage  # noqa: PLC0415

        assert subpackage.__name__ == "atlasforge.compare"
        assert subpackage.compare_runs is not api.compare_runs  # the wrapper adds to it

    def test_importing_the_package_pulls_in_no_ml_framework(self) -> None:
        code = (
            "import sys, atlasforge; "
            "bad = [m for m in ('torch', 'transformers', 'librosa') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
        assert proc.returncode == 0, proc.stderr.decode()
