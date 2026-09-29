import json
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from atlasforge.errors import BackendTimeout, ConfigError, DatasetError
from atlasforge.eval.dataset import Dataset, load_dataset
from atlasforge.eval.runner import (
    MANIFEST_NAME,
    RESULTS_NAME,
    Record,
    RunConfig,
    read_results,
    run,
)
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
    """Deterministic backend. ``fail`` maps prompt text to an exception to raise."""

    def __init__(
        self, *, model: str = "fake/model", fail: dict[str, BaseException] | None = None
    ) -> None:
        self.model = model
        self.fail = fail or {}
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        prompt = messages[-1]["content"]
        with self._lock:
            self.calls.append(prompt)
        if prompt in self.fail:
            raise self.fail[prompt]
        return Generation(text=f"echo:{prompt}", latency_ms=5.0, request_id=f"req-{prompt}")

    def transcribe(self, audio: AudioInput, lang: Lang) -> Transcript:
        with self._lock:
            self.calls.append(f"{Path(str(audio)).name}|{lang}")
        return Transcript(text=f"heard:{lang}", lang=lang, latency_ms=7.0)

    def info(self) -> BackendInfo:
        return BackendInfo(backend="fake", model=self.model, revision="abc123")

    def close(self) -> None:
        return None


def make_dataset(tmp_path: Path, prompts: Sequence[str], task: str = "generation") -> Dataset:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": p, "input": p, "reference": "r"}) for p in prompts) + "\n",
        encoding="utf-8",
    )
    return load_dataset(path, task)  # type: ignore[arg-type]


class TestBasicRun:
    def test_writes_manifest_and_results(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path, ["a", "b", "c"])
        summary = run(FakeBackend(), ds, tmp_path / "out")
        assert (summary.total, summary.skipped, summary.ok, summary.failed) == (3, 0, 3, 0)
        manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["dataset_sha256"] == ds.sha256
        assert manifest["model"] == "fake/model"
        assert manifest["revision"] == "abc123"
        assert manifest["gen_params"]["temperature"] == 0.1
        results = read_results(tmp_path / "out" / RESULTS_NAME)
        assert results["b"].prediction == "echo:b"
        assert results["b"].latency_ms == 5.0
        assert results["b"].request_id == "req-b"

    def test_unicode_predictions_stored_readably(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path, ["ṣé"])
        run(FakeBackend(), ds, tmp_path / "out")
        raw = (tmp_path / "out" / RESULTS_NAME).read_text(encoding="utf-8")
        assert "ṣé" in raw  # not \\u-escaped

    def test_on_result_callback(self, tmp_path: Path) -> None:
        seen: list[str] = []
        run(
            FakeBackend(),
            make_dataset(tmp_path, ["a", "b"]),
            tmp_path / "o",
            on_result=lambda r: seen.append(r.id),
        )
        assert seen == ["a", "b"]


class TestErrorIsolation:
    def test_atlasforge_error_recorded_and_run_continues(self, tmp_path: Path) -> None:
        backend = FakeBackend(fail={"b": BackendTimeout("took too long")})
        summary = run(backend, make_dataset(tmp_path, ["a", "b", "c"]), tmp_path / "o")
        assert (summary.ok, summary.failed) == (2, 1)
        record = read_results(tmp_path / "o" / RESULTS_NAME)["b"]
        assert record.error == "BackendTimeout: took too long"
        assert record.prediction is None

    def test_unexpected_error_never_leaks_message(self, tmp_path: Path) -> None:
        backend = FakeBackend(fail={"a": RuntimeError("secret prompt text here")})
        run(backend, make_dataset(tmp_path, ["a"]), tmp_path / "o")
        record = read_results(tmp_path / "o" / RESULTS_NAME)["a"]
        assert record.error == "UnexpectedError: RuntimeError"
        assert "secret" not in (tmp_path / "o" / RESULTS_NAME).read_text(encoding="utf-8")

    def test_keyboard_interrupt_propagates(self, tmp_path: Path) -> None:
        backend = FakeBackend(fail={"a": KeyboardInterrupt()})
        with pytest.raises(KeyboardInterrupt):
            run(backend, make_dataset(tmp_path, ["a"]), tmp_path / "o")


class TestResume:
    def test_second_run_does_no_work(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path, ["a", "b"])
        run(FakeBackend(), ds, tmp_path / "o")
        again = FakeBackend()
        summary = run(again, ds, tmp_path / "o")
        assert again.calls == []
        assert (summary.skipped, summary.ok) == (2, 2)
        assert len((tmp_path / "o" / RESULTS_NAME).read_text(encoding="utf-8").splitlines()) == 2

    def test_errors_are_retried_by_default(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path, ["a", "b"])
        run(FakeBackend(fail={"b": BackendTimeout("x")}), ds, tmp_path / "o")
        healthy = FakeBackend()
        summary = run(healthy, ds, tmp_path / "o")
        assert healthy.calls == ["b"]
        assert (summary.skipped, summary.ok, summary.failed) == (1, 2, 0)
        assert read_results(tmp_path / "o" / RESULTS_NAME)["b"].ok

    def test_errors_not_retried_when_disabled(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path, ["a"])
        run(FakeBackend(fail={"a": BackendTimeout("x")}), ds, tmp_path / "o")
        again = FakeBackend()
        summary = run(again, ds, tmp_path / "o", config=RunConfig(retry_errors=False))
        assert again.calls == []
        assert summary.failed == 1

    def test_truncated_last_line_is_dropped_and_rerun(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path, ["a", "b"])
        run(FakeBackend(), ds, tmp_path / "o")
        results = tmp_path / "o" / RESULTS_NAME
        results.write_bytes(results.read_bytes()[:-10])  # simulate crash mid-write
        again = FakeBackend()
        summary = run(again, ds, tmp_path / "o")
        assert again.calls == ["b"]
        assert summary.ok == 2
        assert all(json.loads(line) for line in results.read_text(encoding="utf-8").splitlines())

    @pytest.mark.parametrize(
        "change",
        ["dataset", "model", "params", "lang"],
    )
    def test_refuses_to_mix_different_runs(self, tmp_path: Path, change: str) -> None:
        ds = make_dataset(tmp_path, ["a"])
        run(FakeBackend(), ds, tmp_path / "o")
        backend, config = FakeBackend(), RunConfig()
        if change == "dataset":
            ds = make_dataset(tmp_path, ["a", "b"])
        elif change == "model":
            backend = FakeBackend(model="other/model")
        elif change == "params":
            config = RunConfig(gen_params=GenParams(temperature=0.9))
        else:
            config = RunConfig(lang="ha")
        with pytest.raises(ConfigError, match="different run") as info:
            run(backend, ds, tmp_path / "o", config=config)
        assert info.value.hint is not None

    def test_corrupt_middle_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / RESULTS_NAME
        path.write_text('{"id": "a", "prediction": "x"}\nGARBAGE\n{"id": "b", "prediction": "y"}\n')
        with pytest.raises(DatasetError, match="corrupt"):
            read_results(path)

    def test_missing_results_file_is_empty(self, tmp_path: Path) -> None:
        assert read_results(tmp_path / "nope.jsonl") == {}

    def test_latest_record_wins(self, tmp_path: Path) -> None:
        path = tmp_path / RESULTS_NAME
        path.write_text('{"id": "a", "error": "boom"}\n{"id": "a", "prediction": "ok"}\n')
        assert read_results(path)["a"] == Record(id="a", prediction="ok")


class TestConcurrency:
    def test_threaded_run_completes_everything_exactly_once(self, tmp_path: Path) -> None:
        prompts = [f"p{i}" for i in range(50)]
        backend = FakeBackend()
        summary = run(
            backend,
            make_dataset(tmp_path, prompts),
            tmp_path / "o",
            config=RunConfig(concurrency=8),
        )
        assert summary.ok == 50
        assert sorted(backend.calls) == sorted(prompts)
        assert set(read_results(tmp_path / "o" / RESULTS_NAME)) == set(prompts)

    def test_results_file_lines_are_never_interleaved(self, tmp_path: Path) -> None:
        run(
            FakeBackend(),
            make_dataset(tmp_path, [f"p{i}" for i in range(40)]),
            tmp_path / "o",
            config=RunConfig(concurrency=8),
        )
        lines = (tmp_path / "o" / RESULTS_NAME).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 40
        assert all("id" in json.loads(x) for x in lines)

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(ConfigError):
            RunConfig(concurrency=0)


class TestAsr:
    def make_asr(self, tmp_path: Path, rows: list[dict[str, str]]) -> Dataset:
        (tmp_path / "a.wav").write_bytes(b"RIFF")
        path = tmp_path / "asr.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return load_dataset(path, "asr")

    def test_uses_example_lang_and_transcribes(self, tmp_path: Path) -> None:
        ds = self.make_asr(
            tmp_path, [{"id": "1", "audio": "a.wav", "reference": "x", "lang": "yo"}]
        )
        backend = FakeBackend()
        run(backend, ds, tmp_path / "o")
        assert backend.calls == ["a.wav|yo"]
        assert read_results(tmp_path / "o" / RESULTS_NAME)["1"].prediction == "heard:yo"

    def test_default_lang_fills_gaps(self, tmp_path: Path) -> None:
        ds = self.make_asr(tmp_path, [{"id": "1", "audio": "a.wav", "reference": "x"}])
        backend = FakeBackend()
        run(backend, ds, tmp_path / "o", config=RunConfig(lang="ha"))
        assert backend.calls == ["a.wav|ha"]

    def test_missing_language_fails_up_front_before_any_work(self, tmp_path: Path) -> None:
        ds = self.make_asr(tmp_path, [{"id": "1", "audio": "a.wav", "reference": "x"}])
        backend = FakeBackend()
        with pytest.raises(ConfigError, match="no language") as info:
            run(backend, ds, tmp_path / "o")
        assert info.value.hint is not None
        assert backend.calls == []
