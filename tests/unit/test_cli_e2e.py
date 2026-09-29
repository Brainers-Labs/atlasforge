"""End-to-end: the real CLI, the real OpenAI backend and a real (fake-model) HTTP server.

Nothing here is mocked between the command line and the socket. Only the model behind
the server is fake: it answers from a lookup table, so results are known in advance.
"""

import json
import math
import shutil
import struct
import sys
import threading
import wave
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from atlasforge.cli import app, main
from atlasforge.errors import RunAborted

runner = CliRunner()
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

N = 40


class FakeModelServer(ThreadingHTTPServer):
    """``answers[model][prompt]`` decides each reply; unknown prompts get ``"x"``."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answers: dict[str, dict[str, str]] = {}
        self.requests: list[dict[str, Any]] = []
        self.fail_prompts: set[str] = set()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _Handler(BaseHTTPRequestHandler):
    server: FakeModelServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.path.endswith("/chat/completions"):
            data = json.loads(raw)
            self.server.requests.append(data)
            prompt = data["messages"][-1]["content"]
            if prompt in self.server.fail_prompts:
                self._send(500, {"error": "boom"})
                return
            answer = self.server.answers.get(data["model"], {}).get(prompt, "x")
            self._send(
                200,
                {
                    "id": "chatcmpl-test",
                    "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            )
        elif self.path.endswith("/audio/transcriptions"):
            self._send(200, {"text": "ina kwana"})
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture
def server() -> Iterator[FakeModelServer]:
    fake = FakeModelServer()
    thread = threading.Thread(target=fake.serve_forever, daemon=True)
    thread.start()
    yield fake
    fake.shutdown()
    fake.server_close()


def write_dataset(tmp_path: Path, name: str = "data.jsonl") -> Path:
    rows = [
        {
            "id": f"e{i}",
            "input": f"q{i}",
            "reference": f"ans{i}",
            "meta": {"domain": "agri" if i < N // 2 else "legal"},
        }
        for i in range(N)
    ]
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def setup_models(server: FakeModelServer, *, correct_for_candidate: int = N // 2) -> None:
    server.answers["base-m"] = {}  # always wrong
    server.answers["cand-m"] = {f"q{i}": f"ans{i}" for i in range(correct_for_candidate)}


def eval_args(server: FakeModelServer, data: Path, out: Path, model: str) -> list[str]:
    return [
        "eval", str(data), "--out", str(out), "--base-url", server.base_url,
        "--model", model, "-m", "exact_match", "-m", "chrf", "--retries", "0",
    ]  # fmt: skip


class TestRun:
    def test_prints_the_answer(self, server: FakeModelServer) -> None:
        server.answers["m"] = {"Ina kwana?": "Lafiya lau"}
        result = runner.invoke(
            app, ["run", "Ina kwana?", "--base-url", server.base_url, "--model", "m"]
        )
        assert result.exit_code == 0
        assert result.output.strip() == "Lafiya lau"

    def test_json_output_and_system_prompt(self, server: FakeModelServer) -> None:
        server.answers["m"] = {"hi": "ok"}
        result = runner.invoke(
            app,
            [
                "run",
                "hi",
                "--system",
                "Be brief.",
                "--json",
                "--base-url",
                server.base_url,
                "--model",
                "m",
                "--seed",
                "3",
            ],
        )
        data = json.loads(result.output)
        assert data["text"] == "ok"
        assert data["model"] == "m"
        assert data["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
        assert data["request_id"] == "chatcmpl-test"
        sent = server.requests[0]
        assert [m["role"] for m in sent["messages"]] == ["system", "user"]
        assert sent["seed"] == 3
        assert sent["temperature"] == 0.1

    def test_reads_the_prompt_from_stdin(self, server: FakeModelServer) -> None:
        server.answers["m"] = {"from stdin": "got it"}
        result = runner.invoke(
            app, ["run", "-", "--base-url", server.base_url, "--model", "m"], input="from stdin"
        )
        assert result.output.strip() == "got it"

    def test_unicode_survives_the_round_trip(self, server: FakeModelServer) -> None:
        prompt, answer = "Báwò ni?", "Ṣé o wà dáadáa"
        server.answers["m"] = {prompt: answer}
        result = runner.invoke(
            app, ["run", prompt, "--base-url", server.base_url, "--model", "m", "--json"]
        )
        assert json.loads(result.output)["text"] == answer


class TestEval:
    def test_end_to_end_writes_reports(self, server: FakeModelServer, tmp_path: Path) -> None:
        setup_models(server)
        data, out = write_dataset(tmp_path), tmp_path / "base"
        result = runner.invoke(app, eval_args(server, data, out, "cand-m"))
        assert result.exit_code == 0, result.output
        assert (out / "run.json").is_file()
        assert (out / "results.jsonl").is_file()
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert report["n_total"] == N
        assert report["n_failed"] == 0
        markdown = (out / "report.md").read_text(encoding="utf-8")
        assert "cand-m" in markdown
        assert "tone-aware" in markdown
        assert "tone-insensitive" in markdown
        # candidate is right on the first half, wrong on the second -> 50%
        aware = next(m for m in report["metrics"] if m["key"] == "exact_match@tone_aware")
        assert aware["mean"] == pytest.approx(0.5)
        assert "exact_match" in result.output
        assert "Wrote" in result.output

    def test_resume_does_no_new_work(self, server: FakeModelServer, tmp_path: Path) -> None:
        setup_models(server)
        data, out = write_dataset(tmp_path), tmp_path / "run"
        runner.invoke(app, eval_args(server, data, out, "cand-m"))
        before = len(server.requests)
        result = runner.invoke(app, eval_args(server, data, out, "cand-m"))
        assert result.exit_code == 0
        assert len(server.requests) == before
        assert f"Resumed: {N} example(s)" in result.output

    def test_refuses_to_resume_into_a_different_model(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        setup_models(server)
        data, out = write_dataset(tmp_path), tmp_path / "run"
        runner.invoke(app, eval_args(server, data, out, "cand-m"))
        result = runner.invoke(app, eval_args(server, data, out, "base-m"))
        assert result.exit_code != 0
        assert "different run" in str(result.exception)

    def test_partial_failures_are_reported_and_counted_as_wrong(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        setup_models(server, correct_for_candidate=N)
        server.fail_prompts = {"q0", "q1", "q2", "q3"}
        data, out = write_dataset(tmp_path), tmp_path / "run"
        result = runner.invoke(app, eval_args(server, data, out, "cand-m"))
        assert result.exit_code == 0
        assert "4 of 40 example(s) failed" in result.output
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert report["n_failed"] == 4
        aware = next(m for m in report["metrics"] if m["key"] == "exact_match@tone_aware")
        assert aware["mean"] == pytest.approx(36 / 40)

    def test_total_failure_exits_nonzero_when_the_breaker_is_off(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        setup_models(server)
        server.fail_prompts = {f"q{i}" for i in range(N)}
        args = [
            *eval_args(server, write_dataset(tmp_path), tmp_path / "run", "cand-m"),
            "--max-consecutive-failures",
            "0",
        ]
        assert runner.invoke(app, args).exit_code == 1

    def test_a_dead_server_stops_the_run_early_and_keeps_progress(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        setup_models(server)
        server.fail_prompts = {f"q{i}" for i in range(N)}
        out = tmp_path / "run"
        args = [
            *eval_args(server, write_dataset(tmp_path), out, "cand-m"),
            "--max-consecutive-failures",
            "5",
        ]
        result = runner.invoke(app, args)
        assert isinstance(result.exception, RunAborted)
        assert "5 failures in a row" in str(result.exception)
        assert "resume" in (result.exception.hint or "")
        assert len(server.requests) == 5  # it did not keep hammering the server
        assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 5

    def test_the_aborted_run_resumes_once_the_server_recovers(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        setup_models(server, correct_for_candidate=N)
        server.fail_prompts = {f"q{i}" for i in range(N)}
        out = tmp_path / "run"
        args = [
            *eval_args(server, write_dataset(tmp_path), out, "cand-m"),
            "--max-consecutive-failures",
            "5",
        ]
        runner.invoke(app, args)
        server.fail_prompts = set()  # the server is back
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert (report["n_ok"], report["n_failed"]) == (N, 0)

    def test_unknown_task(self, server: FakeModelServer, tmp_path: Path) -> None:
        args = [
            *eval_args(server, write_dataset(tmp_path), tmp_path / "run", "cand-m"),
            "--task",
            "nope",
        ]
        result = runner.invoke(app, args)
        assert result.exit_code != 0
        assert "Unknown task" in str(result.exception)

    def test_concurrent_run_gives_the_same_scores(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        setup_models(server)
        data = write_dataset(tmp_path)
        runner.invoke(app, eval_args(server, data, tmp_path / "one", "cand-m"))
        runner.invoke(
            app, [*eval_args(server, data, tmp_path / "many", "cand-m"), "--concurrency", "8"]
        )
        one = json.loads((tmp_path / "one" / "report.json").read_text(encoding="utf-8"))
        many = json.loads((tmp_path / "many" / "report.json").read_text(encoding="utf-8"))
        assert one["per_example"] == many["per_example"]


class TestReportAndCompare:
    def make_runs(self, server: FakeModelServer, tmp_path: Path) -> tuple[Path, Path, Path]:
        setup_models(server)
        data = write_dataset(tmp_path)
        runner.invoke(app, eval_args(server, data, tmp_path / "base", "base-m"))
        runner.invoke(app, eval_args(server, data, tmp_path / "cand", "cand-m"))
        return data, tmp_path / "base", tmp_path / "cand"

    def test_report_rescoring_needs_no_model(self, server: FakeModelServer, tmp_path: Path) -> None:
        data, base, _ = self.make_runs(server, tmp_path)
        (base / "report.md").unlink()
        before = len(server.requests)
        result = runner.invoke(
            app, ["report", str(base), "--dataset", str(data), "-m", "exact_match"]
        )
        assert result.exit_code == 0, result.output
        assert (base / "report.md").is_file()
        assert len(server.requests) == before

    def test_compare_finds_the_improvement_and_writes_reports(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        data, base, cand = self.make_runs(server, tmp_path)
        out = tmp_path / "cmp"
        result = runner.invoke(
            app,
            ["compare", str(data), "--base", str(base), "--candidate", str(cand), "--out", str(out),
             "-m", "exact_match", "-m", "chrf", "--slice", "domain", "--min-slice-n", "10", "--n-boot", "300"],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert "improved" in result.output
        markdown = (out / "comparison.md").read_text(encoding="utf-8")
        assert "base-m" in markdown
        assert "cand-m" in markdown
        assert "domain" in markdown
        data_json = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
        exact = next(m for m in data_json["metrics"] if m["key"] == "exact_match@tone_aware")
        assert exact["verdict"] == "improved"
        assert exact["delta"] == pytest.approx(0.5)
        by_value = {s["value"]: s["status"] for s in data_json["slices"] if s["field"] == "domain"}
        assert by_value["agri"] == "improved"

    def test_default_slice_size_marks_small_slices_insufficient(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        data, base, cand = self.make_runs(server, tmp_path)
        out = tmp_path / "cmp"
        runner.invoke(
            app,
            ["compare", str(data), "--base", str(base), "--candidate", str(cand), "--out", str(out),
             "-m", "exact_match", "--slice", "domain", "--n-boot", "100"],
        )  # fmt: skip
        markdown = (out / "comparison.md").read_text(encoding="utf-8")
        assert "insufficient data" in markdown

    def test_compare_refuses_runs_from_another_dataset(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        _, base, cand = self.make_runs(server, tmp_path)
        other = tmp_path / "other.jsonl"
        other.write_text(
            json.dumps({"id": "z", "input": "z", "reference": "z"}) + "\n", encoding="utf-8"
        )
        result = runner.invoke(
            app, ["compare", str(other), "--base", str(base), "--candidate", str(cand)]
        )
        assert result.exit_code != 0
        assert "not produced from this dataset" in str(result.exception)


class TestDatasetValidate:
    def test_clean_file_exits_zero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["dataset", "validate", str(write_dataset(tmp_path))])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_bad_file_exits_one_and_names_the_line(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"input": "ok", "reference": "x"}\n{broken\n', encoding="utf-8")
        result = runner.invoke(app, ["dataset", "validate", str(bad)])
        assert result.exit_code == 1
        assert "line 2" in result.output
        assert "FAILED" in result.output

    def test_json_output_is_machine_readable(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["dataset", "validate", str(write_dataset(tmp_path)), "--json"])
        data = json.loads(result.output)
        assert data["ok"] if "ok" in data else data["n_examples"] == N

    def test_leakage_check(self, tmp_path: Path) -> None:
        train, test = write_dataset(tmp_path, "train.jsonl"), write_dataset(tmp_path, "test.jsonl")
        result = runner.invoke(app, ["dataset", "validate", str(test), "--against", str(train)])
        assert result.exit_code == 1
        assert "appear in train.jsonl" in result.output


def sine_wav(path: Path, seconds: float = 1.0) -> Path:
    rate = 16_000
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(int(rate * seconds))
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return path


@needs_ffmpeg
class TestTranscribe:
    def test_single_file_prints_text(self, server: FakeModelServer, tmp_path: Path) -> None:
        audio = sine_wav(tmp_path / "a.wav")
        result = runner.invoke(
            app, ["transcribe", str(audio), "--lang", "ha", "--base-url", server.base_url]
        )
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "ina kwana"

    def test_long_audio_is_split_into_chunks(self, server: FakeModelServer, tmp_path: Path) -> None:
        audio = sine_wav(tmp_path / "long.wav", 65.0)
        result = runner.invoke(
            app, ["transcribe", str(audio), "--lang", "yo", "--base-url", server.base_url, "--json"]
        )
        record = json.loads(result.output)
        assert len(record["chunks"]) == 3
        assert record["chunks"][-1]["end_s"] == pytest.approx(65.0, abs=0.05)
        # every chunk says "ina kwana": the two-word seam overlaps are merged away
        assert record["text"].count("ina kwana") < 3

    def test_multiple_files_write_jsonl(self, server: FakeModelServer, tmp_path: Path) -> None:
        a, b = sine_wav(tmp_path / "a.wav"), sine_wav(tmp_path / "b.wav")
        out = tmp_path / "out.jsonl"
        result = runner.invoke(
            app,
            [
                "transcribe",
                str(a),
                str(b),
                "-l",
                "ig",
                "--base-url",
                server.base_url,
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0
        lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
        assert [Path(x["file"]).name for x in lines] == ["a.wav", "b.wav"]
        assert all(x["lang"] == "ig" for x in lines)

    def test_one_bad_file_does_not_stop_the_rest_but_sets_the_exit_code(
        self, server: FakeModelServer, tmp_path: Path
    ) -> None:
        good, missing = sine_wav(tmp_path / "a.wav"), tmp_path / "nope.wav"
        result = runner.invoke(
            app, ["transcribe", str(missing), str(good), "-l", "ha", "--base-url", server.base_url]
        )
        assert result.exit_code == 1
        assert "ina kwana" in result.output
        assert "nope.wav" in result.output

    def test_bad_language(self, server: FakeModelServer, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "transcribe",
                str(sine_wav(tmp_path / "a.wav")),
                "-l",
                "fr",
                "--base-url",
                server.base_url,
            ],
        )
        assert result.exit_code != 0
        assert "Unsupported language" in str(result.exception)


class TestErrorReporting:
    def run_main(self, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
        monkeypatch.setattr(sys, "argv", ["atlasforge", *argv])
        with pytest.raises(SystemExit) as info:
            main()
        return int(info.value.code or 0)

    def test_missing_base_url_exits_2_with_message_and_hint(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("ATLASFORGE_BASE_URL", raising=False)
        code = self.run_main(monkeypatch, "run", "hi")
        assert code == 2
        err = capsys.readouterr().err
        assert "error: The openai backend needs a server URL." in err
        assert "--base-url" in err

    def test_unreachable_server_exits_2_without_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = self.run_main(
            monkeypatch, "run", "hi", "--base-url", "http://127.0.0.1:9/v1", "--retries", "0"
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "Could not reach" in err
        assert "Traceback" not in err

    def test_missing_dataset_file(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        code = self.run_main(monkeypatch, "dataset", "validate", str(tmp_path / "nope.jsonl"))
        assert code == 2
        assert "cannot read" in capsys.readouterr().err
