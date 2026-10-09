import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from atlasforge.backends.factory import build_backend
from atlasforge.backends.local import LocalBackend
from atlasforge.backends.openai import OpenAIBackend
from atlasforge.errors import ConfigError
from atlasforge.eval.dataset import load_dataset
from atlasforge.eval.format import fmt_bound, fmt_delta, fmt_value
from atlasforge.eval.report import to_markdown
from atlasforge.eval.runner import Record
from atlasforge.eval.score import ScoreReport, score_run


class TestFactory:
    def test_openai_needs_a_url_with_a_helpful_hint(self) -> None:
        with pytest.raises(ConfigError, match="server URL") as info:
            build_backend("openai")
        assert "--base-url" in (info.value.hint or "")

    def test_builds_openai(self) -> None:
        backend = build_backend("openai", base_url="http://127.0.0.1:8000/v1", model="m")
        assert isinstance(backend, OpenAIBackend)
        assert backend.info().model == "m"

    def test_builds_local_without_importing_torch(self) -> None:
        code = (
            "import sys; from atlasforge.backends.factory import build_backend;"
            "b = build_backend('local'); print(type(b).__name__);"
            "sys.exit(1 if 'torch' in sys.modules else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0
        assert "LocalBackend" in result.stdout

    def test_local_options_are_passed_through(self) -> None:
        backend = build_backend("local", model="me/x", quantize="4bit", device="cpu")
        assert isinstance(backend, LocalBackend)
        assert backend.info().model == "me/x"
        assert backend.info().device == "cpu"

    def test_unknown_backend(self) -> None:
        with pytest.raises(ConfigError, match="Unknown backend") as info:
            build_backend("gpt")
        assert "openai" in (info.value.hint or "")

    def test_default_model_is_the_official_llm(self) -> None:
        assert (
            build_backend("openai", base_url="http://127.0.0.1/v1").info().model == "NCAIR1/N-ATLaS"
        )


class TestFormat:
    def test_fraction_metrics_are_percentages(self) -> None:
        assert fmt_value("exact_match", 0.714) == "71.4%"
        assert fmt_value("wer", 0.2) == "20.0%"
        assert fmt_delta("accuracy", 0.085) == "+8.5 pts"
        assert fmt_delta("wer", -0.05) == "-5.0 pts"
        assert fmt_bound("macro_f1", 0.031) == "+3.1"

    def test_chrf_is_shown_as_is(self) -> None:
        assert fmt_value("chrf", 54.32) == "54.3"
        assert fmt_delta("chrf", 3.21) == "+3.2"
        assert fmt_bound("chrf++", -1.05) == "-1.1"

    def test_a_strict_variant_formats_like_its_base(self) -> None:
        """Otherwise a strict metric would sit beside its loose twin as a bare 0-1 number."""
        assert fmt_value("accuracy_strict", 0.714) == "71.4%"
        assert fmt_delta("macro_f1_strict", 0.085) == "+8.5 pts"
        assert fmt_value("chrf_strict", 54.32) == "54.3"  # not a fraction, strict or not

    def test_none_is_a_dash(self) -> None:
        assert fmt_value("chrf", None) == "-"
        assert fmt_delta("wer", None) == "-"
        assert fmt_bound("wer", None) == "-"


class TestSingleRunReport:
    def scored(self, tmp_path: Path) -> tuple[ScoreReport, dict[str, Any]]:
        path = tmp_path / "d.jsonl"
        rows = [
            {"id": str(i), "input": f"q{i}", "reference": "wà" if i else "ok"} for i in range(4)
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        ds = load_dataset(path, "generation")
        results = {
            "0": Record(id="0", prediction="ok", latency_ms=100.0),
            "1": Record(id="1", prediction="wa", latency_ms=200.0),
            "2": Record(id="2", prediction="wà", latency_ms=300.0),
            "3": Record(id="3", error="BackendTimeout: x"),
        }
        manifest = {"model": "NCAIR1/N-ATLaS", "revision": "abcdef1234567890", "backend": "local"}
        return score_run(ds, results, metrics=["exact_match"]), manifest

    def test_report_contents(self, tmp_path: Path) -> None:
        report, manifest = self.scored(tmp_path)
        text = to_markdown(report, manifest)
        assert "# AtlasForge evaluation report" in text
        assert "(3 ok, 1 failed, 0 missing)" in text
        assert "NCAIR1/N-ATLaS" in text
        assert "`abcdef1234`" in text
        assert "| exact_match | tone-aware |" in text
        assert "| exact_match | tone-insensitive |" in text
        assert "mean 200 ms" in text
        assert "max 300 ms" in text
        assert "both views" in text

    def test_tone_views_differ_in_the_table(self, tmp_path: Path) -> None:
        report, manifest = self.scored(tmp_path)
        text = to_markdown(report, manifest)
        aware = next(
            line for line in text.splitlines() if "tone-aware" in line and "exact_match" in line
        )
        insensitive = next(
            line
            for line in text.splitlines()
            if "tone-insensitive" in line and "exact_match" in line
        )
        assert "50.0%" in aware  # ok + wà correct, wa wrong, failed wrong -> 2/4
        assert "75.0%" in insensitive  # wa now correct -> 3/4

    def test_works_without_a_manifest(self, tmp_path: Path) -> None:
        report, _ = self.scored(tmp_path)
        text = to_markdown(report)
        assert "**Model:** ?" in text

    def test_no_latency_section_when_nothing_succeeded(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"id": "a", "input": "q", "reference": "x"}) + "\n", encoding="utf-8"
        )
        ds = load_dataset(path, "generation")
        report = score_run(ds, {"a": Record(id="a", error="boom")}, metrics=["exact_match"])
        assert "## Latency" not in to_markdown(report)
