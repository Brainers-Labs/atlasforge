"""doctor checks, with injected fakes: no GPU, network or real environment needed."""

import subprocess
from pathlib import Path

from atlasforge.doctor import (
    Check,
    check_disk,
    check_extras,
    check_ffmpeg,
    check_gpu,
    check_hf_token,
    check_python,
    exit_code,
)


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestPython:
    def test_ok(self) -> None:
        assert check_python((3, 12, 1)).status == "ok"

    def test_too_old_fails(self) -> None:
        result = check_python((3, 9, 0))
        assert result.status == "fail"
        assert result.hint is not None


class TestFfmpeg:
    def test_found(self) -> None:
        assert check_ffmpeg(lambda _: "/usr/bin/ffmpeg").status == "ok"

    def test_missing_warns_with_install_hint(self) -> None:
        result = check_ffmpeg(lambda _: None)
        assert result.status == "warn"
        assert result.hint is not None
        assert "winget" in result.hint


class TestHfToken:
    def test_env_token_is_masked(self, tmp_path: Path) -> None:
        result = check_hf_token({"HF_TOKEN": "hf_abcdefghijkl"}, tmp_path)
        assert result.status == "ok"
        assert "hf_****ijkl" in result.detail
        assert "abcdefgh" not in result.detail

    def test_cached_login(self, tmp_path: Path) -> None:
        cache = tmp_path / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "token").write_text("x")
        assert check_hf_token({}, tmp_path).status == "ok"

    def test_missing_warns(self, tmp_path: Path) -> None:
        result = check_hf_token({}, tmp_path)
        assert result.status == "warn"
        assert result.hint is not None
        assert "NCAIR1" in result.hint

    def test_hf_home_respected(self, tmp_path: Path) -> None:
        (tmp_path / "token").write_text("x")
        assert check_hf_token({"HF_HOME": str(tmp_path)}, Path("/nonexistent")).status == "ok"


class TestExtras:
    def test_all_present(self) -> None:
        assert all(c.status == "ok" for c in check_extras(lambda _: object()))

    def test_all_missing_gives_install_hints(self) -> None:
        checks = check_extras(lambda _: None)
        assert {c.name for c in checks} == {"extra:local", "extra:asr", "extra:finetune"}
        local = next(c for c in checks if c.name == "extra:local")
        assert local.status == "warn"
        assert local.hint == 'pip install "atlasforge[local]"'

    def test_partial(self) -> None:
        checks = check_extras(lambda name: None if name == "librosa" else object())
        asr = next(c for c in checks if c.name == "extra:asr")
        assert asr.status == "warn"
        assert "librosa" in asr.detail


class TestGpu:
    def test_no_nvidia_smi(self) -> None:
        assert check_gpu(lambda _: None).status == "warn"

    def test_big_gpu_ok(self) -> None:
        result = check_gpu(
            lambda _: "nvidia-smi", lambda *a, **k: _completed("NVIDIA A10G, 24576\n")
        )
        assert result.status == "ok"
        assert "24 GB" in result.detail

    def test_medium_gpu_suggests_4bit(self) -> None:
        result = check_gpu(lambda _: "nvidia-smi", lambda *a, **k: _completed("RTX 3060, 12288\n"))
        assert result.status == "warn"
        assert result.hint is not None
        assert "4-bit" in result.hint

    def test_small_gpu_suggests_endpoint(self) -> None:
        result = check_gpu(lambda _: "nvidia-smi", lambda *a, **k: _completed("GT 1030, 2048\n"))
        assert result.status == "warn"
        assert result.hint is not None
        assert "endpoint" in result.hint

    def test_multi_gpu_uses_largest(self) -> None:
        out = "T4, 15360\nA100, 40960\n"
        result = check_gpu(lambda _: "nvidia-smi", lambda *a, **k: _completed(out))
        assert result.status == "ok"

    def test_gpu_name_with_comma(self) -> None:
        out = "NVIDIA GeForce RTX 4090, Laptop, 16384\n"
        result = check_gpu(lambda _: "nvidia-smi", lambda *a, **k: _completed(out))
        assert result.status == "ok"
        assert "NVIDIA GeForce RTX 4090, Laptop" in result.detail

    def test_query_failure_is_a_warning_not_a_crash(self) -> None:
        def boom(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            raise OSError("nope")

        assert check_gpu(lambda _: "nvidia-smi", boom).status == "warn"

    def test_garbage_output_ignored(self) -> None:
        result = check_gpu(lambda _: "nvidia-smi", lambda *a, **k: _completed("garbage\n"))
        assert result.status == "warn"


class TestDisk:
    def test_real_path_returns_a_check(self, tmp_path: Path) -> None:
        assert check_disk(tmp_path).status in {"ok", "warn"}

    def test_unreadable_path_warns(self, tmp_path: Path) -> None:
        assert check_disk(tmp_path / "does" / "not" / "exist").status == "warn"


class TestExitCode:
    def test_warnings_do_not_fail(self) -> None:
        assert exit_code([Check("a", "warn", "x"), Check("b", "ok", "y")]) == 0

    def test_failure_fails(self) -> None:
        assert exit_code([Check("a", "fail", "x")]) == 1
