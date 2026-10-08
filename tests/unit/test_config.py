"""`atlasforge.toml`: what it may set, what it may not, and which wins.

The command-line tests use a recording backend, so they assert what the command *asked for*
rather than what a model would have answered.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from atlasforge import config
from atlasforge.cli import app
from atlasforge.doctor import check_config
from atlasforge.errors import ConfigError
from atlasforge.types import BackendInfo, Generation, GenParams, Message

runner = CliRunner()


def write(directory: Path, text: str, name: str = config.FILE_NAME) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def settings(**values: Any) -> str:
    """A project file body for the given values, spelled the way a user would write it."""
    lines = [
        f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value}"
        for key, value in values.items()
    ]
    return "\n".join([f"[{config.TABLE}]", *lines, ""])


class TestReading:
    def test_no_file_is_an_empty_config(self, tmp_path: Path) -> None:
        loaded = config.load_config(tmp_path)
        assert loaded == config.Config()
        assert not loaded
        assert config.find_file(tmp_path) is None

    def test_a_file_sets_values(self, tmp_path: Path) -> None:
        path = write(tmp_path, settings(backend="openai", model="N-ATLaS-local"))
        loaded = config.load_config(tmp_path)
        assert loaded.path == path
        assert loaded.values == {"backend": "openai", "model": "N-ATLaS-local"}
        assert loaded

    def test_the_nearest_file_wins(self, tmp_path: Path) -> None:
        write(tmp_path, settings(model="outer"))
        child = tmp_path / "packages" / "app"
        write(child, settings(model="inner"))
        assert config.load_config(child).values == {"model": "inner"}

    def test_a_file_in_a_parent_directory_is_found(self, tmp_path: Path) -> None:
        write(tmp_path, settings(backend="local"))
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert config.load_config(deep).values == {"backend": "local"}

    def test_other_tables_belong_to_other_tools(self, tmp_path: Path) -> None:
        write(tmp_path, '[tool.ruff]\nline-length = 100\n\n[atlasforge]\nmodel = "m"\n')
        assert config.load_config(tmp_path).values == {"model": "m"}

    def test_an_integer_timeout_means_seconds(self, tmp_path: Path) -> None:
        write(tmp_path, settings(timeout=60))
        assert config.load_config(tmp_path).values == {"timeout": 60.0}

    def test_an_explicit_file_can_be_read(self, tmp_path: Path) -> None:
        path = write(tmp_path, settings(device="cpu"))
        assert config.load_file(path).values == {"device": "cpu"}

    def test_a_missing_file_named_explicitly_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as caught:
            config.load_file(tmp_path / "nope.toml")
        assert caught.value.hint is not None


class TestRefusals:
    def test_an_unknown_key_names_the_valid_ones(self, tmp_path: Path) -> None:
        """A typo that silently does nothing is worse than one that stops the run."""
        write(tmp_path, settings(tempurature=0.1))
        with pytest.raises(ConfigError) as caught:
            config.load_config(tmp_path)
        assert "tempurature" in str(caught.value)
        assert "backend" in (caught.value.hint or "")

    def test_a_wrong_type_says_what_to_write(self, tmp_path: Path) -> None:
        write(tmp_path, '[atlasforge]\nretries = "twice"\n')
        with pytest.raises(ConfigError) as caught:
            config.load_config(tmp_path)
        assert "retries must be a whole number, not str" in str(caught.value)
        assert "retries = 2" in (caught.value.hint or "")

    def test_a_boolean_is_not_a_number(self, tmp_path: Path) -> None:
        write(tmp_path, settings(retries="true"))
        with pytest.raises(ConfigError):
            config.load_config(tmp_path)

    def test_a_file_that_is_not_toml_says_so(self, tmp_path: Path) -> None:
        write(tmp_path, "[atlasforge\nbackend = \n")
        with pytest.raises(ConfigError) as caught:
            config.load_config(tmp_path)
        assert "not valid TOML" in str(caught.value)

    def test_the_table_must_be_a_table(self, tmp_path: Path) -> None:
        write(tmp_path, 'atlasforge = "openai"\n')
        with pytest.raises(ConfigError) as caught:
            config.load_config(tmp_path)
        assert "not a table" in str(caught.value)


class Recording:
    """A backend that stands in for a real one, so the test can read the resolved options."""

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, messages: Sequence[Message], params: GenParams | None = None) -> Generation:
        return Generation(text="ok", latency_ms=1.0, finish_reason="stop")

    def info(self) -> BackendInfo:
        return BackendInfo(backend=self.name, model="m")

    def close(self) -> None:
        pass


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_build(name: str, **options: Any) -> Recording:
        seen["name"] = name
        seen.update(options)
        return Recording(name)

    monkeypatch.setattr("atlasforge.cli.build_backend", fake_build)
    return seen


class TestOnTheCommandLine:
    def test_a_config_file_supplies_what_the_flag_did_not(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, settings(backend="openai", base_url="http://127.0.0.1:9000/v1", model="X"))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "hello"])
        assert result.exit_code == 0, result.output
        assert recorded["name"] == "openai"
        assert recorded["base_url"] == "http://127.0.0.1:9000/v1"
        assert recorded["model"] == "X"

    def test_a_flag_beats_the_file(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, settings(model="from-the-file"))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "hello", "--model", "from-the-flag"])
        assert result.exit_code == 0, result.output
        assert recorded["model"] == "from-the-flag"

    def test_the_environment_variable_beats_the_file(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, settings(base_url="http://from-the-file/v1"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ATLASFORGE_BASE_URL", "http://from-the-env/v1")
        result = runner.invoke(app, ["run", "hello"])
        assert result.exit_code == 0, result.output
        assert recorded["base_url"] == "http://from-the-env/v1"

    def test_a_broken_file_stops_the_run_with_the_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(tmp_path, settings(tempurature=0.1))
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "hello", "--base-url", "http://x/v1"])
        assert isinstance(result.exception, ConfigError)
        assert "tempurature" in str(result.exception)

    def test_a_config_file_that_is_not_there_changes_nothing(
        self, tmp_path: Path, recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "hello", "--base-url", "http://x/v1"])
        assert result.exit_code == 0, result.output
        assert recorded["base_url"] == "http://x/v1"


class TestInDoctor:
    def test_no_file_is_reported_as_such(self) -> None:
        check = check_config(lambda: None)
        assert check.status == "ok"
        assert config.FILE_NAME in check.detail

    def test_a_file_is_named_with_its_settings(self, tmp_path: Path) -> None:
        path = write(tmp_path, settings(model="X", timeout=30))
        check = check_config(lambda: path)
        assert check.status == "ok"
        assert str(path) in check.detail
        assert "model=X" in check.detail
        assert "timeout=30.0" in check.detail

    def test_a_broken_file_fails_the_check(self, tmp_path: Path) -> None:
        path = write(tmp_path, settings(nope="x"))
        check = check_config(lambda: path)
        assert check.status == "fail"
        assert check.hint is not None
