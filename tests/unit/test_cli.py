import json
import subprocess
import sys

from typer.testing import CliRunner

from atlasforge import __version__
from atlasforge.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "doctor" in result.output


def test_doctor_json_is_machine_readable() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    checks = json.loads(result.stdout)
    assert {"python", "gpu", "disk", "ffmpeg", "hf-token"} <= {c["name"] for c in checks}
    assert all(set(c) == {"name", "status", "detail", "hint"} for c in checks)
    assert result.exit_code in (0, 1)


def test_doctor_table_shows_extras_hint_without_markup_errors() -> None:
    result = runner.invoke(app, ["doctor"])
    assert "python" in result.output.lower()
    # square brackets in hints such as atlasforge[local] must not be eaten as rich markup
    if "extra:local" in result.output and "missing" in result.output:
        assert "atlasforge[local]" in result.output


def test_importing_cli_does_not_import_torch() -> None:
    code = "import sys, atlasforge.cli; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
