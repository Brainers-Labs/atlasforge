"""Command-line interface. Thin over the Python API.

No ``from __future__ import annotations`` here on purpose: Typer reads the
annotations at runtime to build the CLI.
"""

import json
from dataclasses import asdict
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from atlasforge import __version__
from atlasforge.doctor import Check, exit_code, run_checks
from atlasforge.errors import AtlasForgeError

app = typer.Typer(
    name="atlasforge",
    help="Run, evaluate, compare and fine-tune the official N-ATLaS models.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

_STYLE = {"ok": "green", "warn": "yellow", "fail": "bold red"}
_LABEL = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"atlasforge {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
        ),
    ] = False,
) -> None:
    """AtlasForge command-line interface."""


def _render(checks: list[Check]) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Detail")
    for check in checks:
        table.add_row(
            Text(_LABEL[check.status], style=_STYLE[check.status]),
            Text(check.name),
            Text(check.detail),
        )
    console.print(table)
    for check in checks:
        if check.hint and check.status != "ok":
            console.print(Text(f"{check.name}: {check.hint}", style="dim"), soft_wrap=True)


@app.command()
def doctor(
    json_output: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON.")
    ] = False,
) -> None:
    """Check Python, GPU, disk, ffmpeg, Hugging Face token and optional extras."""
    checks = run_checks()
    if json_output:
        typer.echo(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        _render(checks)
    raise typer.Exit(exit_code(checks))


def main() -> None:
    """Console-script entry point: turn AtlasForge errors into clean output."""
    try:
        app()
    except AtlasForgeError as exc:
        typer.echo(f"error: {exc.format()}", err=True)
        raise SystemExit(2) from exc
