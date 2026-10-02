"""The documentation is tested like code: it may not show a flag that does not exist.

These checks need no docs toolchain. The full strict site build runs separately in CI.
"""

import importlib.util
import os
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import typer.main

from atlasforge.cli import app
from atlasforge.demo import build_demo

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md"))
FENCE = re.compile(r"^```([\w+-]*)[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
SHELL_LANGS = {"bash", "sh", "shell", "powershell", ""}


def blocks(path: Path) -> Iterator[tuple[str, str]]:
    for match in FENCE.finditer(path.read_text(encoding="utf-8")):
        yield match.group(1).lower(), match.group(2)


def command_lines(path: Path) -> Iterator[str]:
    """Every ``atlasforge ...`` command in a page's shell blocks, continuations joined."""
    for lang, body in blocks(path):
        if lang not in SHELL_LANGS:
            continue
        joined = re.sub(r"(\\|`)\s*\n\s*", " ", body)
        for raw in joined.splitlines():
            line = raw.strip().removeprefix("$ ").strip()
            if line.startswith("atlasforge ") or line == "atlasforge":
                yield line


def option_names(command: Any) -> set[str]:
    names = {"--help"}
    for param in command.params:
        if param.param_type_name == "option":
            names.update(param.opts)
            names.update(getattr(param, "secondary_opts", []))
    return names


def check_command(line: str) -> list[str]:
    """Problems with one documented command line (empty list means it is valid)."""
    line = re.split(r"\s(?:\||>|&&)\s", line)[0]
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        return [f"cannot parse {line!r}: {exc}"]
    command: Any = typer.main.get_command(app)
    path = ["atlasforge"]
    rest = tokens[1:]
    while rest and getattr(command, "commands", None) and not rest[0].startswith("-"):
        name = rest[0]
        if name not in command.commands:
            return [f"unknown command {' '.join([*path, name])!r}"]
        command, path, rest = command.commands[name], [*path, name], rest[1:]

    allowed = option_names(command) | ({"--version", "-V"} if len(path) == 1 else set())
    problems = []
    for token in rest:
        if token.startswith("-") and token not in ("-", "--") and re.match(r"^--?[A-Za-z]", token):
            flag = token.split("=", 1)[0]
            if flag not in allowed:
                problems.append(f"{' '.join(path)} has no option {flag!r}")
    return problems


ALL_COMMANDS = [(page, line) for page in PAGES for line in command_lines(page)]


def test_the_docs_actually_contain_commands() -> None:
    assert len(ALL_COMMANDS) > 25, "the command extractor found almost nothing; is it broken?"


@pytest.mark.parametrize(
    ("page", "line"), ALL_COMMANDS, ids=lambda v: v.name if isinstance(v, Path) else v[:60]
)
def test_every_documented_command_uses_real_commands_and_flags(page: Path, line: str) -> None:
    assert check_command(line) == [], f"{page.relative_to(ROOT)}: {line}"


def test_the_checker_itself_catches_a_bad_flag_and_a_bad_command() -> None:
    assert check_command("atlasforge eval data.jsonl --out x --nonsense") != []
    assert check_command("atlasforge frobnicate") != []
    assert check_command("atlasforge dataset validate d.jsonl --against t.jsonl --json") == []
    assert check_command("atlasforge --version") == []
    assert check_command("atlasforge run - --no-send-repetition-penalty") == []
    assert check_command("atlasforge doctor | cat") == []


# ------------------------------------------------------------------ runnable Python snippets


def runnable_snippets() -> list[tuple[Path, str]]:
    return [
        (page, body)
        for page in PAGES
        for lang, body in blocks(page)
        if lang == "python" and body.lstrip().startswith("# docs:run")
    ]


@pytest.mark.parametrize(
    ("page", "code"),
    runnable_snippets(),
    ids=lambda v: v.name if isinstance(v, Path) else "snippet",
)
def test_runnable_python_snippets_run_against_the_demo_data(
    page: Path, code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_demo(tmp_path / "atlasforge-demo")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(page), "exec"), {})  # noqa: S102 - our own documentation


def test_there_are_runnable_snippets() -> None:
    assert len(runnable_snippets()) >= 2


# ------------------------------------------------------------------------ site structure


def test_every_page_is_in_the_navigation_and_every_nav_entry_exists() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    in_nav = set(re.findall(r"[\w./-]+\.md", config))
    on_disk = {p.relative_to(DOCS).as_posix() for p in PAGES}
    assert on_disk - in_nav == set(), "pages missing from mkdocs.yml nav"
    assert in_nav - on_disk == set(), "nav entries with no file"


def test_pages_do_not_contain_unescaped_template_syntax() -> None:
    """The docs are rendered through Jinja; a stray double brace would be eaten."""
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for match in re.finditer(r"\{\{(.*?)\}\}", text):
            assert re.match(r"\s*\w+\(.*\)\s*$|\s*version\s*$", match.group(1)), (
                f"{page.name}: {match.group(0)}"
            )


# ---------------------------------------------------------------- generated sections work


class FakeEnv:
    """Just enough of the mkdocs-macros environment to register and call macros."""

    def __init__(self) -> None:
        self.variables: dict[str, Any] = {}
        self.macros: dict[str, Any] = {}

    def macro(self, function: Any) -> Any:
        self.macros[function.__name__] = function
        return function


@pytest.fixture(scope="module")
def macros() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("docs_macros", ROOT / "docs_macros.py")
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = FakeEnv()
    module.define_env(env)
    return env.macros


def test_cli_reference_covers_every_command_and_option(macros: dict[str, Any]) -> None:
    text = macros["cli_reference"]()
    root: Any = typer.main.get_command(app)
    for name, command in root.commands.items():
        if getattr(command, "commands", None):
            for sub in command.commands:
                assert f"### `atlasforge {name} {sub}`" in text
        else:
            assert f"### `atlasforge {name}`" in text
    assert "`--send-repetition-penalty` / `--no-send-repetition-penalty`" in text
    assert "ATLASFORGE_BASE_URL" in text


def test_reference_tables_render(macros: dict[str, Any]) -> None:
    assert "`chrf++`" in macros["metrics_table"]()
    assert "| `lora_r` | `16` |" in macros["fine_tune_settings"]()
    assert "`BackendTimeout`" in macros["errors_table"]()
    assert "tone-insensitive" in macros["normalize_demo"]().lower()


def test_transcripts_are_real_and_os_independent(macros: dict[str, Any]) -> None:
    previous = Path.cwd()
    try:
        compare = macros["transcript"]("compare")
        dry_run = macros["transcript"]("dry_run")
    finally:
        os.chdir(previous)
    assert "Regressions" in compare
    assert "numeracy" in compare
    assert "\\" not in compare
    assert "Dry run: nothing was trained." in dry_run
    assert "\\" not in dry_run


def test_file_examples_are_real_json(macros: dict[str, Any]) -> None:
    for name in ("dataset", "results", "run", "report", "comparison"):
        text = macros["file_example"](name)
        assert text.startswith("```json")
        assert text.endswith("```")
    assert '"synthetic-demo/base"' in macros["file_example"]("run")
