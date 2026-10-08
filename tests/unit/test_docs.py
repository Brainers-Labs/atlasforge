"""The documentation is tested like code: it may not show a flag that does not exist.

These checks need no docs toolchain. The full strict site build runs separately in CI.
"""

import ast
import importlib.util
import os
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import typer.main

import atlasforge
from atlasforge.cli import app
from atlasforge.config import KEYS
from atlasforge.demo import build_demo
from atlasforge.eval.flags import FLAG_NAMES
from atlasforge.livecheck import SECRET_ENV

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


def python_blocks() -> list[tuple[Path, str]]:
    return [(page, body) for page in PAGES for lang, body in blocks(page) if lang == "python"]


@pytest.mark.parametrize(
    ("page", "code"),
    python_blocks(),
    ids=lambda v: v.name if isinstance(v, Path) else "snippet",
)
def test_every_python_block_is_valid_python(page: Path, code: str) -> None:
    """An example we cannot run here (it needs a model) must still parse as Python."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(f"{page.relative_to(ROOT)}: {exc}")


# ------------------------------------------------------------------------ site structure


def test_every_page_is_in_the_navigation_and_every_nav_entry_exists() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    in_nav = set(re.findall(r"[\w./-]+\.md", config))
    on_disk = {p.relative_to(DOCS).as_posix() for p in PAGES}
    assert on_disk - in_nav == set(), "pages missing from mkdocs.yml nav"
    assert in_nav - on_disk == set(), "nav entries with no file"


def test_every_guide_is_listed_on_the_guides_index() -> None:
    """The nav has a check; the guides index is the other way in, and it is hand-written.

    A guide reachable only from the sidebar is a guide half the readers never see — and it is the
    index, not the nav, that says what each one is *for*.
    """
    index = (DOCS / "guides" / "index.md").read_text(encoding="utf-8")
    guides = sorted((DOCS / "guides").glob("*.md"))
    missing = [p.name for p in guides if p.name != "index.md" and f"({p.name})" not in index]
    assert missing == [], f"guides missing from the guides index: {missing}"


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


def test_the_flags_table_covers_every_flag(macros: dict[str, Any]) -> None:
    """The concepts page cannot quietly omit a flag: the table is the registry."""
    table = macros["flags_table"]()
    for name in FLAG_NAMES:
        assert f"| `{name}` |" in table
    assert "not language ID" in table


def test_the_config_table_covers_every_key(macros: dict[str, Any]) -> None:
    """The configuration page cannot document a key the loader would reject, or miss one."""
    table = macros["config_table"]()
    for key in KEYS:
        assert f"| `{key}` |" in table
    assert "whole number" in table


def test_the_api_reference_names_every_entry_point() -> None:
    """A new top-level entry point that nobody documents is a feature with no front door.

    The `::: ` directives are checked by the site build (an unknown module or member fails it), but
    the *choice* of what to list is manual — this pins the names ``atlasforge.__all__`` promises.
    """
    page = (DOCS / "reference" / "python-api.md").read_text(encoding="utf-8")
    missing = [n for n in atlasforge.__all__ if not n.startswith("_") and n not in page]
    assert missing == [], f"entry points missing from the Python API reference: {missing}"


def test_the_api_reference_lists_documentable_eval_modules() -> None:
    """The eval subpackage is documented module by module, so a new one should not be missed.

    This is what caught ``metrics`` and ``format`` — the functions behind every score and every
    rendered figure — being absent while eight of their siblings were listed.
    """
    page = (DOCS / "reference" / "python-api.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"::: (\S+)", page))
    on_disk = {
        f"atlasforge.eval.{path.stem}"
        for path in (ROOT / "src" / "atlasforge" / "eval").glob("*.py")
        if not path.stem.startswith("_")
    }
    assert on_disk - listed == set(), "eval modules with no API reference entry"


# ---------------------------------------------------------------- files the tool writes

ARTIFACT = re.compile(r"^[\w.-]+\.(json|jsonl|md|html)$")

#: Modules whose file constants are sample *data* rather than a format AtlasForge defines; the
#: dataset format itself is documented as `.jsonl`. Named rather than pattern-matched so the
#: exclusion is visible and stays small.
NOT_A_FORMAT = {"src/atlasforge/demo.py"}


def documented_artifacts() -> dict[str, set[str]]:
    """Every ``NAME: Final = "...json"``-style constant in ``src/``, by value.

    Read out of the source rather than listed here, so a new file a command writes is picked up
    without anyone remembering to add it to this test.
    """
    found: dict[str, set[str]] = {}
    root = ROOT / "src"
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in NOT_A_FORMAT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            if ARTIFACT.match(value.value):
                found.setdefault(value.value.split("/")[-1], set()).add(relative)
    return found


def test_every_file_the_tool_writes_is_documented_in_the_formats_page() -> None:
    """``bench.json`` and ``bench.md`` were written for a day without appearing here.

    The page is the reference for "what does this thing put on my disk", and it was complete for
    every command except the newest one — the same shape of gap as an undocumented extra.
    """
    page = (DOCS / "reference" / "file-formats.md").read_text(encoding="utf-8")
    missing = {name: where for name, where in documented_artifacts().items() if name not in page}
    assert missing == {}, f"files written by the code but absent from file-formats.md: {missing}"


def test_the_artifact_scan_finds_the_files_we_know_about() -> None:
    """A regex that silently matched nothing would make the check above vacuous."""
    found = documented_artifacts()
    assert {"report.json", "run.json", "bench.json"} <= set(found)
    assert "toy_qa.jsonl" not in found, "the demo's data files are not a format"


# ---------------------------------------------------------------- environment variables


def test_the_credentials_the_tool_reads_are_documented() -> None:
    """``SECRET_ENV`` is the tool's own list of what must never reach a log; the page must agree.

    The environment page is hand-written, so this pins the part of it that is a promise rather
    than prose: the names in the redaction list, and the one variable the CLI reads as a default.
    """
    page = (DOCS / "reference" / "environment.md").read_text(encoding="utf-8")
    missing = [name for name in (*SECRET_ENV, "ATLASFORGE_BASE_URL") if f"`{name}`" not in page]
    assert missing == [], f"variables missing from the environment reference: {missing}"


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
