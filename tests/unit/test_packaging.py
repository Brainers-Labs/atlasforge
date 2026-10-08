"""The distribution name is not the import name, so nothing may blur the two.

``pyproject.toml`` declares ``brainers-atlasforge`` because the plain ``atlasforge`` belongs to
an unrelated project on PyPI. That makes an install hint a small hazard rather than a cosmetic
string: ``pip install "atlasforge[local]"`` is not a typo, it reaches the *other* package and
would then fail on an extra it does not have. The rename that chose the org-scoped name missed
five such hints in ``src/`` and the assertions that pinned them, which is why this file exists.

The name is read from ``pyproject.toml`` — the one place it is declared — rather than repeated,
so renaming it again cannot leave a stale string behind in the source, the docs or an extra.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

#: A quoted install command. Unquoted mentions are prose about the other project on PyPI
#: ("`pip install atlasforge` would install that one") and are deliberately not matched.
INSTALL = re.compile(r'pip install\s+(?:--pre\s+)?"([A-Za-z0-9._-]+)(?:\[[^\]]*\])?"')

#: Anything named like us: the bare name, or one of our extras.
OURS = re.compile(r"^atlasforge(-[A-Za-z0-9._-]+)?$")

TEXT_FILES = [
    *(ROOT / "src").rglob("*.py"),
    *(ROOT / "docs").rglob("*.md"),
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
]


def project() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        parsed: dict[str, Any] = tomllib.load(handle)
    table = parsed["project"]
    assert isinstance(table, dict), "pyproject.toml has no [project] table"
    return table


def distribution() -> str:
    """The name that goes to PyPI."""
    return str(project()["name"])


def install_commands() -> list[tuple[Path, str, str]]:
    """Every ``pip install "..."`` in the source and the docs, as (file, line, name)."""
    found: list[tuple[Path, str, str]] = []
    for path in TEXT_FILES:
        lines = path.read_text(encoding="utf-8").splitlines()
        found.extend(
            (path, str(number), match.group(1))
            for number, line in enumerate(lines, start=1)
            for match in INSTALL.finditer(line)
        )
    return found


def wrong_name(text: str) -> str | None:
    """The first name in ``text`` that looks like ours but is not, or ``None``.

    Separate from the file walk so the rule itself can be tested without a bad file on disk.
    """
    declared = distribution()
    for match in INSTALL.finditer(text):
        candidate = match.group(1)
        if candidate != declared and OURS.match(candidate):
            return candidate
    return None


def test_the_distribution_is_not_the_import_name() -> None:
    """The whole point of the rename: ``pip install atlasforge`` must not be what we tell people."""
    assert distribution() == "brainers-atlasforge"
    assert distribution() != "atlasforge", "the plain name is taken on PyPI; see the docstring"


def test_the_checker_itself_catches_the_wrong_name() -> None:
    assert wrong_name('pip install "atlasforge[local]"') == "atlasforge"
    assert wrong_name('pip install "brainers-atlasforge[local]"') is None
    assert wrong_name('pip install "vllm"') is None
    # Prose about the other project is not an install command and must not be flagged.
    assert wrong_name("`pip install atlasforge` would install that one") is None


def test_no_install_command_names_another_project() -> None:
    bad = [
        f'{path.relative_to(ROOT)}:{line}: pip install "{found}"'
        for path, line, found in install_commands()
        if found != distribution() and OURS.match(found)
    ]
    assert bad == [], f"these install commands name a different distribution: {bad}"


def test_there_are_install_commands_to_check() -> None:
    """A regex that silently matches nothing would make the check above vacuous."""
    assert len(install_commands()) >= 3


def test_every_extra_that_refers_to_us_uses_the_declared_name() -> None:
    """An extra is resolved by the installer, so a stale one installs a stranger's package."""
    extras: dict[str, list[str]] = project()["optional-dependencies"]
    bad = [
        f"{extra}: {requirement}"
        for extra, requirements in extras.items()
        for requirement in requirements
        if requirement.split("[", 1)[0].split(">=", 1)[0].startswith("atlasforge")
        and not requirement.startswith(distribution())
    ]
    assert bad == [], f"self-referencing extras name the wrong distribution: {bad}"


@pytest.mark.parametrize("extra", ["local", "asr", "finetune", "bench", "docs", "dev"])
def test_the_extras_the_hints_tell_people_to_install_exist(extra: str) -> None:
    """Every ``pip install "...[extra]"`` in the source must be an extra we actually declare."""
    assert extra in project()["optional-dependencies"]


def test_every_extra_is_offered_on_the_installation_page() -> None:
    """An extra that only its own guide mentions is one nobody browsing the docs will find.

    ``bench`` was declared, hinted by its own command and documented in its own guide, but left out
    of the installation page's table of extras — the one place the other five are listed together.
    The reverse direction is checked above (a hint cannot name a nonexistent extra); neither
    direction on its own would have caught a declared-but-unlisted extra.
    """
    page = (ROOT / "docs" / "get-started" / "installation.md").read_text(encoding="utf-8")
    missing = [extra for extra in project()["optional-dependencies"] if f"[{extra}]" not in page]
    assert missing == [], f"extras declared but not offered on the installation page: {missing}"
