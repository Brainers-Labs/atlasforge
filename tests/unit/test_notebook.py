"""The quickstart notebook is a submission artefact, so it is run, not just stored.

A notebook rots faster than anything else in a repository: nothing imports it, nothing type-checks
it, and the first person to notice it is broken is a reader who has already lost confidence. So the
portable cells are executed here, in order, in a temporary directory, and the files they claim to
write are checked for.

Cells marked ``# notebook: colab-only`` are skipped: they install packages or need a GPU, and
running them in this suite would reach the network or install over the test environment. Everything
else has to work on a plain CPython with the package installed — which is the same claim the
notebook's first cell makes to a reader.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "atlasforge-quickstart.ipynb"
SKIP_MARKER = "# notebook: colab-only"
MAGIC_PREFIXES = ("%", "!", "?")


def cells(kind: str | None = None) -> Iterator[dict[str, Any]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if kind is None or cell["cell_type"] == kind:
            yield cell


def source(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def is_colab_only(cell: dict[str, Any]) -> bool:
    return any(line.strip() == SKIP_MARKER for line in source(cell).splitlines())


def as_script(cell: dict[str, Any]) -> str:
    """The cell's code as Python: IPython magic lines removed, so it can be compiled.

    A ``%pip install`` is not Python. Stripping those lines is honest here because every magic in
    this notebook sits in a cell the tests skip anyway, and the check exists to catch the *other*
    lines going stale — a renamed function, a changed keyword argument.
    """
    lines = [
        line
        for line in source(cell).splitlines()
        if not line.lstrip().startswith(MAGIC_PREFIXES) and line.strip() != SKIP_MARKER
    ]
    return "\n".join(lines) + "\n"


def portable_cells() -> list[dict[str, Any]]:
    return [cell for cell in cells("code") if not is_colab_only(cell)]


def test_the_notebook_is_a_valid_v4_notebook() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert len(notebook["cells"]) >= 8


def test_every_cell_has_a_unique_id() -> None:
    ids = [cell["id"] for cell in cells()]
    assert len(ids) == len(set(ids))


def test_it_opens_by_saying_it_needs_no_model() -> None:
    """The first thing a reader sees has to be true, or the rest of the notebook is wasted."""
    opening = source(next(cells("markdown"))).lower()
    assert "no model" in opening
    assert "synthetic" in opening


@pytest.mark.parametrize("cell", list(cells("code")), ids=lambda cell: cell["id"])
def test_every_code_cell_is_valid_python(cell: dict[str, Any]) -> None:
    try:
        ast.parse(as_script(cell))
    except SyntaxError as exc:
        pytest.fail(f"cell {cell['id']!r}: {exc}")


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the portable cells in order and hand back the namespace they left behind."""
    work = tmp_path_factory.mktemp("notebook")
    namespace: dict[str, Any] = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(work)
        for cell in portable_cells():
            exec(compile(as_script(cell), f"<{cell['id']}>", "exec"), namespace)  # noqa: S102
    namespace["_work"] = work
    return namespace


def test_the_notebook_runs_top_to_bottom(ran: dict[str, Any]) -> None:
    """The end state, not the absence of an exception: the files it promises are on disk."""
    work: Path = ran["_work"]
    for produced in (
        work / "atlasforge-demo" / "toy_qa.jsonl",
        work / "atlasforge-demo" / "runs" / "base" / "report.md",
        work / "atlasforge-demo" / "runs" / "base" / "report.json",
        work / "atlasforge-demo" / "runs" / "base" / "report.html",
        work / "comparison" / "comparison.md",
        work / "comparison" / "comparison.json",
        work / "comparison" / "comparison.html",
    ):
        assert produced.is_file(), f"the notebook did not write {produced.name}"


def test_it_compares_the_two_runs_it_built(ran: dict[str, Any]) -> None:
    base = ran["base"]
    tuned = ran["tuned"]
    assert base.name == "base"
    assert tuned.name == "tuned"
    assert base != tuned


def test_the_reported_regression_is_the_one_the_demo_promises(ran: dict[str, Any]) -> None:
    """The notebook says a slice regresses. If the demo data changes, that sentence must change."""
    page = (ran["_work"] / "comparison" / "comparison.md").read_text(encoding="utf-8")
    assert "has_number" in page
    assert "domain" in page


def test_the_display_cell_degrades_without_ipython(ran: dict[str, Any]) -> None:
    """The last portable cell must not require a notebook front end to finish."""
    assert ran.get("html")


def test_the_install_cell_is_the_only_network_touching_one() -> None:
    """One place that installs, and it is skipped here — so a reader can see what it does."""
    network = [cell["id"] for cell in cells("code") if "pip install" in source(cell)]
    assert network == ["install"]


def test_the_colab_only_cells_are_the_install_and_the_real_model() -> None:
    skipped = [cell["id"] for cell in cells("code") if is_colab_only(cell)]
    assert skipped == ["install", "real"]


def test_no_cell_contains_a_real_looking_token() -> None:
    """The token in the last cell is a placeholder, and it must stay one."""
    for cell in cells():
        body = source(cell)
        for token in ("hf_", "sk-"):
            for line in body.splitlines():
                if token in line:
                    assert "..." in line or "hf_****" in line, f"cell {cell['id']}: {line}"


def test_the_last_section_points_at_the_real_models() -> None:
    last = list(cells("markdown"))[-1]
    body = source(last).lower()
    assert "n-atlas" in body
    assert "colab" in body


def test_the_install_line_installs_the_published_distribution() -> None:
    """The package is on PyPI now, so the notebook installs it from there rather than from a clone.

    A ``git+https`` line has to name a branch, which drifts from the released version without
    anything noticing — the failure this test was written for when there was no release to install.
    ``--pre`` stays in the line even though pip does not need it while the alpha is the only
    published version — a stable release is what will make it necessary.
    """
    install = source(next(cell for cell in cells("code") if cell["id"] == "install"))
    assert "--pre brainers-atlasforge" in install
    assert "git+" not in install


def test_the_readme_next_to_it_explains_the_skip_marker() -> None:
    readme = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    assert SKIP_MARKER in readme


def test_every_cell_kind_is_used() -> None:
    kinds: Sequence[str] = [cell["cell_type"] for cell in cells()]
    assert kinds.count("markdown") >= 5
    assert kinds.count("code") >= 4
