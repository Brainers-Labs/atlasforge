"""The reports in ``examples/`` are a submission artefact, so they are checked for staleness.

Committed example output is a liability: it silently becomes a description of a past version of the
code. These regenerate it from the current code and fail if a single byte differs, so a change that
alters a report cannot be merged without the example being regenerated in the same commit.

Nothing here is mocked. The demo data is the real demo data, and the reports are produced by the
real CLI through the same entry point a user would type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from atlasforge.cli import app
from atlasforge.demo import DATASET_NAME, build_demo

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "reports"
runner = CliRunner()

#: The committed files, and how each one is produced. ``report`` writes beside the run it names;
#: ``compare`` writes a ``comparison/`` directory in the working directory.
REPORT_FILES = ("report.md", "report.json", "report.html")
COMPARISON_FILES = ("comparison.md", "comparison.json", "comparison.html")
COMMITTED = (*REPORT_FILES, *COMPARISON_FILES)


@pytest.fixture(scope="module")
def produced(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Regenerate every example from the current code, in a throwaway directory."""
    work = tmp_path_factory.mktemp("examples")
    demo = build_demo(work / "atlasforge-demo")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(demo)
        result = runner.invoke(app, ["report", "runs/base", "--dataset", DATASET_NAME])
        assert result.exit_code == 0, result.output
        result = runner.invoke(
            app,
            [
                "compare",
                DATASET_NAME,
                "--base",
                "runs/base",
                "--candidate",
                "runs/tuned",
                "--slice",
                "domain",
            ],
        )
        assert result.exit_code == 0, result.output
        yield demo


def regenerated(demo: Path, name: str) -> Path:
    """Where the freshly produced file of that name landed."""
    if name in REPORT_FILES:
        return demo / "runs" / "base" / name
    return demo / "comparison" / name


def test_the_example_directory_holds_exactly_what_it_claims() -> None:
    assert sorted(p.name for p in EXAMPLES.iterdir()) == sorted(COMMITTED)


@pytest.mark.parametrize("name", COMMITTED)
def test_the_committed_report_is_the_one_this_code_produces(name: str, produced: Path) -> None:
    """Byte for byte, so a report cannot quietly describe an older version of the tool.

    This works only because a report is deterministic: no timestamp, no absolute path, no run id
    that varies. That is worth keeping — if a change makes an example irreproducible, this test is
    how you find out, and the fix is usually to stop writing the varying value into the report.
    """
    committed = (EXAMPLES / name).read_text(encoding="utf-8")
    fresh = regenerated(produced, name).read_text(encoding="utf-8")
    assert committed == fresh, (
        f"examples/reports/{name} is out of date. Regenerate it: see examples/README.md."
    )


def test_the_html_pages_stay_self_contained(produced: Path) -> None:
    """The property the docs promise: one file, no network, safe to email."""
    for name in ("report.html", "comparison.html"):
        html = regenerated(produced, name).read_text(encoding="utf-8")
        for forbidden in ("<script", "http://", "https://cdn", "<link "):
            assert forbidden not in html, f"{name} is no longer self-contained ({forbidden})"


def test_the_comparison_shows_a_regression_not_only_an_improvement() -> None:
    """The example is chosen for this: a page that only ever improves teaches the wrong lesson."""
    assert "has_number" in (EXAMPLES / "comparison.md").read_text(encoding="utf-8")


def test_the_readme_says_the_reports_are_synthetic() -> None:
    """A reader who finds the files without the context must not think a model produced them."""
    readme = (ROOT / "examples" / "README.md").read_text(encoding="utf-8").lower()
    assert "synthetic" in readme
    assert "not produced by n-atlas" in readme
