"""Every NCAIR1 repository the project names is one of the five that exist.

The submission rests on a claim that is easy to break by accident: *the core path uses only the
official NCAIR1 models*. A single stale identifier in a docstring, a guide or a config example
undermines it — and, worse, a user who copies that example gets a 404 from Hugging Face rather than a
model. This is the same failure mode as the old ``pip install "atlasforge[local]"`` hint, which is
why it is checked the same way: read the identifiers out of the tree and compare them with the one
place that defines them.

Not a network check. It cannot know whether ``NCAIR1/Hausa-ASR`` is still reachable or still the
right checkpoint — only that we say the same thing everywhere, and that nothing invented appears.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from atlasforge.asr.models import ASR_MODELS
from atlasforge.backends.factory import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[2]
#: Dots are allowed inside a name but not at the end, or ``NCAIR1/Yoruba-ASR.`` would capture the
#: full stop of the sentence and report a repository that is one character too long.
NCAIR1 = re.compile(r"\bNCAIR1/[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*")
#: A LoRA adapter is part of the model identity: ``NCAIR1/N-ATLaS+my-adapter``.
ADAPTER = re.compile(r"\+[A-Za-z0-9._-]+$")

SELF = Path(__file__).resolve()
TEXT_FILES = [
    path
    for path in (
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "tests").rglob("*.py"),
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CHANGELOG.md",
        ROOT / "examples" / "README.md",
    )
    # This file is the one place that names a wrong identifier on purpose, to show the guard works.
    if path.resolve() != SELF
]

#: The five repositories, plus the base of an adapter id.
KNOWN = frozenset({DEFAULT_MODEL, *ASR_MODELS.values()})


def mentions() -> list[tuple[Path, int, str]]:
    """Every ``NCAIR1/...`` identifier in the tree, with where it was found."""
    found: list[tuple[Path, int, str]] = []
    for path in TEXT_FILES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            found.extend((path, number, match.group(0)) for match in NCAIR1.finditer(line))
    return found


def test_the_registry_is_the_five_official_repositories() -> None:
    """The definition itself: one LLM and four monolingual speech models, all under ``NCAIR1``."""
    assert len(KNOWN) == 5
    assert DEFAULT_MODEL == "NCAIR1/N-ATLaS"
    assert set(ASR_MODELS) == {"ha", "yo", "ig", "en"}
    assert all(repo.startswith("NCAIR1/") for repo in KNOWN)


def test_there_are_identifiers_to_check() -> None:
    """A guard that matches nothing passes silently, which is how a broken guard hides."""
    assert len(mentions()) > 20


@pytest.mark.parametrize(
    ("path", "number", "identifier"),
    mentions(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_named_repository_is_one_of_the_five(
    path: Path, number: int, identifier: str
) -> None:
    base = ADAPTER.sub("", identifier)
    assert base in KNOWN, (
        f"{path.relative_to(ROOT)}:{number} names {identifier}, which is not a real repository"
    )


def test_it_catches_a_plausible_but_wrong_identifier() -> None:
    """``NCAIR1/N-ATLaS-8B`` looks right and is not: the rename that introduced this test."""
    assert ADAPTER.sub("", "NCAIR1/N-ATLaS-8B") not in KNOWN
    assert ADAPTER.sub("", "NCAIR1/N-ATLaS") in KNOWN


def test_an_adapter_identifier_is_accepted_as_its_base_model() -> None:
    assert ADAPTER.sub("", "NCAIR1/N-ATLaS+my-adapter") == "NCAIR1/N-ATLaS"


def test_the_access_page_lists_exactly_the_five() -> None:
    """The one page a user is sent to in order to accept the licences has to be complete."""
    page = (ROOT / "docs" / "get-started" / "access-and-licences.md").read_text(encoding="utf-8")
    listed = {match.group(0) for match in NCAIR1.finditer(page)}
    assert listed == KNOWN


def test_the_transcription_guide_maps_every_language_to_its_model() -> None:
    page = (ROOT / "docs" / "guides" / "transcribe-speech.md").read_text(encoding="utf-8")
    for lang, repo in ASR_MODELS.items():
        assert f"`{lang}`" in page, f"{lang} is missing from the transcription guide"
        assert repo in page, f"{repo} is missing from the transcription guide"
