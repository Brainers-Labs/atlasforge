"""Shared pytest configuration.

Tests marked ``live`` need a GPU, gated weights and HF_TOKEN. They are skipped
unless ``ATLASFORGE_LIVE=1`` so the default suite runs anywhere with no credentials.
"""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("ATLASFORGE_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test: set ATLASFORGE_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
