"""AtlasForge: run, evaluate, compare and fine-tune the official N-ATLaS models.

Importing this package is deliberately cheap: it pulls in no ML framework, and no
AtlasForge submodule beyond :mod:`atlasforge.api` (which itself imports nothing but the
standard library). Everything else is imported when you first use it.

The two entry points are :func:`atlasforge.evaluate` and :func:`atlasforge.compare_runs`;
``atlasforge.compare`` is the subpackage holding the lower-level pieces.

    >>> import atlasforge
    >>> atlasforge.evaluate("data.jsonl", out_dir="runs/base")   # doctest: +SKIP
"""

from atlasforge.api import (
    Evaluation,
    compare_runs,
    evaluate,
    score_finished_run,
    write_comparison,
    write_reports,
)

__version__ = "0.1.0a1"

__all__ = [
    "Evaluation",
    "__version__",
    "compare_runs",
    "evaluate",
    "score_finished_run",
    "write_comparison",
    "write_reports",
]
