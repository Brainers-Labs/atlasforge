"""Base-vs-candidate comparison with honest statistics."""

from atlasforge.compare.compare import (
    ComparisonReport,
    MetricComparison,
    RunInfo,
    compare_runs,
    compare_scores,
)
from atlasforge.compare.report import to_markdown
from atlasforge.compare.slices import MIN_SLICE_N, SliceResult
from atlasforge.compare.stats import BootstrapResult, McNemarResult, mcnemar_exact, paired_bootstrap

__all__ = [
    "MIN_SLICE_N",
    "BootstrapResult",
    "ComparisonReport",
    "McNemarResult",
    "MetricComparison",
    "RunInfo",
    "SliceResult",
    "compare_runs",
    "compare_scores",
    "mcnemar_exact",
    "paired_bootstrap",
    "to_markdown",
]
