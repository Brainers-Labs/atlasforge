"""Render a comparison as Markdown suitable for pasting into a submission or README."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from atlasforge.compare.compare import POOLED_ONLY

if TYPE_CHECKING:
    from atlasforge.compare.compare import ComparisonReport, MetricComparison, RunInfo
    from atlasforge.compare.slices import SliceResult

_FRACTION_METRICS: Final = frozenset({"exact_match", "accuracy", "macro_f1", "wer", "cer"})


def to_markdown(report: ComparisonReport) -> str:
    """Full report: runs, overall metrics, regressions, slices, and method notes."""
    lines: list[str] = ["# AtlasForge comparison report", ""]
    lines += _header(report)
    lines += _overall(report)
    lines += _regressions(report)
    lines += _slices(report)
    lines += _method(report)
    return "\n".join(lines).rstrip() + "\n"


def _header(report: ComparisonReport) -> list[str]:
    return [
        f"- **Task:** {report.task}",
        f"- **Examples:** {report.n_total}",
        f"- **Dataset fingerprint (sha256):** `{report.dataset_sha256[:16]}`",
        "",
        "| Run | Model | Revision | Backend | Failed | Missing |",
        "|---|---|---|---|---|---|",
        _run_row("Base", report.base),
        _run_row("Candidate", report.candidate),
        "",
    ]


def _run_row(label: str, info: RunInfo) -> str:
    return (
        f"| {label} | {info.model or '?'} | {_short(info.revision)} | {info.backend or '?'} "
        f"| {info.n_failed} | {info.n_missing} |"
    )


def _overall(report: ComparisonReport) -> list[str]:
    lines = [
        "## Overall",
        "",
        (
            "Delta is candidate minus base. The interval covers the mean per-example difference; "
            "a change counts as *improved* or *regressed* only if the whole interval is on one "
            "side of zero."
        ),
        "",
        "| Metric | View | Base | Candidate | Delta | 95% CI | Verdict | W / T / L | McNemar p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    pooled: list[MetricComparison] = []
    for m in report.metrics:
        if m.verdict == POOLED_ONLY:
            pooled.append(m)
            continue
        lines.append(
            f"| {m.name} | {_view(m.view)} | {_value(m.name, m.base_mean)} "
            f"| {_value(m.name, m.candidate_mean)} | {_delta(m.name, m.delta)} "
            f"| {_interval(m)} | {m.verdict} | {m.wins} / {m.ties} / {m.losses} "
            f"| {_p(m)} |"
        )
    lines.append("")

    corpus = [
        m for m in report.metrics if m.base_corpus is not None or m.candidate_corpus is not None
    ]
    if corpus:
        lines += [
            "**Pooled figures** (computed over the whole corpus, no interval):",
            "",
            "| Metric | View | Base | Candidate | Delta |",
            "|---|---|---|---|---|",
        ]
        for m in corpus:
            delta = (
                None
                if m.base_corpus is None or m.candidate_corpus is None
                else m.candidate_corpus - m.base_corpus
            )
            lines.append(
                f"| {m.name} | {_view(m.view)} | {_value(m.name, m.base_corpus)} "
                f"| {_value(m.name, m.candidate_corpus)} | {_delta(m.name, delta)} |"
            )
        lines.append("")
    if pooled:
        names = ", ".join(sorted({m.name for m in pooled}))
        lines += [f"*{names}: pooled metric, no per-example values, so no interval or test.*", ""]
    return lines


def _regressions(report: ComparisonReport) -> list[str]:
    metrics = report.regressed_metrics
    slices = report.regressed_slices
    lines = ["## Regressions", ""]
    if not metrics and not slices:
        return [*lines, "No statistically clear regressions found.", ""]
    lines.extend(
        f"- **{m.name}** ({_view(m.view)}) got worse overall: {_delta(m.name, m.delta)}."
        for m in metrics
    )
    lines.extend(
        f"- **{s.field} = {s.value}** (n={s.n}): {_slice_delta(report, s)}, "
        f"CI [{_slice_bound(report, s.low)}, {_slice_bound(report, s.high)}]."
        for s in slices
    )
    return [*lines, ""]


def _slices(report: ComparisonReport) -> list[str]:
    if not report.slices:
        return []
    lines = [f"## Slices ({report.primary})", ""]
    for name in dict.fromkeys(s.field for s in report.slices):
        lines += [
            f"**{name}**",
            "",
            "| Value | n | Base | Candidate | Delta | 95% CI | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in (s for s in report.slices if s.field == name):
            metric = (report.primary or "").split("@")[0]
            if s.status == "insufficient data":
                lines.append(
                    f"| {s.value} | {s.n} | {_value(metric, s.base_mean)} "
                    f"| {_value(metric, s.candidate_mean)} | - | - | insufficient data (n < "
                    f"{report.settings.get('min_slice_n')}) |"
                )
            else:
                lines.append(
                    f"| {s.value} | {s.n} | {_value(metric, s.base_mean)} "
                    f"| {_value(metric, s.candidate_mean)} | {_delta(metric, s.delta)} "
                    f"| [{_slice_bound(report, s.low)}, {_slice_bound(report, s.high)}] | {s.status} |"
                )
        lines.append("")
    return lines


def _method(report: ComparisonReport) -> list[str]:
    s = report.settings
    return [
        "## Method",
        "",
        (
            f"- Paired bootstrap of per-example differences: {s.get('n_boot')} resamples, "
            f"seed {s.get('seed')}, {int(float(s.get('confidence', 0.95)) * 100)}% percentile "
            "interval."
        ),
        "- Binary metrics also get an exact McNemar test on the discordant pairs.",
        "- Failed or missing predictions are scored as wrong (an empty answer), for both runs.",
        (
            "- Every text metric is shown under both views: *tone-aware* keeps tone marks, "
            "*tone-insensitive* strips them. Underdots and Hausa hooked letters are always kept."
        ),
        (
            f"- Slices with fewer than {s.get('min_slice_n')} examples are reported as "
            "insufficient data. Slice intervals are not adjusted for multiple comparisons: treat "
            "a single flagged slice as a lead to investigate, not a proven effect."
        ),
        "- This shows the candidate scored differently on this dataset. It does not show why.",
        "",
    ]


def _value(name: str, x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if name in _FRACTION_METRICS else f"{x:.1f}"


def _delta(name: str, x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x * 100:+.1f} pts" if name in _FRACTION_METRICS else f"{x:+.1f}"


def _interval(m: MetricComparison) -> str:
    if m.low is None or m.high is None:
        return "-"
    if m.name in _FRACTION_METRICS:
        return f"[{m.low * 100:+.1f}, {m.high * 100:+.1f}]"
    return f"[{m.low:+.1f}, {m.high:+.1f}]"


def _p(m: MetricComparison) -> str:
    return "-" if m.mcnemar is None else f"{m.mcnemar.p_value:.3g}"


def _view(view: str) -> str:
    return view.replace("_", "-")


def _short(revision: str | None) -> str:
    return "-" if not revision else f"`{revision[:10]}`"


def _metric_name(report: ComparisonReport) -> str:
    return (report.primary or "").split("@")[0]


def _slice_delta(report: ComparisonReport, s: SliceResult) -> str:
    return _delta(_metric_name(report), s.delta)


def _slice_bound(report: ComparisonReport, x: float | None) -> str:
    if x is None:
        return "-"
    name = _metric_name(report)
    return f"{x * 100:+.1f}" if name in _FRACTION_METRICS else f"{x:+.1f}"
