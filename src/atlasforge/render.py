"""Terminal rendering (rich). Pure presentation: no logic that belongs in the library.

Every dynamic string goes through ``Text`` so square brackets in data (``atlasforge[local]``,
model names, prompts) are never mistaken for rich markup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from rich.table import Table
from rich.text import Text

from atlasforge.compare.compare import POOLED_ONLY
from atlasforge.eval.format import fmt_bound, fmt_delta, fmt_value

if TYPE_CHECKING:
    from rich.console import Console

    from atlasforge.compare.compare import ComparisonReport
    from atlasforge.eval.score import ScoreReport
    from atlasforge.eval.validate import Issue, ValidationReport

_VERDICT_STYLE: Final = {"improved": "green", "regressed": "bold red"}
_VIEW_LABEL: Final = {"tone_aware": "aware", "tone_insensitive": "insens."}
_SEVERITY: Final = {
    "error": ("ERROR", "bold red"),
    "warning": ("WARN", "yellow"),
    "info": ("note", "dim"),
}


def print_score(console: Console, report: ScoreReport) -> None:
    """Metrics table for one run."""
    table = Table(header_style="bold")
    for column in ("Metric", "Mean", "Pooled", "n"):
        table.add_column(column, justify="left" if column == "Metric" else "right", overflow="fold")
    for s in report.metrics:
        table.add_row(
            Text(f"{s.name} ({_VIEW_LABEL.get(s.view, s.view)})"),
            Text(fmt_value(s.name, s.mean)),
            Text(fmt_value(s.name, s.corpus)),
            Text(str(s.n)),
        )
    console.print(table)
    console.print(
        Text(
            f"{report.n_total} examples: {report.n_ok} ok, {report.n_failed} failed, "
            f"{report.n_missing} missing (failures count as wrong)",
            style="yellow" if report.n_failed or report.n_missing else "dim",
        )
    )


def print_comparison(console: Console, report: ComparisonReport) -> None:
    """Overall metrics, then any regressions."""
    # Compact on purpose: it must fit an 80-column terminal without hiding the verdict.
    # Cells fold (wrap) instead of truncating; W/T/L and full detail live in the report file.
    table = Table(header_style="bold")
    for column in ("Metric", "Base", "Cand.", "Delta", "95% CI", "Verdict"):
        table.add_column(column, overflow="fold")
    for m in report.metrics:
        if m.verdict == POOLED_ONLY:
            continue
        interval = (
            "-"
            if m.low is None or m.high is None
            else f"[{fmt_bound(m.name, m.low)}, {fmt_bound(m.name, m.high)}]"
        )
        table.add_row(
            Text(f"{m.name} ({_VIEW_LABEL.get(m.view, m.view)})"),
            Text(fmt_value(m.name, m.base_mean)),
            Text(fmt_value(m.name, m.candidate_mean)),
            Text(fmt_delta(m.name, m.delta)),
            Text(interval),
            Text(m.verdict, style=_VERDICT_STYLE.get(m.verdict, "")),
        )
    console.print(table)

    regressions = [
        *(
            f"{m.name} ({m.view.replace('_', '-')}) got worse overall"
            for m in report.regressed_metrics
        ),
        *(
            f"{s.field} = {s.value} (n={s.n}): {fmt_delta((report.primary or '').split('@')[0], s.delta)}"
            for s in report.regressed_slices
        ),
    ]
    if regressions:
        console.print(Text("Regressions", style="bold red"))
        for line in regressions:
            console.print(Text(f"  - {line}", style="red"))
    else:
        console.print(Text("No statistically clear regressions.", style="green"))

    thin = [s for s in report.slices if s.status == "insufficient data"]
    if thin:
        console.print(
            Text(
                f"{len(thin)} slice(s) had too few examples to judge (see the report).",
                style="dim",
            )
        )


def print_validation(console: Console, report: ValidationReport) -> None:
    """Findings grouped by severity, then a short summary."""
    for severity in ("error", "warning", "info"):
        for issue in (i for i in report.issues if i.severity == severity):
            _print_issue(console, issue)
    label, style = ("OK", "green") if report.ok else ("FAILED", "bold red")
    console.print(
        Text(
            f"{label}: {report.n_examples} valid example(s) from {report.n_lines} line(s), "
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s).",
            style=style,
        )
    )


def _print_issue(console: Console, issue: Issue) -> None:
    label, style = _SEVERITY[issue.severity]
    where = f"line {issue.line}: " if issue.line is not None and issue.code == "line-error" else ""
    console.print(Text(f"{label:<5} {where}{issue.message}", style=style), soft_wrap=True)
    if issue.hint:
        console.print(Text(f"      {issue.hint}", style="dim"), soft_wrap=True)
