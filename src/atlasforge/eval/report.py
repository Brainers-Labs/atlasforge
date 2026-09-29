"""Render a single scored run as Markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atlasforge.eval.format import fmt_value

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atlasforge.eval.score import ScoreReport


def to_markdown(report: ScoreReport, manifest: Mapping[str, Any] | None = None) -> str:
    """Report for one run: identity, metrics under both tone views, latency and method notes."""
    m = manifest or {}
    lines = [
        "# AtlasForge evaluation report",
        "",
        f"- **Task:** {report.task}",
        (
            f"- **Examples:** {report.n_total} "
            f"({report.n_ok} ok, {report.n_failed} failed, {report.n_missing} missing)"
        ),
        f"- **Dataset fingerprint (sha256):** `{report.dataset_sha256[:16]}`",
        f"- **Model:** {m.get('model') or '?'}"
        + (f" (revision `{str(m['revision'])[:10]}`)" if m.get("revision") else ""),
        f"- **Backend:** {m.get('backend') or '?'}",
        "",
        "## Metrics",
        "",
        "| Metric | View | Mean | Pooled | n |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {s.name} | {s.view.replace('_', '-')} | {fmt_value(s.name, s.mean)} "
        f"| {fmt_value(s.name, s.corpus)} | {s.n} |"
        for s in report.metrics
    )
    lines += ["", *_latency(report), *_method()]
    return "\n".join(lines).rstrip() + "\n"


def _latency(report: ScoreReport) -> list[str]:
    stats = report.latency_ms
    if stats["mean"] is None:
        return []

    def ms(key: str) -> str:
        value = stats[key]
        return "-" if value is None else f"{value:.0f} ms"

    return [
        "## Latency",
        "",
        (
            f"mean {ms('mean')} | p50 {ms('p50')} | p95 {ms('p95')} | max {ms('max')} "
            "(successful calls only)"
        ),
        "",
    ]


def _method() -> list[str]:
    return [
        "## Method",
        "",
        (
            "- *Mean* averages per-example scores; *pooled* is computed over the whole corpus "
            "(WER, CER, chrF, macro-F1)."
        ),
        (
            "- Every text metric is shown under both views: *tone-aware* keeps tone marks, "
            "*tone-insensitive* strips them. Underdots and Hausa hooked letters are always kept."
        ),
        (
            "- Failed or missing predictions are scored as an empty answer, so they count against "
            "the model."
        ),
        "- chrF is on a 0-100 scale. Other metrics are percentages; for WER and CER lower is better.",
        "",
    ]
