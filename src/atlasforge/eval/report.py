"""Render a single scored run as Markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atlasforge.eval import flags
from atlasforge.eval.format import fmt_value

if TYPE_CHECKING:
    from collections.abc import Mapping

    from atlasforge.eval.asr_analysis import AsrAnalysis
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
    lines += ["", *_latency(report), *_flags(report), *_asr(report), *_method()]
    return "\n".join(lines).rstrip() + "\n"


def asr_summary(analysis: AsrAnalysis) -> str:
    """The one-line summary of an alignment, shared by the Markdown and the HTML report.

    The arithmetic lives here once: an insertion is an error but not a *reference* word, so the
    denominator is hits + substitutions + deletions. Two renderers drifting apart on that would
    print two different word error rates for the same run.
    """
    reference_words = analysis.hits + analysis.substitutions + analysis.deletions
    rate = analysis.errors / reference_words if reference_words else 0.0
    return (
        f"{analysis.n} utterance(s) aligned: {analysis.hits} correct words, "
        f"{analysis.substitutions} substituted, {analysis.deletions} dropped, "
        f"{analysis.insertions} inserted (pooled word error rate {rate:.1%})."
    )


def _asr(report: ScoreReport) -> list[str]:
    """What the speech model heard instead. Alignment-based, so it explains the WER."""
    analysis = report.asr
    if analysis is None:
        return []
    lines = ["## ASR error analysis", "", asr_summary(analysis), ""]
    if analysis.tone_only_substitutions:
        lines += [
            (
                f"{analysis.tone_only_substitutions} of the {analysis.substitutions} "
                "substitutions differ only in tone marks, and are marked `tone-only` below. "
                "Those are the ones the tone-insensitive view forgives."
            ),
            "",
        ]
    for title, entries in (
        ("Most substituted", analysis.top_substitutions),
        ("Most dropped", analysis.top_deletions),
        ("Most inserted", analysis.top_insertions),
    ):
        if not entries:
            continue
        lines += [
            f"**{title}**",
            "",
            "| Heard instead | Times | Tone-only | Examples |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {_escape(entry.describe())} | {entry.count} | {'yes' if entry.tone_only else ''} "
            f"| {', '.join(f'`{e}`' for e in entry.examples)} |"
            for entry in entries
        ]
        lines.append("")
    if analysis.by_length:
        lines += [
            "**Word error rate by reference length** (words in the reference):",
            "",
            "| Reference length | Utterances | Pooled WER |",
            "|---|---|---|",
        ]
        lines += [
            f"| {bucket.label} | {bucket.n} | "
            f"{'-' if bucket.wer is None else f'{bucket.wer:.1%}'} |"
            for bucket in analysis.by_length
        ]
        lines.append("")
    return lines


def _escape(text: str) -> str:
    """Keep a `|` in a transcript from breaking the table it is reported in."""
    return text.replace("|", "\\|")


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


def _flags(report: ScoreReport) -> list[str]:
    """Failure-mode flags. Deterministic rules over the text, never a quality score."""
    if report.task == "classification":
        return []
    lines = ["## Failure-mode flags", ""]
    if not report.flags:
        return [
            *lines,
            (
                f"No flags on any of the {report.n_ok} answer(s): none was empty, repeated, "
                "cut off, in the wrong format or script, or missing a number or entity from "
                "the reference."
            ),
            "",
        ]
    lines += [
        (
            f"{report.n_flagged} of {report.n_ok} answer(s) raised at least one flag. Each one "
            "is a rule you can read and re-check, not a quality score and not a hallucination "
            "rate."
        ),
        "",
        "| Flag | Answers | What it means |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{name}` | {count} | {flags.DESCRIPTIONS[name]} |"
        for name, count in report.flags.items()
    )
    lines += [
        "",
        (
            "A flag is a place to look, not a verdict: an answer can be flagged and still be "
            "right. Per-example flags are in `report.json` under `per_example_flags`."
        ),
        "",
    ]
    return lines


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
