"""Self-contained HTML reports: one file, no JavaScript, no external resources.

Markdown and JSON stay the formats AtlasForge writes for machines and for pasting into a
submission. This module adds the same numbers as a page a person can read in a browser — the
metrics under both tone views, the failure-mode flags, and the speech error analysis — with the
charts drawn inline.

The file stands on its own: its styles, its charts and every number in it are part of the one
document, so it opens from an email attachment, a USB stick or an air-gapped machine, and nothing
in it can reach the network. There is no JavaScript at all: this is a *static view* of a report
that was already computed, not a dashboard that recomputes anything.

Charts are built from exactly the numbers the tables show, so the picture and the table cannot
disagree. A reader who wants the figure to three digits reads the table; the bar is there to make
the size of a difference visible at a glance.

Everything that comes from a report — a metric name, a slice value, a transcript word — is escaped
on the way in, so a model that answers with ``<script>`` can only ever produce text on the page.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Final

from atlasforge import __version__
from atlasforge.eval import flags as flag_rules
from atlasforge.eval.format import fmt_bound, fmt_delta, fmt_value, is_fraction
from atlasforge.eval.report import asr_summary

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Any

    from atlasforge.compare.compare import ComparisonReport, MetricComparison
    from atlasforge.eval.score import ScoreReport

# --- chart geometry. One coordinate system for every chart, in viewBox units. -------------

_CHART_W: Final = 720
_LABEL_W: Final = 210
_VALUE_W: Final = 84
_BAR_X: Final = _LABEL_W
_BAR_W: Final = _CHART_W - _LABEL_W - _VALUE_W
_VALUE_X: Final = _BAR_X + _BAR_W + 6
_ROW_H: Final = 26
_BAR_H: Final = 12
_GROUP_H: Final = 40
_GROUP_BAR_H: Final = 10
_GROUP_GAP: Final = 8
_AXIS_H: Final = 22
_PERCENT_TICKS: Final = ((0.0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%"))

_CSS: Final = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1b1f24; --muted: #5b6471; --line: #d8dee6; --track: #e9eef4;
  --bar: #2563eb; --bar-2: #93b4f5; --good: #1a7f37; --bad: #b42318; --flat: #8b95a3;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e7ecf2; --muted: #9aa4b2; --line: #2a3038; --track: #232a33;
    --bar: #60a5fa; --bar-2: #38507a; --good: #4ade80; --bad: #f87171; --flat: #7b8697;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 920px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 16px; }
h2 { font-size: 19px; margin: 32px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }
p { margin: 8px 0; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; margin: 0 0 8px; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
ul { margin: 8px 0; padding-left: 22px; }
li { margin: 2px 0; }
.chart { display: block; width: 100%; max-width: 720px; height: auto; margin: 8px 0 2px; }
.chart text { font: 11px system-ui, sans-serif; fill: var(--fg); }
.chart .tick { fill: var(--muted); font-size: 10px; }
.chart .track { fill: var(--track); }
.chart .bar { fill: var(--bar); }
.chart .bar-2 { fill: var(--bar-2); }
.chart .improved { fill: var(--good); }
.chart .regressed { fill: var(--bad); }
.chart .flat { fill: var(--flat); }
.chart .axis { stroke: var(--line); stroke-width: 1; }
.chart .whisker { stroke: var(--muted); stroke-width: 1; }
.legend { color: var(--muted); font-size: 13px; }
.swatch {
  display: inline-block; width: 10px; height: 10px; margin-right: 4px;
  border-radius: 2px; vertical-align: baseline;
}
.note { color: var(--muted); font-size: 13px; }
footer { color: var(--muted); font-size: 13px; margin-top: 40px; border-top: 1px solid var(--line); padding-top: 8px; }
"""


def report_html(report: ScoreReport, manifest: Mapping[str, Any] | None = None) -> str:
    """One scored run as a self-contained page: identity, metrics, flags, latency, ASR analysis."""
    m = manifest or {}
    facts = [
        ("Task", report.task),
        (
            "Examples",
            (
                f"{report.n_total} ({report.n_ok} ok, {report.n_failed} failed, "
                f"{report.n_missing} missing)"
            ),
        ),
        ("Dataset fingerprint (sha256)", report.dataset_sha256[:16]),
        ("Model", m.get("model") or "?"),
    ]
    if m.get("revision"):
        facts.append(("Revision", str(m["revision"])[:10]))
    facts.append(("Backend", m.get("backend") or "?"))
    body = [
        _facts(facts),
        _metrics_section(report),
        *_latency_section(report),
        *_flags_section(report),
        *_asr_section(report),
        _method_section(False),
    ]
    return _page("AtlasForge evaluation report", body)


def comparison_html(report: ComparisonReport) -> str:
    """A comparison as a self-contained page: overall deltas, regressions, flags and slices."""
    facts = [
        ("Task", report.task),
        ("Examples", report.n_total),
        ("Dataset fingerprint (sha256)", report.dataset_sha256[:16]),
        ("Primary metric", report.primary or "?"),
    ]
    body = [
        _facts(facts),
        _table(
            ("Run", "Model", "Revision", "Backend", "Failed", "Missing", "Flagged"),
            [
                (
                    "Base",
                    report.base.model or "?",
                    _short(report.base.revision),
                    report.base.backend or "?",
                    report.base.n_failed,
                    report.base.n_missing,
                    report.base.n_flagged,
                ),
                (
                    "Candidate",
                    report.candidate.model or "?",
                    _short(report.candidate.revision),
                    report.candidate.backend or "?",
                    report.candidate.n_failed,
                    report.candidate.n_missing,
                    report.candidate.n_flagged,
                ),
            ],
        ),
        _overall_section(report),
        _regressions_section(report),
        *_flags_section_comparison(report),
        *_slices_section(report),
        _method_section(True, report),
    ]
    return _page("AtlasForge comparison report", body)


# --- sections ------------------------------------------------------------------------------


def _metrics_section(report: ScoreReport) -> str:
    by_name: dict[str, dict[str, float | None]] = {}
    for summary in report.metrics:
        by_name.setdefault(summary.name, {})[summary.view] = summary.mean
    rows = [
        (
            name,
            _series(name, views.get("tone_aware")),
            _series(name, views.get("tone_insensitive")),
        )
        for name, views in by_name.items()
    ]
    return "\n".join(
        [
            "<h2>Metrics</h2>",
            _legend(("tone-aware", "bar"), ("tone-insensitive", "bar-2")),
            _grouped(rows, aria="Mean score per metric, under both tone views"),
            _note(
                "Means. Fraction metrics are shown as percentages, chrF on its own 0-100 scale; "
                "WER and CER are the only ones where lower is better. Pooled figures are in the "
                "table."
            ),
            _table(
                ("Metric", "View", "Mean", "Pooled", "n"),
                [
                    (
                        s.name,
                        s.view.replace("_", "-"),
                        fmt_value(s.name, s.mean),
                        fmt_value(s.name, s.corpus),
                        s.n,
                    )
                    for s in report.metrics
                ],
            ),
        ]
    )


def _latency_section(report: ScoreReport) -> list[str]:
    stats = report.latency_ms
    if stats["mean"] is None:
        return []

    def ms(key: str) -> str:
        value = stats[key]
        return "-" if value is None else f"{value:.0f} ms"

    return [
        "<h2>Latency</h2>",
        _facts(
            [
                ("Mean", ms("mean")),
                ("p50", ms("p50")),
                ("p95", ms("p95")),
                ("Max", ms("max")),
            ]
        ),
        _note("Successful calls only: a failure has no meaningful latency to average."),
    ]


def _flags_section(report: ScoreReport) -> list[str]:
    """Failure-mode flags for one run. Deterministic rules, never a quality score."""
    if report.task == "classification":
        return []
    heading = "<h2>Failure-mode flags</h2>"
    if not report.flags:
        return [
            heading,
            (
                f"<p>No flags on any of the {report.n_ok} answer(s): none was empty, repeated, "
                "cut off, in the wrong format or script, or missing a number or entity from the "
                "reference.</p>"
            ),
        ]
    counts = list(report.flags.items())
    return [
        heading,
        (
            f"<p>{report.n_flagged} of {report.n_ok} answer(s) raised at least one flag. Each one "
            "is a rule you can read and re-check, not a quality score and not a hallucination "
            "rate.</p>"
        ),
        _bars(
            [(name, float(count), str(count)) for name, count in counts],
            ticks=_count_ticks(float(max(count for _, count in counts))),
            aria="Answers raising each failure-mode flag",
        ),
        _table(
            ("Flag", "Answers", "What it means"),
            [(name, count, flag_rules.DESCRIPTIONS[name]) for name, count in counts],
        ),
        _note(
            "A flag is a place to look, not a verdict: an answer can be flagged and still be "
            "right. Per-example flags are in report.json under per_example_flags."
        ),
    ]


def _asr_section(report: ScoreReport) -> list[str]:
    analysis = report.asr
    if analysis is None:
        return []
    lines = ["<h2>ASR error analysis</h2>", f"<p>{asr_summary(analysis)}</p>"]
    if analysis.tone_only_substitutions:
        lines.append(
            f"<p>{analysis.tone_only_substitutions} of the {analysis.substitutions} substitutions "
            "differ only in tone marks, and are marked <em>tone-only</em> below. Those are the "
            "ones the tone-insensitive view forgives.</p>"
        )
    words = [
        ("correct words", analysis.hits),
        ("substituted", analysis.substitutions),
        ("dropped", analysis.deletions),
        ("inserted", analysis.insertions),
    ]
    lines += [
        _bars(
            [(label, float(count), str(count)) for label, count in words],
            ticks=_count_ticks(float(max(count for _, count in words))),
            aria="Aligned words: correct, substituted, dropped and inserted",
        ),
        _note(
            "Words over the utterances that had a reference. A dropped word is a reference word "
            "the model did not say; an inserted word is one it said that was not there."
        ),
    ]
    for title, entries in (
        ("Most substituted", analysis.top_substitutions),
        ("Most dropped", analysis.top_deletions),
        ("Most inserted", analysis.top_insertions),
    ):
        if not entries:
            continue
        lines += [
            f"<h3>{title}</h3>",
            _table(
                ("Heard instead", "Times", "Tone-only", "Examples"),
                [
                    (
                        entry.describe(),
                        entry.count,
                        "yes" if entry.tone_only else "",
                        ", ".join(entry.examples),
                    )
                    for entry in entries
                ],
            ),
        ]
    if analysis.by_length:
        lines += [
            "<h3>Word error rate by reference length</h3>",
            _bars(
                [
                    (
                        f"{bucket.label} words (n={bucket.n})",
                        None if bucket.wer is None else bucket.wer * 100,
                        "-" if bucket.wer is None else f"{bucket.wer:.1%}",
                    )
                    for bucket in analysis.by_length
                ],
                maximum=100.0,
                ticks=_PERCENT_TICKS,
                aria="Pooled word error rate by reference length in words",
            ),
            _note(
                "Length is words in the reference, not seconds: measuring duration would mean "
                "decoding audio during every report."
            ),
        ]
    return lines


def _overall_section(report: ComparisonReport) -> str:
    tested = [m for m in report.metrics if m.delta is not None]
    lines = [
        "<h2>Overall</h2>",
        (
            "<p>Delta is candidate minus base. The interval covers the mean per-example "
            "difference; a change counts as <em>improved</em> or <em>regressed</em> only if the "
            "whole interval is on one side of zero.</p>"
        ),
    ]
    if tested:
        lines += [
            _legend(
                ("improved", "improved"), ("regressed", "regressed"), ("no clear change", "flat")
            ),
            _diverging(
                [
                    (
                        f"{m.name} · {m.view.replace('_', '-')}",
                        _scaled(m.name, m.delta),
                        _scaled(m.name, m.low),
                        _scaled(m.name, m.high),
                        m.verdict,
                        fmt_delta(m.name, m.delta),
                    )
                    for m in tested
                ],
                aria="Change per metric, candidate minus base, with 95% intervals",
            ),
            _note(
                "Bars are the change in points (percent for fraction metrics, chrF points for "
                "chrF), with the 95% interval drawn as a whisker."
            ),
        ]
    lines.append(
        _table(
            (
                "Metric",
                "View",
                "Base",
                "Candidate",
                "Delta",
                "95% CI",
                "Verdict",
                "W / T / L",
                "McNemar p",
            ),
            [
                (
                    m.name,
                    m.view.replace("_", "-"),
                    fmt_value(m.name, m.base_mean),
                    fmt_value(m.name, m.candidate_mean),
                    fmt_delta(m.name, m.delta),
                    _interval(m),
                    m.verdict,
                    f"{m.wins} / {m.ties} / {m.losses}",
                    "-" if m.mcnemar is None else f"{m.mcnemar.p_value:.3g}",
                )
                for m in report.metrics
            ],
        )
    )
    corpus = [
        m for m in report.metrics if m.base_corpus is not None or m.candidate_corpus is not None
    ]
    if corpus:
        lines += [
            "<h3>Pooled figures</h3>",
            _table(
                ("Metric", "View", "Base", "Candidate", "Delta"),
                [
                    (
                        m.name,
                        m.view.replace("_", "-"),
                        fmt_value(m.name, m.base_corpus),
                        fmt_value(m.name, m.candidate_corpus),
                        fmt_delta(m.name, _difference(m)),
                    )
                    for m in corpus
                ],
            ),
            _note(
                "Computed over the whole corpus, so there is no per-example difference to put an "
                "interval on."
            ),
        ]
    pooled = sorted({m.name for m in report.metrics if m.delta is None})
    if pooled:
        lines.append(
            _note(f"{', '.join(pooled)}: pooled metric, no per-example values, so no interval.")
        )
    return "\n".join(lines)


def _regressions_section(report: ComparisonReport) -> str:
    metrics = report.regressed_metrics
    slices = report.regressed_slices
    lines = ["<h2>Regressions</h2>"]
    if not metrics and not slices:
        lines.append("<p>No statistically clear regressions found.</p>")
        return "\n".join(lines)
    items = [
        f"<strong>{_h(m.name)}</strong> ({_h(m.view.replace('_', '-'))}) got worse overall: "
        f"{_h(fmt_delta(m.name, m.delta))}."
        for m in metrics
    ]
    metric = _primary_name(report)
    items += [
        f"<strong>{_h(s.field)} = {_h(s.value)}</strong> (n={s.n}): "
        f"{_h(fmt_delta(metric, s.delta))}, CI [{_h(fmt_bound(metric, s.low))}, "
        f"{_h(fmt_bound(metric, s.high))}]."
        for s in slices
    ]
    return "\n".join([*lines, "<ul>", *(f"<li>{item}</li>" for item in items), "</ul>"])


def _flags_section_comparison(report: ComparisonReport) -> list[str]:
    if not report.flags:
        return []
    top = float(max(max(f.base, f.candidate) for f in report.flags))
    return [
        "<h2>Failure-mode flags</h2>",
        (
            "<p>Deterministic rules over the text, counted per answer — the same rules the "
            "single-run report lists. Places to look, not a quality score, and not a "
            "hallucination rate.</p>"
        ),
        _legend(("Base", "bar"), ("Candidate", "bar-2")),
        _grouped(
            [
                (
                    f.name,
                    (float(f.base), str(f.base)),
                    (float(f.candidate), str(f.candidate)),
                )
                for f in report.flags
            ],
            maximum=top,
            ticks=_count_ticks(top),
            aria="Answers raising each failure-mode flag, base against candidate",
        ),
        _table(
            ("Flag", "Base", "Candidate", "Delta", "What it means"),
            [
                (f.name, f.base, f.candidate, f"{f.delta:+d}", flag_rules.DESCRIPTIONS[f.name])
                for f in report.flags
            ],
        ),
    ]


def _slices_section(report: ComparisonReport) -> list[str]:
    if not report.slices:
        return []
    metric = _primary_name(report)
    lines = [f"<h2>Slices ({_h(report.primary or '?')})</h2>"]
    for name in dict.fromkeys(s.field for s in report.slices):
        group = [s for s in report.slices if s.field == name]
        lines.append(f"<h3>{_h(name)}</h3>")
        judged = [s for s in group if s.delta is not None]
        if judged:
            lines += [
                _diverging(
                    [
                        (
                            f"{s.value} (n={s.n})",
                            _scaled(metric, s.delta),
                            _scaled(metric, s.low),
                            _scaled(metric, s.high),
                            s.status,
                            fmt_delta(metric, s.delta),
                        )
                        for s in judged
                    ],
                    aria=f"Change per value of {name}, candidate minus base",
                )
            ]
        lines.append(
            _table(
                ("Value", "n", "Base", "Candidate", "Delta", "95% CI", "Status"),
                [
                    (
                        s.value,
                        s.n,
                        fmt_value(metric, s.base_mean),
                        fmt_value(metric, s.candidate_mean),
                        fmt_delta(metric, s.delta),
                        "-"
                        if s.low is None or s.high is None
                        else f"[{fmt_bound(metric, s.low)}, {fmt_bound(metric, s.high)}]",
                        s.status,
                    )
                    for s in group
                ],
            )
        )
    lines.append(
        _note(
            "Slices are not adjusted for multiple comparisons: treat a single flagged slice as a "
            "lead to investigate, not a proven effect."
        )
    )
    return lines


def _method_section(compared: bool, report: ComparisonReport | None = None) -> str:
    bullets = [
        (
            "<em>Mean</em> averages per-example scores; <em>pooled</em> is computed over the "
            "whole corpus (WER, CER, chrF, macro-F1)."
        ),
        (
            "Every text metric is shown under both views: <em>tone-aware</em> keeps tone marks, "
            "<em>tone-insensitive</em> strips them. Underdots and Hausa hooked letters are always "
            "kept."
        ),
        (
            "Failed or missing predictions are scored as an empty answer, so they count against "
            "the model."
        ),
        "chrF is on a 0-100 scale. Other metrics are percentages; for WER and CER lower is better.",
    ]
    if compared and report is not None:
        settings = report.settings
        bullets += [
            (
                f"Paired bootstrap of per-example differences: {settings.get('n_boot')} "
                f"resamples, seed {settings.get('seed')}, "
                f"{int(float(settings.get('confidence', 0.95)) * 100)}% percentile interval. "
                "Binary metrics also get an exact McNemar test on the discordant pairs."
            ),
            (
                f"Slices with fewer than {settings.get('min_slice_n')} examples are reported as "
                "insufficient data."
            ),
            "This shows the candidate scored differently on this dataset. It does not show why.",
        ]
    return "\n".join(["<h2>Method</h2>", "<ul>", *(f"<li>{b}</li>" for b in bullets), "</ul>"])


# --- page shell ----------------------------------------------------------------------------


def _page(title: str, body: Sequence[str]) -> str:
    """The whole document: styles, body and a provenance footer, in one file."""
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_h(title)}</title>",
            "<style>",
            _CSS.strip(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            f"<h1>{_h(title)}</h1>",
            *body,
            "<footer>",
            f"Written by AtlasForge {_h(__version__)}. Self-contained: no scripts, no network.",
            "</footer>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _facts(pairs: Sequence[tuple[str, object]]) -> str:
    lines = ["<dl>"]
    for key, value in pairs:
        lines.append(f"<dt>{_h(key)}</dt><dd>{_h(value)}</dd>")
    lines.append("</dl>")
    return "\n".join(lines)


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A plain table. Every cell is escaped, so a transcript can never break the markup."""
    lines = ['<div class="scroll">', "<table>", "<thead><tr>"]
    lines += [f"<th>{_h(header)}</th>" for header in headers]
    lines += ["</tr></thead>", "<tbody>"]
    lines += ["<tr>" + "".join(f"<td>{_h(cell)}</td>" for cell in row) + "</tr>" for row in rows]
    lines += ["</tbody>", "</table>", "</div>"]
    return "\n".join(lines)


def _note(text: str) -> str:
    return f'<p class="note">{_h(text)}</p>'


def _legend(*entries: tuple[str, str]) -> str:
    """Swatches matching the chart below. ``(label, css class)``."""
    spans = " ".join(f'<span class="swatch {css}"></span>{_h(label)}' for label, css in entries)
    return f'<p class="legend">{spans}</p>'


def _h(value: object) -> str:
    """Escape for HTML text and attributes. Everything from a report goes through this."""
    return escape(str(value), quote=True)


def _short(revision: str | None) -> str:
    return "-" if not revision else revision[:10]


# --- charts --------------------------------------------------------------------------------


def _bars(
    rows: Sequence[tuple[str, float | None, str]],
    *,
    maximum: float | None = None,
    ticks: Sequence[tuple[float, str]] = _PERCENT_TICKS,
    aria: str,
) -> str:
    """Horizontal bars sharing one scale: ``(label, value, value text)`` per row."""
    top = maximum or max((abs(v) for _, v, _ in rows if v is not None), default=0.0) or 1.0
    baseline = len(rows) * _ROW_H + 2
    parts: list[str] = []
    for index, (label, value, text) in enumerate(rows):
        y = index * _ROW_H + 2
        parts.append(_text(_LABEL_W - 10, y + _BAR_H - 3, label, anchor="end"))
        parts.append(_rect(_BAR_X, y, _BAR_W, _BAR_H, "track"))
        if value is not None:
            parts.append(_rect(_BAR_X, y, _BAR_W * abs(value) / top, _BAR_H, "bar"))
        parts.append(_text(_VALUE_X, y + _BAR_H - 3, text))
    height = baseline + _AXIS_H
    return _svg(height, aria, [*parts, *_axis(baseline, ticks)])


def _grouped(
    rows: Sequence[tuple[str, tuple[float, str] | None, tuple[float, str] | None]],
    *,
    maximum: float = 100.0,
    ticks: Sequence[tuple[float, str]] = _PERCENT_TICKS,
    aria: str,
) -> str:
    """Two bars per row, one above the other: ``(label, first, second)``.

    Each series is ``(position, text)``. The two are separate because a chart can be drawn
    against a shared scale while the number beside the bar keeps its own unit — counts of
    flagged answers are plotted in positions but must read as counts.
    """
    baseline = len(rows) * _GROUP_H + 2
    parts: list[str] = []
    for index, (label, first, second) in enumerate(rows):
        top = index * _GROUP_H + 4
        middle = top + (_GROUP_BAR_H * 2 + _GROUP_GAP) // 2 + 4
        parts.append(_text(_LABEL_W - 10, middle, label, anchor="end"))
        for offset, series, css in (
            (0, first, "bar"),
            (_GROUP_BAR_H + _GROUP_GAP, second, "bar-2"),
        ):
            y = top + offset
            parts.append(_rect(_BAR_X, y, _BAR_W, _GROUP_BAR_H, "track"))
            if series is not None:
                value, text = series
                parts.append(
                    _rect(_BAR_X, y, _BAR_W * min(abs(value), maximum) / maximum, _GROUP_BAR_H, css)
                )
            else:
                text = "-"
            parts.append(_text(_VALUE_X, y + _GROUP_BAR_H - 2, text))
    height = baseline + _AXIS_H
    return _svg(height, aria, [*parts, *_axis(baseline, ticks)])


def _diverging(
    rows: Sequence[tuple[str, float | None, float | None, float | None, str, str]],
    *,
    aria: str,
) -> str:
    """Deltas around a zero line, with interval whiskers: ``(label, delta, low, high, verdict, text)``."""
    reach = (
        max(
            (
                abs(value)
                for _, delta, low, high, _, _ in rows
                for value in (delta, low, high)
                if value is not None
            ),
            default=0.0,
        )
        or 1.0
    )
    centre = _BAR_X + _BAR_W / 2
    half = _BAR_W / 2 - 16
    baseline = len(rows) * _ROW_H + 2
    parts: list[str] = [
        f'<line x1="{centre:.1f}" y1="2" x2="{centre:.1f}" y2="{baseline:.1f}" class="axis"/>'
    ]
    for index, (label, delta, low, high, verdict, text) in enumerate(rows):
        y = index * _ROW_H + 8
        parts.append(_text(_LABEL_W - 10, y + _BAR_H - 3, label, anchor="end"))
        if delta is not None:
            end = centre + delta / reach * half
            css = {"improved": "improved", "regressed": "regressed"}.get(verdict, "flat")
            parts.append(_rect(min(centre, end), y, abs(end - centre), _BAR_H, css))
        if low is not None and high is not None:
            x1 = centre + low / reach * half
            x2 = centre + high / reach * half
            middle = y + _BAR_H / 2
            parts += [
                (
                    f'<line x1="{x1:.1f}" y1="{middle:.1f}" x2="{x2:.1f}" y2="{middle:.1f}" '
                    'class="whisker"/>'
                ),
                (
                    f'<line x1="{x1:.1f}" y1="{y + 2}" x2="{x1:.1f}" y2="{y + _BAR_H - 2}" '
                    'class="whisker"/>'
                ),
                (
                    f'<line x1="{x2:.1f}" y1="{y + 2}" x2="{x2:.1f}" y2="{y + _BAR_H - 2}" '
                    'class="whisker"/>'
                ),
            ]
        parts.append(_text(_VALUE_X, y + _BAR_H - 3, text))
    height = baseline + _AXIS_H
    scale = ((0.0, f"{-reach:.1f}"), (0.5, "0"), (1.0, f"+{reach:.1f}"))
    return _svg(height, aria, [*parts, *_axis(baseline, scale)])


def _axis(baseline: float, ticks: Sequence[tuple[float, str]]) -> list[str]:
    """The baseline, its tick marks and their labels. Positions are 0-1 fractions."""
    parts = [
        (
            f'<line x1="{_BAR_X}" y1="{baseline:.1f}" x2="{_BAR_X + _BAR_W}" '
            f'y2="{baseline:.1f}" class="axis"/>'
        )
    ]
    for position, label in ticks:
        x = _BAR_X + _BAR_W * position
        parts += [
            (
                f'<line x1="{x:.1f}" y1="{baseline:.1f}" x2="{x:.1f}" '
                f'y2="{baseline + 4:.1f}" class="axis"/>'
            ),
            _text(x, baseline + _AXIS_H - 4, label, anchor="middle", cls="tick"),
        ]
    return parts


def _svg(height: float, aria: str, parts: Sequence[str]) -> str:
    return "\n".join(
        [
            (
                f'<svg class="chart" viewBox="0 0 {_CHART_W} {height:.0f}" role="img" '
                f'aria-label="{_h(aria)}">'
            ),
            f"<title>{_h(aria)}</title>",
            *parts,
            "</svg>",
        ]
    )


def _text(x: float, y: float, content: str, *, anchor: str = "start", cls: str = "") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{klass}>{_h(content)}</text>'


def _rect(x: float, y: float, width: float, height: float, css: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" height="{height}" '
        f'class="{css}"/>'
    )


# --- numbers -------------------------------------------------------------------------------


def _scaled(name: str, x: float | None) -> float | None:
    """A value on the chart's scale: percent for fraction metrics, chrF points for chrF."""
    if x is None:
        return None
    return x * 100 if is_fraction(name) else x


def _series(name: str, x: float | None) -> tuple[float, str] | None:
    """One bar: where it ends on the shared scale, and the value it is labelled with."""
    if x is None:
        return None
    return (_scaled(name, x) or 0.0, fmt_value(name, x))


def _count_ticks(top: float) -> tuple[tuple[float, str], ...]:
    """Ticks for a chart of counts: zero, the midpoint and the maximum.

    Percentages are wrong here and would be read as such — six flagged answers is not "6%" of a
    scale nobody can see.
    """
    return ((0.0, "0"), (0.5, f"{top / 2:g}"), (1.0, f"{top:g}"))


def _interval(m: MetricComparison) -> str:
    if m.low is None or m.high is None:
        return "-"
    return f"[{fmt_bound(m.name, m.low)}, {fmt_bound(m.name, m.high)}]"


def _difference(m: MetricComparison) -> float | None:
    if m.base_corpus is None or m.candidate_corpus is None:
        return None
    return m.candidate_corpus - m.base_corpus


def _primary_name(report: ComparisonReport) -> str:
    return (report.primary or "").split("@")[0]
