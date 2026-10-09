"""The HTML reports: self-contained, escaped, and arithmetic-identical to the Markdown.

The page is a second rendering of numbers that already exist, so the tests check the two things a
renderer can get wrong on its own: producing a document that quietly depends on something outside
itself, and printing a figure the Markdown does not.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from atlasforge.compare.compare import FlagComparison, MetricComparison, compare_scores
from atlasforge.compare.report import to_markdown as comparison_markdown
from atlasforge.eval.asr_analysis import AsrAnalysis, LengthBucket, TopError
from atlasforge.eval.dataset import Dataset, Example, load_dataset
from atlasforge.eval.flags import DESCRIPTIONS
from atlasforge.eval.format import fmt_value
from atlasforge.eval.report import asr_summary, to_markdown
from atlasforge.eval.runner import Record
from atlasforge.eval.score import ScoreReport, score_run
from atlasforge.html import comparison_html, report_html

ROWS: list[dict[str, Any]] = [
    {"id": "e0", "input": "q0", "reference": "ans0", "meta": {"domain": "agri"}},
    {"id": "e1", "input": "q1", "reference": "ans1", "meta": {"domain": "agri"}},
    {"id": "e2", "input": "q2", "reference": "ans2", "meta": {"domain": "legal"}},
    {"id": "e3", "input": "q3", "reference": "ans3", "meta": {"domain": "legal"}},
]


def dataset(tmp_path: Path, rows: list[dict[str, Any]] | None = None) -> Dataset:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows or ROWS) + "\n",
        encoding="utf-8",
    )
    return load_dataset(path, "generation")


def records(predictions: dict[str, str | None]) -> dict[str, Record]:
    return {
        example_id: Record(id=example_id, prediction=prediction, latency_ms=3.0)
        if prediction is not None
        else Record(id=example_id, error="BackendTimeout: x")
        for example_id, prediction in predictions.items()
    }


def scored(tmp_path: Path, predictions: dict[str, str | None] | None = None) -> ScoreReport:
    return score_run(
        dataset(tmp_path),
        records(predictions or {"e0": "ans0", "e1": "wrong", "e2": None, "e3": "ans3"}),
        metrics=["exact_match", "chrf"],
    )


def asr_dataset(pairs: list[tuple[str, str, str]]) -> Dataset:
    """An in-memory speech dataset: no audio is opened to score transcripts."""
    return Dataset(
        task="asr",
        path=Path("asr.jsonl"),
        sha256="0" * 64,
        examples=tuple(
            Example(id=example_id, reference=reference, audio=Path("a.wav"), lang="ha")
            for example_id, reference, _ in pairs
        ),
    )


def asr_report(pairs: list[tuple[str, str, str]]) -> ScoreReport:
    """A real scoring run: scoring computes the analysis itself, so nothing is hand-assembled."""
    return score_run(
        asr_dataset(pairs),
        records({example_id: hypothesis for example_id, _, hypothesis in pairs}),
    )


def comparison(
    tmp_path: Path,
    *,
    base_correct: int = 0,
    candidate_correct: int = 2,
    min_slice_n: int = 30,
) -> Any:
    """Two runs over the same four examples, right on the first N each.

    ``min_slice_n`` defaults to the production value, which is above the two examples each slice
    holds — so the slice tables say "insufficient data" unless a test lowers it deliberately.
    """
    data = dataset(tmp_path)

    def run(correct: int) -> ScoreReport:
        return score_run(
            data,
            records({f"e{i}": (f"ans{i}" if i < correct else "wrong") for i in range(len(ROWS))}),
            metrics=["exact_match"],
        )

    return compare_scores(
        data,
        run(base_correct),
        run(candidate_correct),
        slice_fields=["domain"],
        n_boot=400,
        min_slice_n=min_slice_n,
    )


def metric(
    *,
    verdict: str,
    delta: float | None,
    name: str = "exact_match",
    base_corpus: float | None = None,
    candidate_corpus: float | None = None,
) -> MetricComparison:
    """One metric row with its verdict chosen by hand.

    Four examples cannot produce a clear metric verdict — every bootstrap interval over them
    straddles zero — so the mapping from verdict to colour is tested where it lives, in the
    renderer. Which verdict a run earns is the comparison engine's question, tested in
    ``test_compare``.
    """
    return MetricComparison(
        key=f"{name}@tone_aware",
        name=name,
        view="tone_aware",
        higher_is_better=True,
        n=4,
        base_mean=0.0,
        candidate_mean=delta,
        delta=delta,
        low=delta,
        high=delta,
        verdict=verdict,
        wins=2,
        ties=2,
        losses=0,
        base_corpus=base_corpus,
        candidate_corpus=candidate_corpus,
        mcnemar=None,
    )


def chart(html: str, aria: str) -> str:
    """The one SVG whose accessible name is ``aria`` — the chart, and nothing around it."""
    start = html.index(f'aria-label="{aria}"')
    return html[start : html.index("</svg>", start)]


class TestSelfContained:
    def test_it_is_a_whole_document(self, tmp_path: Path) -> None:
        html = report_html(scored(tmp_path))
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        for tag in ('<html lang="en">', '<meta charset="utf-8">', "<title>", "</style>"):
            assert tag in html

    def test_nothing_in_it_reaches_the_network(self, tmp_path: Path) -> None:
        """An air-gapped machine and an email attachment both have to render this unchanged."""
        html = report_html(scored(tmp_path)) + comparison_html(comparison(tmp_path))
        assert "<script" not in html
        assert "http://" not in html
        assert "https://" not in html
        assert "<link" not in html
        assert "src=" not in html
        assert "@import" not in html

    def test_it_carries_a_dark_variant(self, tmp_path: Path) -> None:
        html = report_html(scored(tmp_path))
        assert "prefers-color-scheme: dark" in html
        assert "color-scheme: light dark" in html

    def test_every_chart_is_labelled_for_a_screen_reader(self, tmp_path: Path) -> None:
        html = report_html(scored(tmp_path))
        svgs = html.count("<svg")
        assert svgs >= 2  # metrics, and the flags chart
        assert html.count('role="img"') == svgs
        assert html.count("<title>") == svgs + 1  # one per chart, plus the document's


class TestTheNumbersMatchTheMarkdown:
    def test_every_metric_value_appears_as_the_markdown_writes_it(self, tmp_path: Path) -> None:
        report = scored(tmp_path)
        html = report_html(report)
        for summary in report.metrics:
            assert fmt_value(summary.name, summary.mean) in html
            assert fmt_value(summary.name, summary.corpus) in html

    def test_the_run_identity_matches_the_manifest(self, tmp_path: Path) -> None:
        html = report_html(
            scored(tmp_path), {"model": "NCAIR1/N-ATLaS", "backend": "local", "revision": "a" * 40}
        )
        assert "NCAIR1/N-ATLaS" in html
        assert "local" in html
        assert "a" * 10 in html
        assert "a" * 40 not in html  # the short revision, as the Markdown shows it

    def test_failures_are_counted_the_same_way(self, tmp_path: Path) -> None:
        report = scored(tmp_path)
        html = report_html(report)
        assert "4 (3 ok, 1 failed, 0 missing)" in html

    def test_a_metric_with_no_value_says_so_instead_of_plotting_zero(self, tmp_path: Path) -> None:
        """macro_f1 has no per-example value on a generation task; a bar at zero would be a lie."""
        report = replace(
            scored(tmp_path),
            metrics=(
                *scored(tmp_path).metrics,
                *(
                    replace(summary, name="macro_f1", mean=None)
                    for summary in scored(tmp_path).metrics[:2]
                ),
            ),
        )
        html = report_html(report)
        assert html.count("macro_f1") >= 2

    def test_the_flag_counts_are_counts_not_percentages(self, tmp_path: Path) -> None:
        """A count chart labelled 0%-100% reads as a share of something the reader cannot see."""
        report = replace(scored(tmp_path), flags={"empty_output": 6}, n_flagged=6)
        html = report_html(report)
        counts = chart(html, "Answers raising each failure-mode flag")
        assert DESCRIPTIONS["empty_output"] in html
        assert ">6</text>" in counts
        assert "%" not in counts

    def test_the_latency_line_is_the_same_line(self, tmp_path: Path) -> None:
        report = scored(tmp_path)
        markdown = to_markdown(report)
        assert "mean 3 ms" in markdown
        assert "3 ms" in report_html(report)


class TestEscaping:
    def test_markup_in_a_manifest_field_cannot_change_the_page(self, tmp_path: Path) -> None:
        html = report_html(scored(tmp_path), {"model": "<script>alert(1)</script>"})
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_markup_in_a_transcript_is_shown_as_text(self, tmp_path: Path) -> None:
        """Transcripts are the least trustworthy text in the report: they come from a model."""
        report = asr_report([("a", "ina kwana", "ina kwanda")])
        analysis = report.asr
        assert analysis is not None
        hostile = replace(
            analysis,
            substitutions=1,
            top_substitutions=(
                TopError(
                    reference="<img src=x onerror=alert(1)>",
                    hypothesis="ok",
                    count=1,
                    tone_only=False,
                    examples=("a",),
                ),
            ),
        )
        html = report_html(replace(report, asr=hostile))
        assert "<img src=x" not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html


class TestSpeechSection:
    def test_the_summary_is_the_shared_one(self) -> None:
        report = asr_report([("a", "ina kwana", "ina kwanda")])
        analysis = report.asr
        assert analysis is not None
        assert asr_summary(analysis) in report_html(report)
        assert asr_summary(analysis) in to_markdown(report)

    def test_tone_only_substitutions_are_marked(self) -> None:
        html = report_html(asr_report([("a", "iná kwana", "ina kwana")]))
        assert "differ only in tone marks" in html
        assert "iná → ina" in html
        assert ">yes</td>" in html

    def test_the_length_buckets_are_a_percentage_chart(self) -> None:
        report = asr_report([("a", "ina kwana", "ina kwanda")])
        buckets = chart(report_html(report), "Pooled word error rate by reference length in words")
        assert ">100%</text>" in buckets
        assert "1-5 words (n=1)" in buckets

    def test_no_speech_section_without_speech(self, tmp_path: Path) -> None:
        assert "ASR error analysis" not in report_html(scored(tmp_path))

    def test_a_classification_report_has_no_flags_section(self, tmp_path: Path) -> None:
        """Flags are computed over generated text; a classification answer has none to flag."""
        report = replace(scored(tmp_path), task="classification", flags={}, n_flagged=0)
        html = report_html(report)
        assert "Failure-mode flags" not in html
        assert "<h2>Metrics</h2>" in html  # the rest of the report is still there

    def test_a_speech_run_with_no_utterances_skips_the_length_chart(self) -> None:
        """An alignment with nothing in it has no buckets, and an empty chart is not a finding."""
        analysis = AsrAnalysis(
            n=0,
            hits=0,
            substitutions=0,
            deletions=0,
            insertions=0,
            tone_only_substitutions=0,
            top_substitutions=(),
            top_deletions=(),
            top_insertions=(),
            by_length=(),
        )
        html = report_html(replace(_blank(), asr=analysis))
        assert "ASR error analysis" in html
        assert "Word error rate by reference length" not in html

    def test_a_bucket_with_no_rate_prints_a_dash_not_a_zero(self) -> None:
        report_empty = AsrAnalysis(
            n=0,
            hits=0,
            substitutions=0,
            deletions=0,
            insertions=0,
            tone_only_substitutions=0,
            top_substitutions=(),
            top_deletions=(),
            top_insertions=(),
            by_length=(LengthBucket(label="1-5", n=0, wer=None),),
        )
        html = report_html(replace(_blank(), asr=report_empty))
        assert "1-5 words (n=0)" in html
        assert ">-</text>" in html


def _blank() -> ScoreReport:
    """A report with no metrics at all, so a section can be tested in isolation."""
    return ScoreReport(
        task="asr",
        dataset_sha256="0" * 64,
        n_total=0,
        n_ok=0,
        n_failed=0,
        n_missing=0,
        metrics=(),
        per_example={},
        latency_ms={"mean": None, "p50": None, "p95": None, "max": None},
        normalization={},
    )


class TestComparisonPage:
    def test_it_names_both_runs(self, tmp_path: Path) -> None:
        report = comparison(tmp_path)
        html = comparison_html(report)
        assert "Base" in html
        assert "Candidate" in html
        assert report.dataset_sha256[:16] in html

    def test_every_delta_is_written_the_way_the_markdown_writes_it(self, tmp_path: Path) -> None:
        report = comparison(tmp_path, min_slice_n=2)
        markdown = comparison_markdown(report)
        html = comparison_html(report)
        assert "+50.0 pts" in markdown  # two of four examples went from wrong to right
        assert "+50.0 pts" in html
        assert "+100.0 pts" in markdown  # the agri slice: both its examples flipped
        assert "+100.0 pts" in html

    def test_an_improvement_is_coloured_as_one(self, tmp_path: Path) -> None:
        report = replace(comparison(tmp_path), metrics=(metric(verdict="improved", delta=0.5),))
        deltas = chart(
            comparison_html(report),
            "Change per metric, candidate minus base, with 95% intervals",
        )
        assert 'class="improved"' in deltas
        assert 'class="whisker"' in deltas  # the interval is drawn, not just stated

    def test_a_regression_is_coloured_as_one(self, tmp_path: Path) -> None:
        """Lower is better for WER, so the sign of a delta is not the good/bad answer."""
        report = replace(comparison(tmp_path), metrics=(metric(verdict="regressed", delta=-0.5),))
        html = comparison_html(report)
        assert 'class="regressed"' in chart(
            html, "Change per metric, candidate minus base, with 95% intervals"
        )
        assert "-50.0 pts" in html
        assert "No statistically clear regressions found" not in html
        assert "exact_match · tone-aware" in html  # the named regression is spelled out

    def test_a_run_with_no_clear_change_is_not_called_a_regression(self, tmp_path: Path) -> None:
        """Four examples, two flipped: the interval straddles zero and the page must say so."""
        html = comparison_html(comparison(tmp_path, min_slice_n=2))
        assert "No statistically clear regressions found" in html
        assert 'class="flat"' in chart(
            html, "Change per metric, candidate minus base, with 95% intervals"
        )

    def test_a_pooled_only_metric_says_so_and_is_not_charted(self, tmp_path: Path) -> None:
        """macro-F1 has no per-example value, so there is no interval to draw or to claim."""
        report = replace(
            comparison(tmp_path),
            metrics=(metric(verdict="not tested (pooled metric)", delta=None, name="macro_f1"),),
        )
        html = comparison_html(report)
        assert "macro_f1: pooled metric, no per-example values, so no interval." in html
        assert "Change per metric, candidate minus base" not in html
        assert ">-</td>" in html  # the interval column, not an invented range

    def test_a_comparison_with_no_slices_omits_the_section(self, tmp_path: Path) -> None:
        report = replace(comparison(tmp_path), slices=())
        html = comparison_html(report)
        assert "<h2>Slices" not in html
        assert "<h2>Overall</h2>" in html

    def test_a_pooled_figure_missing_from_one_run_prints_a_dash(self, tmp_path: Path) -> None:
        """A run that did not report the pooled figure is not a run that scored zero."""
        report = replace(
            comparison(tmp_path),
            metrics=(metric(verdict="improved", delta=0.5, name="chrf", candidate_corpus=41.5),),
        )
        html = comparison_html(report)
        assert "<h3>Pooled figures</h3>" in html
        assert "41.5" in html
        assert ">-</td>" in html

    def test_slices_are_shown_with_their_own_chart(self, tmp_path: Path) -> None:
        html = comparison_html(comparison(tmp_path, min_slice_n=2))
        assert "<h3>domain</h3>" in html
        assert "Change per value of domain" in html
        assert 'class="improved"' in chart(html, "Change per value of domain, candidate minus base")

    def test_a_slice_too_small_to_judge_is_not_charted(self, tmp_path: Path) -> None:
        """Two examples is not evidence. The table says so instead of drawing a confident bar."""
        html = comparison_html(comparison(tmp_path))
        assert "insufficient data" in html
        assert "Change per value of domain" not in html

    def test_a_flag_delta_reads_as_a_count(self, tmp_path: Path) -> None:
        report = replace(
            comparison(tmp_path),
            flags=(FlagComparison(name="empty_output", base=2, candidate=9, delta=7),),
        )
        flags_chart = chart(
            comparison_html(report),
            "Answers raising each failure-mode flag, base against candidate",
        )
        assert ">9</text>" in flags_chart
        assert ">50%</text>" not in flags_chart


class TestMarkdownAndHtmlAgree:
    def test_one_pooled_word_error_rate_across_both_renderers(self) -> None:
        """The arithmetic that was wrong once: an insertion is an error but not a reference word."""
        report = asr_report(
            [
                ("a", "ina kwana", "ina kwanda"),  # substituted
                ("b", "sannu da zuwa", "sannu zuwa"),  # dropped
                ("c", "yaya kake", "yaya kake sosai"),  # inserted
            ]
        )
        analysis = report.asr
        assert analysis is not None
        reference_words = analysis.hits + analysis.substitutions + analysis.deletions
        assert analysis.insertions == 1  # the case the denominator must not count
        rate = f"{analysis.errors / reference_words:.1%}"
        assert rate in to_markdown(report)
        assert rate in report_html(report)
