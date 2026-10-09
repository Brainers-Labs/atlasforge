# Python API

Everything the CLI does is available as a library, and the CLI is a thin layer over it: each
command calls the same functions you can call yourself. Two entry points cover the workflow —
`atlasforge.evaluate()` and `atlasforge.compare_runs()` — and the modules further down are
there for when you want the steps separately.

!!! warning "The API is pre-1.0"
    Names and signatures may still change before a stable release. The CLI is the more stable surface for now.

## Evaluate a model

```python
import atlasforge

evaluation = atlasforge.evaluate(
    "examples/translation_ha_en.jsonl",  # a path, or a Dataset you already loaded
    out_dir="runs/ha_en_base",
    backend="openai",  # or "local", or a Backend you built yourself
    base_url="http://gpu-box:8000/v1",
    task="generation",
    metrics=["exact_match", "chrf"],
)

print(evaluation.metric("chrf@tone_aware").mean)
```

`evaluation` holds the three things you would otherwise assemble by hand: `evaluation.run`
(the totals), `evaluation.scores` (the report) and `evaluation.out_dir`, where `report.md`,
`report.json` and `report.html` were written. The run directory is resumable: call it again and
examples already finished there are skipped.

Pass a `Backend` object to reuse one connection — it stays yours, and AtlasForge will not
close it. Name a backend with a string and AtlasForge creates it and closes it for you.
Speech datasets are wrapped with the long-audio splitter automatically.

## Compare two runs

No model is involved: this is arithmetic over results you already have. It runs offline
against the [demo data](../get-started/quickstart.md) (create it first with `atlasforge demo`):

```python
# docs:run
import atlasforge

comparison = atlasforge.compare_runs(
    "atlasforge-demo/toy_qa.jsonl",
    "atlasforge-demo/runs/base",
    "atlasforge-demo/runs/tuned",
    out_dir="comparison",
)

for metric in comparison.metrics:
    print(f"{metric.name} ({metric.view}): {metric.verdict}")
```

Both runs must have been made from exactly this dataset. The dataset hash in each run's
manifest is checked, so a mismatched pair is refused rather than compared silently. Leave
off `out_dir` to keep it in memory and write nothing.

## Re-score a finished run

Scoring needs no model, and it reads `results.jsonl` rather than re-running anything:

```python
# docs:run
import atlasforge

report = atlasforge.score_finished_run("atlasforge-demo/runs/tuned", "atlasforge-demo/toy_qa.jsonl")

summary = report.metric("exact_match@tone_aware")
print(f"{summary.mean:.1%} over {summary.n} examples; {report.n_failed} failed")
```

`atlasforge.write_reports(run_dir, dataset)` does the same and rewrites `report.md`,
`report.json` and `report.html` — it is what the `atlasforge report` command calls.

`metrics=[...]` anywhere in this API takes built-in names, your own callables, or
`"module:function"` strings naming them. See [your own metrics](metrics.md#your-own-metrics).

## The typical flow

The two entry points above call these; call them directly when you want the steps apart.

| Step | Function |
|---|---|
| Load and validate a dataset | `atlasforge.eval.dataset.load_dataset` |
| Check data health | `atlasforge.eval.validate.validate_dataset` |
| Create a backend | `atlasforge.backends.openai.OpenAIBackend` or `atlasforge.backends.local.LocalBackend` |
| Run it | `atlasforge.eval.runner.run` |
| Score it | `atlasforge.eval.score.score_run` |
| Compare two runs | `atlasforge.compare.compare_runs` |
| Fine-tune | `atlasforge.finetune.train` |
| Write a card | `atlasforge.cards.render_card` |

Importing `atlasforge` itself is cheap and pulls in no machine-learning framework: the entry
points import what they need when you call them.

## Entry points

::: atlasforge.api
    options:
      members:
        - evaluate
        - Evaluation
        - compare_runs
        - score_finished_run
        - write_reports
        - write_comparison

## Datasets

::: atlasforge.eval.dataset
    options:
      members:
        - load_dataset
        - parse_file
        - Dataset
        - Example

## Running

::: atlasforge.eval.runner
    options:
      members:
        - run
        - RunConfig
        - RunSummary
        - Record
        - read_results
        - read_manifest

## Scoring

::: atlasforge.eval.score
    options:
      members:
        - score_run
        - ScoreReport
        - MetricSummary
        - write_report_json

## Rendering a report

The Markdown renderer behind `report.md`, for when you want the same rendering somewhere else:

::: atlasforge.eval.report
    options:
      members:
        - to_markdown
        - asr_summary

::: atlasforge.eval.custom
    options:
      members:
        - resolve
        - name_of
        - declares_higher_is_better

## Single metrics

The functions behind the metric names, for scoring one answer at a time instead of a whole run.
[Scales and direction](metrics.md) apply as written there: chrF is 0-100, WER and CER are lower-is-better,
and `extract_label` is the loose match by default — pass `strict=True` for the whole-answer rule that
`accuracy_strict` and `macro_f1_strict` use.

```python
from atlasforge.eval.metrics import extract_label, chrf

extract_label("the sentiment is not positive", ["positive", "negative"])  # 'positive'
extract_label("the sentiment is not positive", ["positive", "negative"], strict=True)  # None
```

These take text as given: they do not normalise. For the two tone views, normalise first —
:func:`atlasforge.eval.normalize.normalize` below does what the scorers do — or call
`atlasforge.eval.score.score_run`, which applies the same normalisation as the CLI.

::: atlasforge.eval.metrics
    options:
      members:
        - exact_match
        - chrf
        - corpus_chrf
        - wer
        - cer
        - corpus_wer
        - corpus_cer
        - extract_label
        - macro_f1
        - mean
        - percentile

How a figure is *written* is separate from how it is computed, so that every report renders the same
number the same way — `71.4%` for a fraction, `54.3` for chrF:

::: atlasforge.eval.format
    options:
      members:
        - fmt_value
        - fmt_delta
        - fmt_bound
        - is_fraction
        - FRACTION_METRICS

## Failure-mode flags

::: atlasforge.eval.flags
    options:
      members:
        - flags_for
        - missing_terms
        - summarise
        - FLAG_NAMES
        - DESCRIPTIONS

## ASR error analysis

::: atlasforge.eval.asr_analysis
    options:
      members:
        - analyse
        - AsrAnalysis
        - TopError
        - LengthBucket
        - LENGTH_BUCKETS

## Normalisation

::: atlasforge.eval.normalize
    options:
      members:
        - normalize
        - NormalizeConfig
        - tone_aware
        - tone_insensitive
        - strip_tones

## Validation

::: atlasforge.eval.validate
    options:
      members:
        - validate_dataset
        - ValidationReport
        - Issue

## Comparing

::: atlasforge.compare.compare
    options:
      members:
        - compare_runs
        - compare_scores
        - ComparisonReport
        - MetricComparison
        - RunInfo

::: atlasforge.compare.stats
    options:
      members:
        - paired_bootstrap
        - mcnemar_exact
        - BootstrapResult
        - McNemarResult

::: atlasforge.compare.slices
    options:
      members:
        - analyse_slices
        - slice_value
        - SliceResult

## Backends

::: atlasforge.backends.base.Backend

::: atlasforge.backends.openai.OpenAIBackend

::: atlasforge.backends.local.LocalBackend

## Speech

::: atlasforge.asr.chunking
    options:
      members:
        - transcribe_long
        - plan_windows
        - plan_windows_silence_aware
        - frame_levels
        - merge_texts
        - LongAudioBackend

::: atlasforge.asr.audio
    options:
      members:
        - decode_audio
        - pcm_to_wav

## Fine-tuning

::: atlasforge.finetune.config
    options:
      members:
        - QLoRAConfig
        - load_config

::: atlasforge.finetune.qlora
    options:
      members:
        - train
        - TrainingRun

## Benchmarks

The AfroBench-LITE wrapper behind `atlasforge bench afrobench`. It shells out to
`lm-evaluation-harness` rather than reimplementing the tasks, and it takes the task names from the
installed harness — see the [guide](../guides/benchmark-with-afrobench.md).

::: atlasforge.bench
    options:
      members:
        - run_afrobench
        - discover_tasks
        - missing_families
        - build_command
        - parse_results
        - write_bench
        - find_results
        - BenchReport
        - TaskResult

## Model cards

::: atlasforge.cards.model_card
    options:
      members:
        - CardInfo
        - render_card
        - check_card
        - suggested_name

## Shared types

::: atlasforge.types
    options:
      members:
        - GenParams
        - Generation
        - Transcript
        - Chunk
        - BackendInfo
        - Message
        - parse_lang
