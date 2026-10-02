# Python API

Everything the CLI does is available as a library. The CLI is a thin layer over these functions.

!!! warning "The API is pre-1.0"
    Names and signatures may still change before a stable release. The CLI is the more stable surface for now.

## A first example

This runs offline against the [demo data](../get-started/quickstart.md) (create it first with `atlasforge demo`):

```python
# docs:run
from atlasforge.eval.dataset import load_dataset
from atlasforge.eval.runner import read_results
from atlasforge.eval.score import score_run

dataset = load_dataset("atlasforge-demo/toy_qa.jsonl", "generation")
results = read_results("atlasforge-demo/runs/tuned/results.jsonl")
report = score_run(dataset, results, metrics=["exact_match"])

summary = report.metric("exact_match@tone_aware")
print(f"{summary.mean:.1%} over {summary.n} examples; {report.n_failed} failed")
```

## The typical flow

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

Importing `atlasforge` itself is cheap: it imports no machine-learning framework.

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
