# Python API reference

Generated from the source code's docstrings, so it is always in step with the code. For worked examples, see the [Python API guide](../guides/python-api.md).

## Types

::: atlasforge.types
    options:
      members:
        - Lang
        - parse_lang
        - Message
        - GenParams
        - Usage
        - Generation
        - Chunk
        - Transcript
        - BackendInfo

## Backends

::: atlasforge.backends.base

::: atlasforge.backends.factory.build_backend

::: atlasforge.backends.openai.OpenAIBackend
    options:
      members:
        - generate
        - transcribe
        - info
        - close

::: atlasforge.backends.local.LocalBackend
    options:
      members:
        - generate
        - transcribe
        - info
        - close

## Datasets

::: atlasforge.eval.dataset
    options:
      members:
        - Example
        - Dataset
        - load_dataset
        - parse_file

## Running

::: atlasforge.eval.runner
    options:
      members:
        - RunConfig
        - Record
        - RunSummary
        - run
        - read_manifest
        - read_results

## Scoring

::: atlasforge.eval.score
    options:
      members:
        - MetricSummary
        - ScoreReport
        - score_run
        - write_report_json

::: atlasforge.eval.normalize
    options:
      members:
        - NormalizeConfig
        - tone_aware
        - tone_insensitive
        - strip_tones
        - normalize

::: atlasforge.eval.metrics

## Validation

::: atlasforge.eval.validate
    options:
      members:
        - Issue
        - ValidationReport
        - validate_dataset

## Comparison

::: atlasforge.compare.compare
    options:
      members:
        - RunInfo
        - MetricComparison
        - ComparisonReport
        - compare_runs
        - compare_scores

::: atlasforge.compare.stats
    options:
      members:
        - BootstrapResult
        - McNemarResult
        - paired_bootstrap
        - mcnemar_exact

::: atlasforge.compare.slices
    options:
      members:
        - SliceResult
        - slice_value
        - analyse_slices
        - verdict

## Speech

::: atlasforge.asr.audio

::: atlasforge.asr.chunking
    options:
      members:
        - plan_windows
        - merge_texts
        - transcribe_long
        - LongAudioBackend

## Environment checks

::: atlasforge.doctor
    options:
      members:
        - Check
        - run_checks
        - exit_code

## Errors

::: atlasforge.errors

## Utilities

::: atlasforge.security
