# CLI reference

This page is generated from the program's own command definitions each time the docs are built, so it always matches the version you have installed (`atlasforge --version`).

Run `atlasforge COMMAND --help` for the same information in your terminal.

## Global options

| Option | Description |
|---|---|
| `--version`, `-V` | Print the version and exit |
| `--help` | Show help for any command |

## Commands

| Command | Purpose |
|---|---|
| [`doctor`](#atlasforge-doctor) | Check your environment |
| [`demo`](#atlasforge-demo) | Write synthetic demo data (no model needed) |
| [`run`](#atlasforge-run) | Send one prompt to a model |
| [`transcribe`](#atlasforge-transcribe) | Transcribe audio |
| [`eval`](#atlasforge-eval) | Run and score a dataset |
| [`report`](#atlasforge-report) | Re-score a finished run |
| [`compare`](#atlasforge-compare) | Compare two runs |
| [`dataset validate`](#atlasforge-dataset-validate) | Check a dataset |
| [`finetune`](#atlasforge-finetune) | Fine-tune with QLoRA |
| [`card`](#atlasforge-card) | Write a model card |

## Details

{{ cli_reference() }}

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The command ran but reports a problem: `doctor` found a failed check, `dataset validate` found errors, `eval` had every example fail, or `transcribe` had a failed file |
| `2` | AtlasForge could not do what you asked: bad configuration, unreachable server, missing file, refused data. The message and a `->` hint say what to do. |

See [Errors and exit codes](errors.md).
