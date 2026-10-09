# Datasets

AtlasForge reads **JSONL**: one JSON object per line, UTF-8. The same file format is used for evaluation, validation and fine-tuning.

## An example

Here are the first two lines of the demo dataset:

{{ file_example("dataset") }}

## Fields

| Field | Required | Meaning |
|---|---|---|
| `id` | no | A unique identifier. If omitted, a short fingerprint of the content is used. Give explicit ids if two examples could be identical. |
| `input` | one of these two | The user's prompt, as text. |
| `messages` | one of these two | A list of `{"role": ..., "content": ...}` turns, with roles `system`, `user` or `assistant`. Use this for a system prompt or multi-turn context. |
| `audio` | speech only | Path to an audio file, relative to the dataset file. |
| `reference` | depends on task | The expected answer. |
| `lang` | no | `ha`, `yo`, `ig` or `en`. Needed for speech (per example, or as a default with `--lang`). |
| `meta` | no | A free-form object. Every key becomes a possible [slice](statistics.md#slices). |

**Unknown top-level keys are rejected.** A typo like `refrence` fails loudly instead of silently producing an unscored example. Put your own columns under `meta`.

## Tasks

The `--task` option tells AtlasForge what the dataset is for.

| Task | Needs | Default metrics | Notes |
|---|---|---|---|
| `generation` | `input` or `messages`. `reference` is optional. | `exact_match`, `chrf` | Examples without a reference are run but not scored. |
| `classification` | `input` or `messages`, and `reference` | `accuracy`, `macro_f1` | The labels are the distinct references. The model's answer is matched to the label it names. |
| `asr` | `audio` and `reference` | `wer`, `cer` | `input` and `messages` are not allowed. |

!!! tip "Classification: ask for the label only"
    The scorer finds the earliest whole-word label in the answer. It does not understand negation, so "not positive" would count as `positive`. Prompt the model to answer with the label and nothing else.

## Slicing with `meta`

Put anything you want to break results down by under `meta`:

```json
{"id": "q1", "input": "...", "reference": "...", "meta": {"domain": "agri", "source": "survey-2026"}}
```

Then `atlasforge compare ... --slice domain --slice source` reports each value separately. See [Statistics](statistics.md#slices) for how small slices are handled.

## Speech datasets

```json
{"id": "u1", "audio": "clips/u1.ogg", "reference": "ina kwana", "lang": "ha"}
```

Audio paths are resolved relative to the dataset file, and every file must exist when the dataset is loaded. See [Transcribe speech](../guides/transcribe-speech.md).

## Strictness and fingerprints

- Loading stops at the **first** problem with a line number. [`dataset validate`](../guides/validate-your-data.md) lists **all** problems at once.
- Blank lines are ignored. A leading BOM and Windows line endings are accepted. A file that is not valid UTF-8 is rejected.
- Lines are split on `\n` only, so a JSON string containing the Unicode line separator is not broken in two.
- The whole file is **fingerprinted** (SHA-256). Every run records the fingerprint, and `compare` refuses to pair runs made from different dataset files.

## Good practice

- Keep a **held-out test set** separate from anything you train on. Check it with `dataset validate --against`.
- Keep diacritics. A dataset with tone marks stripped teaches and tests the wrong thing; the validator warns about it.
- State where the data came from and its licence. The [model card](../guides/publish-a-model-card.md) requires this.
