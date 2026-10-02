# Dataset format

A dataset is a **JSONL** file: UTF-8 text with one JSON object per line. The same format serves all three tasks.

```json
{"id": "q1", "input": "Translate to English: Sannu", "reference": "Hello", "lang": "ha", "meta": {"domain": "greetings"}}
```

## Fields

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string or integer | no | unique identifier. If omitted, a 12-character fingerprint is derived from the content |
| `input` | string | one of `input`/`messages` (not `asr`) | the user prompt: shorthand for a single user message |
| `messages` | list | one of `input`/`messages` (not `asr`) | a chat: a list of `{"role", "content"}` turns |
| `audio` | string | `asr` only | path to an audio file, relative to the dataset file |
| `reference` | string | `classification` and `asr`; optional for `generation` | the expected output |
| `lang` | string | no (needed for ASR, here or via `--lang`) | `ha`, `yo`, `ig` or `en` (or `hausa`, `yoruba`, `igbo`, `english`) |
| `meta` | object | no | free-form; used for [slice analysis](../guides/compare.md#slicing-by-your-own-fields) |

**Unknown keys are rejected**, so a typo such as `refrence` fails loudly instead of silently dropping your references. Put any extra columns inside `meta`.

## Rules

- Provide **exactly one** of `input` or `messages`, except for `asr`, which takes `audio` instead and rejects both.
- `input` must be a non-empty string. `messages` must be a non-empty list, and every turn needs a `role` of `system`, `user` or `assistant` and a string `content`.
- `audio` is only valid for the `asr` task, and the file must exist. A relative path is resolved against the directory of the dataset file.
- `reference` must be a string when present. It is required for `classification` and `asr`.
- `lang` must be a recognised language, or the line is rejected.
- `meta` must be an object.
- `id` must be unique across the file. Duplicates are errors reported with both line numbers.
- Blank lines are ignored. A UTF-8 byte-order mark is accepted. Lines are split on `\n` only, so characters such as `U+2028` inside a string are safe.
- The file must be valid UTF-8; otherwise loading fails with the offending byte offset.
- A file with no examples is an error.

Loading reports the **first** problem with its line number (`line 7: ...`). Use [`dataset validate`](../guides/validate-dataset.md) to see them all at once.

## Examples by task

=== "generation"

    Free-text output scored against an optional reference.

    ```json
    {"id": "t1", "input": "Translate to English: Ẹ káàárọ̀", "reference": "Good morning", "lang": "yo"}
    {"id": "t2", "messages": [
      {"role": "system", "content": "Answer briefly."},
      {"role": "user", "content": "What is the capital of Nigeria?"}
    ], "reference": "Abuja", "lang": "en"}
    ```

    (Multi-turn `messages` must be on **one line** in a real file; they are wrapped here for reading.)

=== "classification"

    `reference` is the label. Ask the model to answer with the label only.

    ```json
    {"id": "s1", "input": "Is this review positive or negative? 'The food was excellent.'", "reference": "positive", "lang": "en"}
    ```

=== "asr"

    `audio` plus a reference transcript. Each example needs a language, either on the line or via `eval --lang`.

    ```json
    {"id": "c1", "audio": "clips/c1.wav", "reference": "Ina kwana", "lang": "ha"}
    ```

## The dataset fingerprint

Every dataset has a SHA-256 hash of its exact bytes. It is stored in each run's manifest and shown (abbreviated) in reports. `compare` uses it to guarantee that two runs answered the same questions. **Any edit to the file, even whitespace, changes the hash**, which is deliberate: editing a dataset between two runs makes them incomparable, and AtlasForge tells you so rather than comparing anyway.

## Building a good dataset

- **Use real examples from your task**, not invented ones. A dataset that does not represent your use case gives confident but irrelevant numbers.
- **Aim for hundreds of examples.** With a handful, intervals are too wide to conclude anything.
- **Add `meta` fields you will want to slice by** (domain, source, difficulty) while building the dataset, not afterwards. Each slice needs at least 30 examples to ever get a verdict.
- **Keep diacritics.** Do not strip Yoruba tone marks and underdots or Hausa hooked letters; they change meaning. [`validate`](../guides/validate-dataset.md) warns when they appear to be missing.
- **Keep train and test separate**, and prove it with `dataset validate --against`.
- **Mind licences and consent** for any text or audio you include. Never commit sensitive data or raw voice recordings.

## Python

```python
from atlasforge.eval.dataset import load_dataset

dataset = load_dataset("data.jsonl", "generation")
len(dataset)                  # number of examples
dataset.sha256                # the fingerprint
dataset.examples[0].messages  # (({'role': 'user', 'content': '…'}),)
```

For lenient parsing that collects every error instead of stopping, use `parse_file`, which `dataset validate` is built on.
