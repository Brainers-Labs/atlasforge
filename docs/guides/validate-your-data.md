# Validate your data

Most bad evaluation results trace back to bad data. `dataset validate` finds the problems that quietly ruin an evaluation or a fine-tune, **all at once**, before you spend time or GPU hours.

```bash
atlasforge dataset validate data.jsonl --task generation
```

Try it on the [demo data](../get-started/quickstart.md#3-check-the-data) to see a clean result.

## Reading the output

Each finding has a severity:

| Label | Meaning | Effect |
|---|---|---|
| `ERROR` | The data cannot be used as is | Exit code **1** |
| `WARN` | Probably a mistake, but the data is usable | Exit code 0 |
| `note` | Worth knowing | Exit code 0 |

The last line summarises: `OK` or `FAILED`, how many valid examples came from how many lines, and the error and warning counts.

## What it checks

| Check | Severity | Why it matters | Typical fix |
|---|---|---|---|
| **Malformed lines**: bad JSON, unknown keys, missing fields, duplicate ids | error | The line cannot be loaded. Every bad line is listed with its **line number**. | Fix the line |
| **Train/test leakage** (with `--against`) | error | Examples in both splits make an evaluation measure memorisation | Remove them from one side |
| **Empty reference** (classification, speech) | error | The example cannot be scored | Add the answer or drop the row |
| **Duplicate content** | warn | Repeats inflate scores and can leak between splits | Remove repeats |
| **Conflicting references** | warn | One question has two different "correct" answers | Decide which is right |
| **Not NFC** | warn | The same letter stored two ways compares as different in training | Normalise to NFC |
| **Replacement character** (`U+FFFD`) | warn | Encoding damage: the original characters are lost | Re-export the source as UTF-8 |
| **Control characters** | warn | Usually corruption | Clean them |
| **Diacritics missing** | warn | None of 20+ Yoruba, Igbo or Hausa texts has tone marks, underdots or hooked letters, so they were probably stripped | Restore the source text |
| **Class imbalance / single class** (classification) | warn | Accuracy can look great for a model that always guesses the majority label | Use macro-F1, or rebalance |
| **Surrounding whitespace**, **empty input** | note / warn | Cosmetic, or an example with no real prompt | Tidy |

!!! info "What it does not do"
    It reports the **declared** `lang` field. It does **not** detect languages: there is no reliable off-the-shelf language identification for Hausa, Yoruba and Igbo, and a wrong guess would be worse than none. The diacritics check is a heuristic, so it only warns.

## Checking for leakage

Before evaluating or training, check that your test set shares nothing with your training set:

```bash
atlasforge dataset validate test.jsonl --against train.jsonl
```

It compares inputs ignoring case, punctuation and **tone marks**, so a near-copy that differs only by a tone mark is still caught. It reports how many are identical and how many differ only trivially.

!!! tip "`finetune` runs this for you"
    If you give `finetune` an `eval_file`, it refuses to train when the training data leaks into it.

## Using it in CI

```bash
atlasforge dataset validate data.jsonl --json > validation.json
```

The JSON includes every issue with its `severity`, `code`, `line` and `hint`. The exit code is 0 when there are no errors, so it works as a quality gate.

## Speech datasets

```bash
atlasforge dataset validate speech.jsonl --task asr
```

It checks that each `audio` file exists and that every row has a `reference`. Leakage checking is not implemented for speech datasets, and says so.
