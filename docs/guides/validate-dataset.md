# Validate a dataset

Bad data quietly ruins both evaluation and fine-tuning. A duplicated example inflates a score; a test example that also appears in training measures memorisation; stripped diacritics change the meaning of Yoruba and Igbo text. `atlasforge dataset validate` finds these problems **before** you spend compute on them.

```bash
atlasforge dataset validate data.jsonl --task generation
atlasforge dataset validate test.jsonl --task generation --against train.jsonl
```

Unlike `eval`, which stops at the first bad line, `validate` collects **everything** in one pass.

## Options

| Option | Default | Meaning |
|---|---|---|
| `file` (argument) | required | the JSONL dataset |
| `--task`, `-t` | `generation` | `generation`, `classification` or `asr` |
| `--against` | none | another split to check for train/test **leakage** (not available for ASR) |
| `--json` | off | print the full report as JSON |

**Exit code:** `0` if there are no *errors* (warnings and notes never fail a dataset), `1` if there are errors.

## Example

*(A deliberately broken file.)*

```text
ERROR line 2: duplicate id 'a' (first seen on line 1)
      Give each example a unique 'id'.
ERROR line 3: unknown key(s): refrence
      Allowed keys: audio, id, input, lang, messages, meta, reference. Put extra columns under 'meta'.
ERROR line 4: invalid JSON (Expecting value)
FAILED: 1 valid example(s) from 4 line(s), 3 error(s), 0 warning(s).
```

Findings are grouped by severity (errors, then warnings, then notes) and each carries a hint for the fix.

## What it checks

### Errors (the dataset cannot be used as is)

| Code | Meaning |
|---|---|
| `line-error` | a line is malformed: invalid JSON, not an object, an unknown key (typos like `refrence` are caught), a missing or wrong-typed field, a duplicate `id`, a missing audio file. Up to 50 are listed, then a count of the rest |
| `no-examples` | no valid example remains |
| `empty-reference` | a reference is empty after normalisation, in a `classification` or `asr` dataset (it cannot be scored). In `generation` it is a warning |
| `train-test-leakage` | with `--against`: examples whose input also appears in the other split, counting how many are identical and how many differ only in case, punctuation or tone marks |

### Warnings (probably a problem, your call)

| Code | Meaning | Why it matters |
|---|---|---|
| `replacement-character` | text contains `U+FFFD` | encoding damage; the original characters are lost, re-export as UTF-8 |
| `control-characters` | text contains control characters (other than newline, carriage return, tab) | usually a paste or export accident |
| `non-nfc` | text is not in Unicode NFC form | scoring normalises anyway, but training data should be consistent |
| `empty-input` | an input is empty after normalisation | nothing for the model to answer |
| `duplicate-content` | same input **and** reference as an earlier example | inflates scores; can leak between splits |
| `conflicting-references` | the same input has different references | the same question has more than one "correct" answer |
| `diacritics-missing` | a language's expected diacritics look stripped (see below) | stripped diacritics change meaning and hurt training and scoring |
| `single-class` | classification data with only one label | a model that always says that label scores perfectly |
| `class-imbalance` | one class is over 80% of the data, or the smallest is under 10% | prefer macro-F1 over accuracy, or rebalance |
| `against-parse-errors` | lines in the `--against` file could not be parsed | they were skipped in the leakage check |

### Notes

| Code | Meaning |
|---|---|
| `surrounding-whitespace` | leading or trailing whitespace in some texts |
| `leakage-skipped` | `--against` was given for an ASR dataset; leakage checking is not implemented for audio |

Duplicate and leakage checks compare **tone-insensitive, normalised** text, so near-copies that differ only in case, punctuation or tone marks are caught.

## The diacritics check

For Hausa, Yoruba and Igbo, `validate` looks at how often characteristic marks appear, per declared language, and warns when a whole language shows **none**:

| Language | Expected | Warns when none of the texts contain |
|---|---|---|
| `yo` | tone marks, underdotted letters | tone marks (or underdots) |
| `ig` | underdotted letters | underdots |
| `ha` | hooked letters (`ɓ ɗ ƙ ƴ`) | hooked letters |

It only judges a language once there are at least **20** texts for it, and these are warnings, not errors, because it is a heuristic: a dataset of short ASCII proper nouns could legitimately trip it.

!!! note "It does not identify languages"
    There is no reliable off-the-shelf language identifier for Hausa, Yoruba and Igbo, so `validate` reports the language you *declared* in each example's `lang` field plus the diacritic statistics, and does not guess. It will not tell you that a line labelled `yo` is actually Hausa.

## Statistics in the JSON report

`--json` prints a report with `issues` and a `stats` object:

```json
{
  "path": "data.jsonl", "task": "generation", "sha256": "…",
  "n_lines": 200, "n_examples": 200,
  "issues": [ {"severity": "warning", "code": "duplicate-content", "message": "…", "line": null, "count": 3, "hint": "…"} ],
  "stats": {
    "langs": {"ha": 80, "yo": 70, "ig": 50},
    "diacritics": {"yo": {"texts": 140, "tone_rate": 0.62, "underdot_rate": 0.41, "hooked_rate": 0.0}},
    "input_words": {"min": 3, "median": 11.0, "max": 64},
    "reference_words": {"min": 1, "median": 9.0, "max": 70},
    "labels": {"positive": 120, "negative": 80}
  }
}
```

(`labels` appears for classification only; the numbers above are illustrative.)

## Recommended use

1. **Before evaluating**: validate the test set.
2. **Before fine-tuning**: validate the training set, and validate the test set `--against` the training set to prove they do not overlap.
3. **In CI**: `atlasforge dataset validate data.jsonl --json` and fail on exit code `1`.

Next: [Evaluate a model](evaluate.md).
