# Tone-aware scoring

Yoruba, Igbo and Hausa are written with marks that change what a word means, and that models and people routinely get wrong or leave out. A scorer that treats text as plain ASCII cannot tell a model that is wrong from one that merely omits tone marks. AtlasForge therefore reports every text metric **twice**.

## The two views

| View | Tone marks | Use it to answer |
|---|---|---|
| **Tone-aware** | Kept | "Did the model write the word exactly right, tones included?" |
| **Tone-insensitive** | Ignored | "Did the model get the right word, regardless of tone marks?" |

Neither is hidden or preferred. When they disagree, that is information: the model is getting the words right but the tones wrong, which is a different problem from getting the words wrong.

## What normalisation does

Before comparing, both the model's answer and the reference are normalised the same way:

1. Lower-cased.
2. Converted to Unicode **NFC**, so a letter written as one character and the same letter written as base-plus-combining-marks compare equal. Without this, identical Yoruba text can score as different.
3. *Tone-insensitive view only:* tone marks (grave, acute, circumflex, macron, caron) are removed.
4. Typographic apostrophes are unified, punctuation becomes a space, and whitespace is collapsed. Hausa keeps a leading apostrophe because it is part of the word (for example `'yar`); other languages drop stray quote marks.

!!! important "What is never removed"
    Only **tone marks** are stripped. Letters that merely look accented are real letters and are always kept: the dot-below in Yoruba and Igbo (`ẹ`, `ọ`, `ṣ`, `ị`, `ụ`), the dot-above `ṅ` in Igbo, and Hausa's hooked letters (`ɓ`, `ɗ`, `ƙ`, `ƴ`). Removing them would change the word.

## Real examples

These rows are produced by the normaliser itself while this page is built:

{{ normalize_demo() }}

## Reading disagreements between the views

| Pattern | Likely meaning |
|---|---|
| Tone-aware low, tone-insensitive high | Words are right, tone marks are missing or wrong. Common with models trained on text that dropped diacritics. |
| Both low | The words themselves are wrong. |
| Both high | Right words and right tones. |
| Fine-tune improves *aware* but not *insensitive* | The gain is mostly in tone marks. Real, but not the same as understanding the task better. |

The [quickstart](../get-started/quickstart.md) shows exactly the last pattern.

## Where it applies

- **Text metrics** (exact match, accuracy, chrF, WER, CER): both views, always.
- **Speech (WER, CER):** both views. If a speech model omits tone marks, the gap between the views shows it, and the [error analysis](../guides/transcribe-speech.md#what-it-heard-instead) counts exactly how many substitutions were tone-only.
- **Dataset checks:** duplicates and train/test leakage are detected on the *tone-insensitive* form, so a near-copy that differs only in tone marks is still caught.
- **Merging long transcripts:** the overlap between chunks is matched ignoring tone marks.

## Limits

Normalisation is rule-based and conservative. It does not understand spelling variants, dialect differences or code-switching, and it makes no claim to. The two views are fixed for now: a custom normalisation is not configurable. Every raw answer is kept in `results.jsonl` (see [Run directories](run-directories.md)), so you can always score them yourself.
