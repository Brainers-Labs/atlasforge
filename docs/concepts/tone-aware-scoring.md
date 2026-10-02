# Tone-aware scoring

Hausa, Yoruba and Igbo are written with marks and letters that English tooling routinely mangles. If a scoring tool handles them carelessly, its numbers are wrong in ways that are invisible. AtlasForge makes the handling explicit and reports **both** sensible answers.

## The problem

The same Yoruba word can arrive in several forms:

- with tone marks (`kọ́pa`) or without (`kopa`),
- as one precomposed character or as a base letter plus combining marks (different Unicode, same text on screen),
- with a curly apostrophe or a straight one,
- in different capitalisation or with different punctuation.

A plain `==` treats all of these as different answers. A model that outputs a correct answer in a different Unicode form gets scored as wrong.

At the same time, **over**-normalising is worse. Removing every accent would also delete letters that are part of the alphabet, turning distinct words into the same word.

## Two views, always both

Every text metric in AtlasForge is computed twice.

| View | Tone marks | Use it to answer |
|---|---|---|
| **tone-aware** | kept | "Did the model get the tones right?" |
| **tone-insensitive** | removed | "Did the model get the words right, ignoring tone-marking habits?" |

Neither is silently preferred. The report shows both side by side, and metric keys carry the view: `chrf@tone_aware`, `chrf@tone_insensitive`. Wherever a single number is needed (for example choosing the metric for slice analysis), AtlasForge defaults to the tone-aware one and says so.

A large gap between the two views is itself informative: it means the model mostly produces the right words but is inconsistent with tones.

## What is removed, and what is never removed

**Tone marks removed in the tone-insensitive view:** grave (`̀`), acute (`́`), circumflex (`̂`), macron (`̄`) and caron (`̌`).

**Always kept, in both views, because they are letters and not tones:**

| Letters | Language | Why they stay |
|---|---|---|
| `ẹ`, `ọ`, `ṣ` (dot below) | Yoruba, Igbo | distinct letters of the alphabet |
| `ị`, `ọ`, `ụ` (dot below) | Igbo | distinct vowels |
| `ṅ` (dot above) | Igbo | a distinct consonant |
| `ɓ`, `ɗ`, `ƙ`, `ƴ` (hooked) | Hausa | distinct consonants, not accented forms |

This is enforced by the normaliser: it removes only the five tone combining marks after decomposing the text, then recomposes it. Tests cover the underdot letters, the Igbo dot-above `ṅ`, and the Hausa hooked letters (including their uppercase forms).

## The normalisation pipeline

For each text, in order:

1. **Lowercase** (done before Unicode composition, because lowercasing can itself produce combining sequences).
2. **Unicode form.** Tone-aware: compose to NFC. Tone-insensitive: decompose, drop the tone marks, recompose to NFC.
3. **Unify apostrophes.** Typographic quotes and look-alike characters (curly single quotes, modifier-letter apostrophes, backtick, acute accent) become a straight `'`.
4. **Punctuation and symbols become spaces.** Apostrophes are exempt from this step; they are handled next.
5. **Clean apostrophes.** An apostrophe is meaningful only inside a word. For Hausa, an apostrophe attached to a word (for example the leading one in *'yar*) is part of the word and is kept; only a free-standing apostrophe is removed. For the other languages, any apostrophe that is not between two word characters is treated as a quote mark and removed.
6. **Collapse whitespace.**

The function is **idempotent**: normalising twice gives the same result as once. The exact configuration used for each view is recorded in `report.json` under `normalization`.

!!! example "Example"
    Under the tone-insensitive view, `Kọ́pà àtijọ́!` and `kọpa atijọ` normalise to the same string, while `kopa` (without the underdot) stays different, because the underdot changes the letter.

## Where each language setting matters

The `lang` field of an example selects the language-specific rules (currently only the Hausa apostrophe behaviour). Examples without a `lang` use the language-neutral rules. The language is validated, so a typo like `"yr"` is rejected when the dataset loads.

## Reading the numbers

- If tone-aware and tone-insensitive scores are close, tones are not your problem.
- If the tone-insensitive score is much higher, the model gets the words right but the tone marks wrong or inconsistent. For speech output this is common and often acceptable; for written output you may care.
- If **both** are low, the model is getting the words wrong, and normalisation cannot help.

The [dataset validator](../guides/validate-dataset.md) complements this: it warns when a dataset's diacritics look stripped (for example, no Yoruba text containing any tone mark), which would make the tone-aware view meaningless.

See the [normalisation reference](../reference/normalization.md) for the Python API.
