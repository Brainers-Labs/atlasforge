# Normalisation

Normalisation is applied to both the model's answer and the reference before any text metric compares them. The *why* is in [Tone-aware scoring](../concepts/tone-aware-scoring.md); this page is the precise specification.

## Pipeline

`normalize(text, config)` applies these steps in order:

| # | Step | Detail |
|---|---|---|
| 1 | Lowercase | on by default; done **before** Unicode composition, because lowercasing can itself create combining sequences |
| 2 | Unicode form | `tones="keep"`: compose to **NFC**. `tones="strip"`: decompose to NFD, remove the tone marks, recompose to NFC |
| 3 | Unify apostrophes | typographic quotes and look-alikes become a straight `'` |
| 4 | Punctuation to space | every Unicode punctuation (`P*`) and symbol (`S*`) character becomes a space; apostrophes are exempt here |
| 5 | Clean apostrophes | free-standing and quote-mark apostrophes are removed; see below |
| 6 | Collapse whitespace | runs of whitespace become a single space; ends are trimmed |

Steps 3 to 5 run only when `punctuation="strip"` (the default).

### Tone marks

Removed when `tones="strip"`: combining grave (U+0300), acute (U+0301), circumflex (U+0302), macron (U+0304) and caron (U+030C).

**Never removed:** the underdot (U+0323) on `ẹ ọ ṣ ị ụ`, the dot above on Igbo `ṅ`, and Hausa hooked letters `ɓ ɗ ƙ ƴ`. These are letters, not tones.

### Apostrophes

| Language setting | Rule |
|---|---|
| `lang="ha"` | only an apostrophe with **no** letter or digit on either side is removed; apostrophes attached to a word (such as the leading one in *'yar*) are kept |
| any other (or none) | an apostrophe not between two word characters is removed |

An apostrophe **inside** a word (as in *n'ụzọ*) is always kept.

## Configuration

```python
from atlasforge.eval.normalize import NormalizeConfig, normalize

config = NormalizeConfig(lang="yo", tones="strip", lowercase=True, punctuation="strip")
normalize("Kọ́pà àtijọ́!", config)   # 'kọpa atijọ'
```

| Field | Values | Default | Meaning |
|---|---|---|---|
| `lang` | `ha`, `yo`, `ig`, `en` or `None` | `None` | selects language-specific rules (currently only the Hausa apostrophe behaviour). Validated |
| `tones` | `keep`, `strip` | `keep` | whether tone marks are kept |
| `lowercase` | bool | `True` | lowercase the text |
| `punctuation` | `strip`, `keep` | `strip` | remove punctuation and symbols, or leave them |

Invalid values raise `ConfigError` immediately, so a mistyped setting cannot silently change your results.

Two helpers build the standard views:

```python
from atlasforge.eval.normalize import tone_aware, tone_insensitive

tone_aware("yo")        # NormalizeConfig(lang="yo", tones="keep")
tone_insensitive("yo")  # NormalizeConfig(lang="yo", tones="strip")
```

`strip_tones(text)` is also exposed on its own: it removes only tone marks and returns NFC text.

## Properties

- **Idempotent:** `normalize(normalize(x)) == normalize(x)`, tested across tone modes and languages.
- **Equivalence of Unicode forms:** a precomposed character and its base-plus-combining form normalise to the same string.
- **Recorded:** the exact settings of each view are written to `report.json` under `normalization`:

    ```json
    "normalization": {
      "tone_aware":       {"tones": "keep",  "lowercase": true, "punctuation": "strip"},
      "tone_insensitive": {"tones": "strip", "lowercase": true, "punctuation": "strip"}
    }
    ```

## Where it is used

| Place | Why |
|---|---|
| scoring (`score_run`) | normalises prediction and reference under each view |
| `dataset validate` | duplicates, conflicts and leakage are detected on tone-insensitive normalised text |
| long-audio merging | seam words are matched tone-insensitively, because a word is often transcribed with different tone marks on each side of a window boundary |

## What it does not do

- No spelling correction, stemming or number normalisation. `5` and `five` are different.
- No language identification.
- No conversion between orthographies; text is compared as written.
