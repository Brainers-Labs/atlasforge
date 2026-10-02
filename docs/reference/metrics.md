# Metrics

All metrics compare a model's **normalised** answer with the **normalised** reference (see [Normalisation](normalization.md)). Text metrics are computed under both the tone-aware and tone-insensitive views, giving keys such as `chrf@tone_aware` and `chrf@tone_insensitive`.

## At a glance

| Metric | Key | Scale | Better | Per-example | Pooled | Intended for |
|---|---|---|---|---|---|---|
| Exact match | `exact_match` | 0 or 1 | higher | yes | no | generation |
| chrF | `chrf` | 0-100 | higher | yes | yes | generation |
| chrF++ | `chrf++` | 0-100 | higher | yes | yes | generation |
| Word error rate | `wer` | fraction, can exceed 1 | **lower** | yes | yes | ASR |
| Character error rate | `cer` | fraction, can exceed 1 | **lower** | yes | yes | ASR |
| Accuracy | `accuracy` | 0 or 1 | higher | yes | no | classification |
| Macro-F1 | `macro_f1` | 0-1 | higher | **no** | yes | classification |

- **Mean** in a report averages the per-example values. **Pooled** is computed over the whole corpus. They can differ, and for WER the pooled figure (total errors over total reference words) is the standard one to quote.
- `accuracy` and `macro_f1` require a `classification` dataset; asking for them on another task is a configuration error.
- Per-example values drive the paired statistics in [`compare`](../guides/compare.md). A metric without them (macro-F1) is reported as `not tested (pooled metric)`.

## Definitions

### Exact match

`1.0` if the normalised prediction equals the normalised reference exactly, else `0.0`.

### chrF and chrF++

Character n-gram F-score as implemented in [sacreBLEU](https://github.com/mjpost/sacrebleu), on a **0-100** scale. chrF++ adds word n-grams (word order 2). Chosen because it is far more forgiving than BLEU for morphologically rich and tonally marked text. Edge cases: two empty strings score 100; one empty string scores 0.

The **pooled** chrF is sacreBLEU's corpus score, which aggregates n-gram statistics across all examples; it differs slightly from the mean of sentence scores.

### WER and CER

Word and character error rates from [jiwer](https://github.com/jitsi/jiwer): `(substitutions + deletions + insertions) / reference length`. Values can **exceed 1** when the model inserts more than the reference contains. Lower is better.

- An empty reference scores `0` if the prediction is also empty, else `1`.
- The **pooled** WER and CER exclude pairs with an empty reference, and are `None` if none remain.

### Accuracy (classification)

The model's free-text answer is mapped to a label: AtlasForge finds the label that appears **earliest** in the normalised answer as a whole word or phrase (the longest wins on a tie). The example is correct if that label equals the reference label; if no label is found, it is wrong. The set of possible labels is the set of references in the dataset.

!!! warning "Negation is not understood"
    "Not positive" would be read as `positive`. Ask the model for the label only.

### Macro-F1

The unweighted mean of per-class F1 over the classes present in the references, where F1 is `2·TP / (2·TP + FP + FN)`. A prediction with no label found counts as a miss for its true class and a false positive for none. It treats rare classes as equally important as common ones, so prefer it to accuracy on imbalanced data.

## Latency

Reports also include latency statistics over **successful calls only**: mean, median (p50), 95th percentile (p95) and max, in milliseconds. Percentiles use linear interpolation.

## How failures are scored

A failed or missing prediction is scored as an **empty answer**:

| Metric | Value of an empty answer |
|---|---|
| exact match, accuracy | wrong (0) |
| chrF | 0 |
| WER, CER | all words deleted (1.0) |
| macro-F1 | a miss for its class |

This is intentional: dropping failed examples would let an unreliable server look better than a reliable one. The report states `n_ok`, `n_failed` and `n_missing`.

## Python

```python
from atlasforge.eval import metrics as m

m.exact_match("abuja", "abuja")        # 1.0
m.chrf("good morning", "good evening")  # 0-100
m.wer("a b c", "a x c")                 # 0.333...
m.macro_f1(["a", "b", None], ["a", "b", "b"])
```

Metric functions expect **already normalised** text; use `atlasforge.eval.normalize.normalize` first, or go through `score_run`, which does it for you. The full list is in the [API reference](python-api.md#atlasforge.eval.metrics).
