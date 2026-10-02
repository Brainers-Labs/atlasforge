# Metrics

Select metrics with `--metric` / `-m` (repeatable). Every text metric is computed under **both** [tone views](../concepts/tone-aware-scoring.md) and reported as `name@tone_aware` and `name@tone_insensitive`.

{{ metrics_table() }}

## Scales and direction

- Fractions are shown as percentages (`62.0%`) and their differences as **points** (`+17.4 pts`).
- chrF is already on a 0-100 scale and is shown as is.
- For `wer` and `cer`, **lower is better**, and AtlasForge flips the verdict accordingly: a drop in WER is an `improved`.
- WER and CER can exceed 100% when the model inserts more words than the reference has.

## Empty answers and references

- A failed or missing prediction is scored as an **empty answer**.
- If a reference is empty after normalisation, WER and CER for that example are 0 when the answer is also empty and 1 otherwise. Such examples are excluded from the pooled WER and CER. `dataset validate` flags them.

## Notes on specific metrics

**`exact_match`.** Compares normalised text, so case, punctuation and (in the insensitive view) tone marks do not matter.

**`accuracy` and `macro_f1`.** The label set is the distinct references. The answer is matched to the label it names: the earliest whole-word occurrence, the longest label winning ties. If no label is found the answer counts as wrong. Negation is not understood.

**`chrf`, `chrf++`.** Computed with sacrebleu. Both sides empty scores 100; one side empty scores 0. chrF measures overlap, not meaning, so a fluent wrong answer can score well.

**`wer`, `cer`.** Computed with jiwer. *Mean* is the average of per-utterance rates; *pooled* divides total errors by total reference words (or characters), the standard speech-recognition figure.

**`macro_f1`.** Pooled only. It has no per-example value, so `compare` reports the two pooled numbers but no interval or verdict.

## Latency

Reports also include latency (mean, p50, p95, max) computed from successful calls only. It is informational: no verdict is attached to it.
