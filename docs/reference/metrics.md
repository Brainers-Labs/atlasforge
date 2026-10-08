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

**`accuracy_strict` and `macro_f1_strict`.** The same two figures with a stricter match: the **whole** answer must be a label. An answer that merely mentions one is no longer a hit, so `"the sentiment is not positive"` finds nothing instead of `positive`. Use it when the prompt asks for the label and nothing else and the model has taken to editorialising — the loose metric can score a model well for naming a label it then contradicts, and the strict one cannot. Keep the loose one beside it: the **gap between the two is the measure of how often the model answers more than it was asked**, which is a different question from whether it got the label right.

```console
$ atlasforge report runs/base -m accuracy -m accuracy_strict
```

Both are classification-only and are scored on the same normalised text as everything else, so case, punctuation and tone marks are handled the same way.

**`chrf`, `chrf++`.** Computed with sacrebleu. Both sides empty scores 100; one side empty scores 0. chrF measures overlap, not meaning, so a fluent wrong answer can score well.

**`wer`, `cer`.** Computed with jiwer. *Mean* is the average of per-utterance rates; *pooled* divides total errors by total reference words (or characters), the standard speech-recognition figure.

**`macro_f1`.** Pooled only. It has no per-example value, so `compare` reports the two pooled numbers but no interval or verdict.

## Your own metrics

A metric is any callable that takes the answer, the reference and the whole example, and
returns a number:

```python
def mentions_dosage(prediction, reference, example):
    return float("mg" in prediction and "mg" in reference)
```

Pass it in `metrics=[...]` from Python, or name it `module:function` on the command line
(`--metric my_metrics:mentions_dosage`, where `my_metrics.py` is importable from where you
run AtlasForge). This works everywhere a built-in name does — `atlasforge eval`,
`atlasforge report`, `atlasforge compare` — and it runs offline, since only scoring changes:

```python
# docs:run
import atlasforge


def mentions_dosage(prediction, reference, example):
    """Did the answer mention mg, and did the reference?"""
    return float("mg" in prediction and "mg" in reference)


report = atlasforge.score_finished_run(
    "atlasforge-demo/runs/tuned",
    "atlasforge-demo/toy_qa.jsonl",
    metrics=["exact_match", mentions_dosage],
)
print(report.metric("mentions_dosage@tone_aware").mean)
```

Two things to know:

- **Both strings arrive normalised for the view being scored** — tone-aware or tone-insensitive,
  lower-cased, punctuation stripped — exactly like every built-in metric. Use
  `example.reference` if you need the original text, and `example.lang` if your metric is
  language-specific.
- **Say the direction if lower is better.** Set `higher_is_better = False` on the function and
  the report and the comparison verdict read the right way round; the default is higher-is-better.

A custom metric is reported under both views like any other, with a mean over examples. It has
no pooled figure — no corpus-wide number is computed for it — but it has per-example values, so
`compare` runs the paired statistics on it normally. A custom metric may not reuse a built-in
name, and a metric that raises fails the run rather than being swallowed.

## Latency

Reports also include latency (mean, p50, p95, max) computed from successful calls only. It is informational: no verdict is attached to it.
