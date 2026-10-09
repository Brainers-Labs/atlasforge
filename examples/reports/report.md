# AtlasForge evaluation report

- **Task:** generation
- **Examples:** 92 (90 ok, 2 failed, 0 missing)
- **Dataset fingerprint (sha256):** `7803765cfec0fdab`
- **Model:** synthetic-demo/base (revision `sample-dat`)
- **Backend:** synthetic

## Metrics

| Metric | View | Mean | Pooled | n |
|---|---|---|---|---|
| exact_match | tone-aware | 62.0% | - | 92 |
| chrf | tone-aware | 67.2 | 54.0 | 92 |
| exact_match | tone-insensitive | 75.0% | - | 92 |
| chrf | tone-insensitive | 77.4 | 70.5 | 92 |

## Latency

mean 957 ms | p50 956 ms | p95 1054 ms | max 1063 ms (successful calls only)

## Failure-mode flags

6 of 90 answer(s) raised at least one flag. Each one is a rule you can read and re-check, not a quality score and not a hallucination rate.

| Flag | Answers | What it means |
|---|---|---|
| `number_mismatch` | 6 | the answer's numbers differ from the reference's |

A flag is a place to look, not a verdict: an answer can be flagged and still be right. Per-example flags are in `report.json` under `per_example_flags`.

## Method

- *Mean* averages per-example scores; *pooled* is computed over the whole corpus (WER, CER, chrF, macro-F1).
- Every text metric is shown under both views: *tone-aware* keeps tone marks, *tone-insensitive* strips them. Underdots and Hausa hooked letters are always kept.
- Failed or missing predictions are scored as an empty answer, so they count against the model.
- chrF is on a 0-100 scale. Other metrics are percentages; for WER and CER lower is better.
