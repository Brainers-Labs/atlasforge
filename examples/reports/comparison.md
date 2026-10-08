# AtlasForge comparison report

- **Task:** generation
- **Examples:** 92
- **Dataset fingerprint (sha256):** `7803765cfec0fdab`

| Run | Model | Revision | Backend | Failed | Missing | Flagged |
|---|---|---|---|---|---|---|
| Base | synthetic-demo/base | `sample-dat` | synthetic | 2 | 0 | 6 |
| Candidate | synthetic-demo/tuned | `sample-dat` | synthetic | 0 | 0 | 14 |

## Overall

Delta is candidate minus base. The interval covers the mean per-example difference; a change counts as *improved* or *regressed* only if the whole interval is on one side of zero.

| Metric | View | Base | Candidate | Delta | 95% CI | Verdict | W / T / L | McNemar p |
|---|---|---|---|---|---|---|---|---|
| exact_match | tone-aware | 62.0% | 79.3% | +17.4 pts | [+3.3, +31.5] | improved | 31 / 46 / 15 | 0.0259 |
| chrf | tone-aware | 67.2 | 84.5 | +17.3 | [+5.4, +29.5] | improved | 31 / 46 / 15 | - |
| exact_match | tone-insensitive | 75.0% | 81.5% | +6.5 pts | [-5.4, +18.5] | no clear change | 21 / 56 / 15 | 0.405 |
| chrf | tone-insensitive | 77.4 | 85.9 | +8.5 | [-1.8, +19.3] | no clear change | 21 / 56 / 15 | - |

**Pooled figures** (computed over the whole corpus, no interval):

| Metric | View | Base | Candidate | Delta |
|---|---|---|---|---|
| chrf | tone-aware | 54.0 | 90.7 | +36.7 |
| chrf | tone-insensitive | 70.5 | 92.5 | +22.0 |

## Regressions

- **has_number = yes** (n=40): -20.0 pts, CI [-37.6, -2.5].
- **domain = numeracy** (n=40): -20.0 pts, CI [-37.6, -2.5].

## Failure-mode flags

Deterministic rules over the text, counted per answer — the same rules the single-run report lists. They are places to look, not a quality score, and not a hallucination rate.

| Flag | Base | Candidate | Delta | What it means |
|---|---|---|---|---|
| `number_mismatch` | 6 | 14 | +8 | the answer's numbers differ from the reference's |

## Slices (exact_match@tone_aware)

**lang**

| Value | n | Base | Candidate | Delta | 95% CI | Status |
|---|---|---|---|---|---|---|
| en | 80 | 71.2% | 78.8% | +7.5 pts | [-6.2, +22.5] | no clear change |
| yo | 12 | 0.0% | 83.3% | - | - | insufficient data (n < 30) |

**length**

| Value | n | Base | Candidate | Delta | 95% CI | Status |
|---|---|---|---|---|---|---|
| short (<=5 words) | 92 | 62.0% | 79.3% | +17.4 pts | [+3.3, +31.5] | improved |

**has_number**

| Value | n | Base | Candidate | Delta | 95% CI | Status |
|---|---|---|---|---|---|---|
| yes | 40 | 85.0% | 65.0% | -20.0 pts | [-37.6, -2.5] | regressed |
| no | 52 | 44.2% | 90.4% | +46.2 pts | [+28.8, +61.5] | improved |

**domain**

| Value | n | Base | Candidate | Delta | 95% CI | Status |
|---|---|---|---|---|---|---|
| numeracy | 40 | 85.0% | 65.0% | -20.0 pts | [-37.6, -2.5] | regressed |
| agri | 40 | 57.5% | 92.5% | +35.0 pts | [+15.0, +52.5] | improved |
| greetings | 12 | 0.0% | 83.3% | - | - | insufficient data (n < 30) |

## Method

- Paired bootstrap of per-example differences: 1000 resamples, seed 0, 95% percentile interval.
- Binary metrics also get an exact McNemar test on the discordant pairs.
- Failed or missing predictions are scored as wrong (an empty answer), for both runs.
- Every text metric is shown under both views: *tone-aware* keeps tone marks, *tone-insensitive* strips them. Underdots and Hausa hooked letters are always kept.
- Slices with fewer than 30 examples are reported as insufficient data. Slice intervals are not adjusted for multiple comparisons: treat a single flagged slice as a lead to investigate, not a proven effect.
- This shows the candidate scored differently on this dataset. It does not show why.
