# Honest statistics

A comparison that says "model B scored 3 points higher" is not yet an answer. On a small dataset a 3-point gap is often luck. `atlasforge compare` is built to tell you when a difference is real enough to act on, and to say "we cannot tell" when it is not.

## The questions it answers

1. **Overall:** is the candidate better, worse or indistinguishable from the base, on each metric?
2. **Where:** in which parts of your data did it improve, and where did it **regress**?

## Paired comparison

Both models answer the **same** examples. AtlasForge therefore compares each example with itself: the unit of analysis is the per-example difference `candidate − base`. This cancels out how hard each example is, which makes the comparison far more sensitive than comparing two averages.

For that to be valid, both runs must come from exactly the same dataset file and task. `compare` checks the dataset hash recorded in each run's manifest and refuses otherwise.

## Confidence intervals: the paired bootstrap

For each metric, AtlasForge resamples your examples with replacement (the *paired bootstrap*), computes the mean per-example difference for each resample, and reports the 2.5th to 97.5th percentile of those means as a **95% confidence interval**.

| Setting | Default | Option |
|---|---|---|
| Resamples | 1000 | `--n-boot` |
| Random seed | 0 | `--seed` |
| Confidence | 95% | fixed |

The result is **deterministic**: the same seed gives the same interval, so reports are reproducible. If the two runs give identical scores, the interval has zero width.

## The verdict rule

| Verdict | When |
|---|---|
| **improved** | the whole interval is on the good side of zero |
| **regressed** | the whole interval is on the bad side of zero |
| **no clear change** | the interval includes zero |

"Good side" respects the metric's direction. For accuracy, exact match and chrF, higher is better. For **WER and CER, lower is better**, so a negative delta is an improvement and AtlasForge flips the verdict accordingly.

!!! example "Reading an interval"
    `+50.0 pts  [+35.0, +65.0]  improved` means the mean gain was 50 points and, allowing for sampling luck, the true gain is plausibly anywhere from 35 to 65. Because the interval is entirely above zero, it is called an improvement. `+3.0 pts  [-4.0, +10.0]  no clear change` means the data cannot distinguish a 3-point gain from no gain, or even from a loss.

## Win / tie / loss counts

Next to each verdict the report shows `W / T / L`: on how many examples the candidate was better, equal or worse (oriented so that "win" always means better, including for WER). It is a quick sanity check: a big average gain from a handful of wins and many losses deserves suspicion.

## McNemar's exact test for right-or-wrong metrics

For metrics that are 0 or 1 per example (exact match, accuracy), AtlasForge also reports an **exact two-sided McNemar test**. It looks only at the *discordant* examples (base right and candidate wrong, or the reverse) and asks whether the split between the two kinds is lopsided beyond chance. The reported p-value is exact, not an approximation, so it stays valid for small counts. When no examples are discordant, the p-value is `1.0`.

The bootstrap interval and the McNemar test usually agree. They answer slightly different questions, so both are shown.

## Pooled metrics

Some figures are computed over the whole corpus rather than per example: pooled **WER and CER** (total errors divided by total reference words or characters, the standard ASR figure), corpus **chrF**, and **macro-F1**. These are shown separately as *pooled figures*. Macro-F1 has no per-example value, so it gets no interval and is marked `not tested (pooled metric)`. Use it for the headline number and use the per-example metrics for the statistics.

## Slices: where did it get worse?

An overall gain can hide a regression in one part of your data. With `--slice` you break a chosen metric down by:

| Slice field | Buckets |
|---|---|
| `lang` | the example's declared language, or `(none)` |
| `length` | `short (<=5 words)`, `medium (6-15 words)`, `long (>15 words)`, by **reference** word count |
| `has_number` | `yes` / `no`: whether the reference contains a digit |
| *any `meta` key* | the value of that key, e.g. `--slice domain`; `(missing)` if absent |

`lang`, `length` and `has_number` are always included. Each slice gets its own paired bootstrap interval and verdict, and slices are listed **worst first**.

### The 30-example rule

A slice with fewer than **30** examples (`--min-slice-n`) is reported as *insufficient data*. It gets no delta, no interval and no verdict, so it can never be presented as an improvement or a regression. A headline claim built on a dozen examples is noise.

### Slices are not corrected for multiple comparisons

If you slice many ways, some slice will look different by chance. The report says so. Treat a single flagged slice as **a lead to investigate, not a proven effect**.

The metric used for slices is `--primary`, which defaults to the first tone-aware metric. Choose another with, for example, `--primary chrf@tone_aware`.

## Failures are part of the comparison

A failed or missing prediction is scored as an empty answer, for both runs. The report header shows each run's failed and missing counts, so a candidate that looks better because the base had a flaky server is visible, not hidden.

## What this does not tell you

The report's own method note says it plainly: *this shows the candidate scored differently on this dataset. It does not show why.*

- It cannot tell you whether your **dataset represents your real task**. A tidy result on a narrow dataset is a tidy result on a narrow dataset.
- It does not adjust for **comparing many metrics**; with several metrics, one may cross zero by chance.
- With **few examples**, intervals are wide and most verdicts will be "no clear change". That is the tool being honest, not failing. Add data.
- It does not measure fluency, factuality or safety. It measures agreement with the references you supplied.

## Sample size

The interval narrows roughly with the square root of the number of examples, so four times the data halves the width. There is no universal "enough"; a quick way to find out is to run `compare` and look at how wide the intervals are for the differences you care about.
