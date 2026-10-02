# Statistics

`compare` does not just subtract two averages. This page explains what it computes and, just as important, what it does **not** claim.

## Paired, not independent

Both models answered the **same examples**, so the comparison is *paired*: for every example it looks at the candidate's score minus the base's score. This cancels out how hard each example is. Without pairing, a lucky draw of easy questions could look like an improvement.

## The confidence interval

For each metric, AtlasForge computes the mean per-example difference (candidate minus base) and a **95% confidence interval** by a *paired bootstrap*:

1. Resample the examples with replacement, as many as the dataset has.
2. Compute the mean difference on that resample.
3. Repeat 1,000 times (`--n-boot`).
4. Take the 2.5th and 97.5th percentiles of those means.

The interval is the range the true difference plausibly lies in. It is **deterministic**: the same seed (`--seed`, default 0) always gives the same interval, so results are reproducible.

| If the interval is... | The verdict is |
|---|---|
| Entirely above zero | `improved` (or `regressed` for error rates, where lower is better) |
| Entirely below zero | `regressed` (or `improved` for error rates) |
| Spanning zero | `no clear change` |

Notice what this means: **a positive average is not enough.** If the interval touches zero, AtlasForge says `no clear change`, even when the average went up.

## The exact significance test

When a metric is right-or-wrong per example (such as exact match), `compare` also runs an **exact McNemar test** on the examples where the two models disagree. It reports a *p-value*: how surprising the pattern of disagreements would be if the models were equally good. A small p-value (conventionally below 0.05) is evidence the difference is real. It is shown in the Markdown report next to the verdict.

## Win / tie / loss

The report also counts examples where the candidate did better, the same, or worse. This is the plain-language version of the same question.

## Slices

A *slice* is a subset of the dataset: one language, short answers, questions containing a number, or any value of a `meta` field you chose. Slices are how a regression hides: the overall number improves while one slice gets worse.

Each slice gets its own interval and verdict, with one rule:

!!! warning "Fewer than 30 examples is not judged"
    A slice smaller than `--min-slice-n` (default **30**) is reported as `insufficient data` and gets no verdict and no improvement or regression claim. A result from a dozen examples is noise, and presenting it as a finding would be misleading. The numbers are still shown so you can see them.

Built-in slices are `lang`, `length` (by the reference's word count: short is 5 words or fewer, medium is 6-15, long is over 15) and `has_number` (the reference contains a digit). Add your own with `--slice <meta-key>`.

!!! note "Many slices means some will look significant by chance"
    Slice intervals are **not adjusted for multiple comparisons**. If you look at 20 slices, about one will cross the line by luck. Treat a single flagged slice as a lead to investigate, not a proven effect. The report says this too.

## Failures count as wrong

If a call failed, or a result is missing, that example is scored as an **empty answer**: wrong for accuracy, all words deleted for WER, zero for chrF. This applies to both models. It is stricter than dropping failures, and deliberately so: a model that crashes on hard questions must not look better than one that answers them.

## Mean versus pooled

Some metrics have two numbers:

| | Meaning |
|---|---|
| **Mean** | The average of per-example scores. This is what the interval and the verdict are about. |
| **Pooled** | Computed over the whole corpus at once (WER, CER, chrF, macro-F1). WER is *total word errors divided by total reference words*, which is the standard speech-recognition figure. |

The two can differ, especially when example lengths vary a lot. Macro-F1 has only a pooled value, so it has no interval.

## What this does not claim

- **Not causation.** It shows the candidate scored differently on this dataset. It does not show why.
- **Not generalisation.** Results describe *your* dataset. A different dataset may behave differently.
- **Not a substitute for a good dataset.** A small or unrepresentative dataset gives wide intervals or misleading ones. Use [`dataset validate`](../guides/validate-your-data.md) and a held-out test set.
