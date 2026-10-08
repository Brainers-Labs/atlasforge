# Compare two models

Find out whether a change helped, how sure you can be, and where it got worse.

**You need:** two finished [run directories](../concepts/run-directories.md) made from **the same dataset file**: for example a base model and a fine-tuned one. Try it now on the [offline demo](../get-started/quickstart.md#5-compare-the-two-models).

## 1. Evaluate both models on the same dataset

```bash
atlasforge eval data.jsonl --out runs/base  --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
atlasforge eval data.jsonl --out runs/tuned --base-url http://127.0.0.1:8001/v1 --model my-finetune
```

Use identical settings (temperature, token limit) for both, so the only difference is the model.

## 2. Compare

```bash
atlasforge compare data.jsonl \
  --base runs/base \
  --candidate runs/tuned \
  --slice domain \
  --out comparison
```

`compare` needs no model: it reads the two saved runs. It writes `comparison.md` to read, `comparison.json` for a script, and `comparison.html` — the same page treatment as a run's report, with each metric's change drawn against its confidence interval — then prints a summary.

!!! failure "It refuses runs from different datasets"
    If either run was made from a different dataset file (even a reordered or edited copy), `compare` stops with *"was not produced from this dataset"*. Pairing examples from different data is meaningless.

## 3. Read it

Work through the output in this order.

1. **The table.** Look at the *Verdict* column, not the delta alone. `improved` or `regressed` means the confidence interval excludes zero. `no clear change` means the data cannot tell the models apart.
2. **Both tone views.** If *aware* and *insens.* disagree, the difference is partly about tone marks. See [Tone-aware scoring](../concepts/tone-aware-scoring.md).
3. **The Regressions list.** This is the point of the tool. An overall gain can hide a slice that got worse.
4. **"Insufficient data".** Slices under 30 examples are shown but not judged.
5. **The failure-mode flags.** How often each deterministic rule fired in each run, and the difference — 8 more number mismatches is a different story from 8 more empty answers. See [Failure-mode flags](../concepts/failure-modes.md).

The [statistics page](../concepts/statistics.md) explains the intervals, the significance test and the verdict rule in full.

## Slicing

Built-in slices are free. To slice by your own field, put it under `meta` in the dataset and pass `--slice`:

```bash
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned \
  --slice domain --slice source
```

By default the slices are computed for the first metric under the *tone-aware* view. Choose another with `--primary`:

```bash
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --primary chrf@tone_aware
```

Change the minimum slice size with `--min-slice-n` (the default, 30, is a sensible floor; lowering it makes noise look like findings).

## Reproducibility

The intervals use 1,000 bootstrap resamples and seed 0 by default. The same inputs always give the same output. Change them with `--n-boot` and `--seed`; the report records the values used.

## The same thing in Python

This runs offline against the demo data (create it first with `atlasforge demo`):

```python
# docs:run
from atlasforge.compare import compare_runs, to_markdown
from atlasforge.eval.dataset import load_dataset

dataset = load_dataset("atlasforge-demo/toy_qa.jsonl", "generation")
result = compare_runs(
    dataset,
    "atlasforge-demo/runs/base",
    "atlasforge-demo/runs/tuned",
    slice_fields=["domain"],
)
print([s.value for s in result.regressed_slices])
print(to_markdown(result)[:60])
```

## Use it in a submission

`comparison.md` is written to be pasted into a report, a README or a competition submission. It states the dataset fingerprint, the two models, the method, and its own limits. `comparison.html` is the same report as a page to open or send. For a published adapter, feed `comparison.json` to the [model card](publish-a-model-card.md).
