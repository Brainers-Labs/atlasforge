# Compare two models

`atlasforge compare` answers the question AtlasForge exists for:

> *Did my change actually make the model better, and where did it get worse?*

It takes two finished evaluation runs on the same dataset and produces a report with confidence intervals, verdicts and a per-slice breakdown. Read [Honest statistics](../concepts/honest-statistics.md) first if you want to understand what the verdicts mean.

## Prerequisites

You need two run directories created by `atlasforge eval`, from the **same dataset file and task**:

```bash
atlasforge eval data.jsonl --out runs/base  --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
atlasforge eval data.jsonl --out runs/tuned --base-url http://127.0.0.1:8001/v1 --model my-finetune
```

"Base" is the reference point (often the official model); "candidate" is the thing you changed (a fine-tune, a quantisation, a different prompt).

!!! note "Compare models and serving setups, not prompt variants"
    `compare` pairs runs by the dataset's content hash, and prompts live inside the dataset (`input` or `messages`). Two prompt wordings are therefore two different datasets and cannot be compared directly in v0.1. What you *can* compare is anything that changes the model or how it is served: a fine-tune against its base, an int4 quantisation against fp16, two checkpoints, or two decoding settings, all on one dataset file.

## Run it

```bash
atlasforge compare data.jsonl \
  --base runs/base --candidate runs/tuned \
  --out cmp --slice domain
```

| Option | Default | Meaning |
|---|---|---|
| `dataset` (argument) | required | the dataset **both** runs used |
| `--base` | required | run directory of the base model |
| `--candidate` | required | run directory of the candidate |
| `--out`, `-o` | `comparison` | where to write the report |
| `--task`, `-t` | `generation` | must match the runs |
| `--metric`, `-m` | task defaults | repeatable; restrict to these metrics |
| `--primary` | first tone-aware metric | the metric used for slices, e.g. `chrf@tone_aware` |
| `--slice` | built-ins | repeatable; add a `meta` key to slice by |
| `--n-boot` | `1000` | bootstrap resamples |
| `--seed` | `0` | bootstrap seed (same seed, same intervals) |
| `--min-slice-n` | `30` | slices smaller than this are "insufficient data" |

`compare` **re-scores both runs itself** from their `results.jsonl`, so you can pass different `--metric` values than you used in `eval`.

If a run directory was produced from a different dataset or task, `compare` stops:

```text
error: runs/tuned was not produced from this dataset and task.
  -> Re-run it on the same dataset file, or pass the dataset it was run on.
```

## Reading the console output

*(Stub-server run with synthetic data, to show the layout.)*

```text
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                ┃ Base ┃ Cand. ┃ Delta     ┃ 95% CI         ┃ Verdict  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ exact_match (aware)   │ 0.0% │ 50.0% │ +50.0 pts │ [+35.0, +65.0] │ improved │
│ chrf (aware)          │ 0.0  │ 50.0  │ +50.0     │ [+35.0, +65.0] │ improved │
│ exact_match (insens.) │ 0.0% │ 50.0% │ +50.0 pts │ [+35.0, +65.0] │ improved │
│ chrf (insens.)        │ 0.0  │ 50.0  │ +50.0     │ [+35.0, +65.0] │ improved │
└───────────────────────┴──────┴───────┴───────────┴────────────────┴──────────┘
No statistically clear regressions.
2 slice(s) had too few examples to judge (see the report).
```

Under the table AtlasForge lists any statistically clear regressions, or states that there are none, and counts slices that were too small to judge.

## The report

`cmp/comparison.md` has these sections:

1. **Header**: task, example count, dataset fingerprint, and a table identifying both runs (model, revision, backend, and how many examples **failed** or went **missing** in each).
2. **Overall**: for each metric and view, base and candidate scores, the delta, the 95% interval, the verdict, a win/tie/loss count, and a McNemar p-value for right-or-wrong metrics.
3. **Pooled figures**: corpus-level WER, CER, chrF and macro-F1 with no interval.
4. **Regressions**: every metric or slice that **statistically** got worse.
5. **Slices**: results for each slice field, worst first.
6. **Method**: the exact settings and caveats, so the report can be understood without this documentation.

A slice section from the same run:

```markdown
**domain**

| Value | n | Base | Candidate | Delta | 95% CI | Status |
|---|---|---|---|---|---|---|
| agri | 20 | 0.0% | 100.0% | - | - | insufficient data (n < 30) |
| legal | 20 | 0.0% | 0.0% | - | - | insufficient data (n < 30) |
```

Both slices show the means, but no verdict: 20 examples is below the 30-example floor.

## Slicing by your own fields

Anything you put under `meta` in the dataset becomes a slice:

```json
{"id": "a1", "input": "...", "reference": "...", "lang": "yo", "meta": {"domain": "agriculture", "source": "extension-office"}}
```

```bash
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --slice domain --slice source
```

Plan for slices when you **build** the dataset: with a 30-example floor per bucket, a dataset of 200 examples split four ways has little room. The built-in `lang`, `length` and `has_number` slices need no preparation.

## Using the JSON in scripts and CI

`cmp/comparison.json` has the same information as structured data:

```json
{
  "task": "generation",
  "dataset_sha256": "e6337504...",
  "n_total": 40,
  "base":      {"model": "base-m", "revision": null, "backend": "openai", "n_failed": 0, "n_missing": 0},
  "candidate": {"model": "cand-m", "revision": null, "backend": "openai", "n_failed": 0, "n_missing": 0},
  "primary": "exact_match@tone_aware",
  "metrics": [
    {
      "key": "exact_match@tone_aware", "name": "exact_match", "view": "tone_aware",
      "higher_is_better": true, "n": 40,
      "base_mean": 0.0, "candidate_mean": 0.5,
      "delta": 0.5, "low": 0.35, "high": 0.65,
      "verdict": "improved", "wins": 20, "ties": 20, "losses": 0,
      "base_corpus": null, "candidate_corpus": null,
      "mcnemar": {"base_only": 0, "candidate_only": 20, "p_value": 1.9e-06}
    }
  ],
  "slices": [
    {"field": "domain", "value": "legal", "n": 20, "base_mean": 0.0, "candidate_mean": 0.0,
     "delta": null, "low": null, "high": null, "status": "insufficient data"}
  ],
  "settings": {"n_boot": 1000, "seed": 0, "confidence": 0.95, "min_slice_n": 30,
               "slice_fields": ["lang", "length", "has_number", "domain"]}
}
```

Scores are fractions here (`0.5` is 50%), while chrF is on its own 0-100 scale. A "no regressions" gate for CI, for example, is:

```bash
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --out cmp
python - <<'PY'
import json, sys
report = json.load(open("cmp/comparison.json"))
bad = [m["key"] for m in report["metrics"] if m["verdict"] == "regressed"]
bad += [f'{s["field"]}={s["value"]}' for s in report["slices"] if s["status"] == "regressed"]
if bad:
    sys.exit("regressed: " + ", ".join(bad))
PY
```

`compare` itself always exits `0` when it produces a report. It does not fail on regressions, because whether a regression is acceptable is your call.

## Common situations

| You see | It means | What to do |
|---|---|---|
| everything "no clear change" | the data cannot distinguish the models | add examples, or accept that any difference is small |
| improved overall, one slice regressed | your change helped on average but hurt a group | look at that slice's examples in `results.jsonl`; add training data for it |
| a slice flagged but you tried many slices | may be chance (no multiple-comparison correction) | treat as a lead; confirm with more data |
| large failed counts in one run | that run's numbers include many wrong-by-failure answers | fix the server or timeouts and resume that run first |
| `not tested (pooled metric)` | macro-F1 has no per-example values | use `accuracy` for the test; use macro-F1 for the headline |
| "insufficient data" on every slice | slices are under 30 examples | use fewer, broader slices, or more data |
