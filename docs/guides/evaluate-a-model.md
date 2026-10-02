# Evaluate a model

Run a dataset through a model and get a scored report.

**You need:** a [dataset](../concepts/datasets.md) and a model to talk to (see [Choose your setup](../get-started/choose-your-setup.md)). No model yet? Practise with the [offline demo](../get-started/quickstart.md).

## 1. Check the data first

```bash
atlasforge dataset validate data.jsonl --task generation
```

Fix any errors. Warnings are worth reading too. See [Validate your data](validate-your-data.md).

## 2. Run the evaluation

=== "Against a server"

    ```bash
    atlasforge eval data.jsonl \
      --out runs/base \
      --base-url http://127.0.0.1:8000/v1 \
      --model NCAIR1/N-ATLaS
    ```

=== "Loading the model locally"

    ```bash
    atlasforge eval data.jsonl \
      --out runs/base \
      --backend local \
      --model NCAIR1/N-ATLaS
    ```

    Needs the `local` extra and enough GPU memory. See [Installation](../get-started/installation.md).

=== "With a base URL from the environment"

    ```bash
    export ATLASFORGE_BASE_URL=http://127.0.0.1:8000/v1
    atlasforge eval data.jsonl --out runs/base
    ```

While it runs you see a progress bar. When it finishes you get a metrics table and two files, `runs/base/report.md` and `runs/base/report.json`.

## 3. Read the result

This is the table from the offline demo run (yours will show your own numbers):

{{ transcript("report") }}

- **aware / insens.** are the two [tone views](../concepts/tone-aware-scoring.md).
- **Mean** averages per-example scores; **Pooled** is computed over the whole corpus (see [Statistics](../concepts/statistics.md#mean-versus-pooled)).
- **Failed** examples count as wrong. If many failed, look at `runs/base/results.jsonl`: each failure line names the error.

## Choosing metrics

By default the metrics depend on `--task`. Override with `--metric` (repeat it for several):

```bash
atlasforge eval data.jsonl --out runs/base -m exact_match -m chrf
```

See the [metrics reference](../reference/metrics.md) for all of them.

## Classification

```bash
atlasforge eval reviews.jsonl --task classification --out runs/clf
```

Every example needs a `reference` (the label). Ask the model to answer with the label only; see [Datasets](../concepts/datasets.md#tasks).

## Options you will use

| Option | Use it to |
|---|---|
| `--concurrency 8` | Send several requests at once (servers only). Keep it at `1` for a local GPU. |
| `--temperature 0` | Make answers deterministic, for repeatable comparisons |
| `--max-new-tokens` | Cap the answer length |
| `--timeout`, `--retries` | Tune patience for a slow or flaky server |
| `--no-send-repetition-penalty` | Stop sending `repetition_penalty` if your server rejects it |
| `--max-consecutive-failures` | Stop early when the server is down (default 20) |

The full list is in the [CLI reference](../reference/cli.md#atlasforge-eval).

## If it stops

- **Server down.** After 20 failures in a row it stops and tells you. Fix the server and run the same command again: finished examples are kept. See [Run directories](../concepts/run-directories.md#the-circuit-breaker).
- **Ctrl-C or a crash.** Run the same command again; it resumes.
- **"holds a different run".** You changed the dataset, model or settings but reused `--out`. Use a new directory.

## Re-scoring without the model

```bash
atlasforge report runs/base --dataset data.jsonl -m chrf++
```

`report` reads the saved answers and writes a fresh report. It never calls a model.

## The same thing in Python

```python
from atlasforge.backends.openai import OpenAIBackend
from atlasforge.eval.dataset import load_dataset
from atlasforge.eval.runner import RunConfig, read_results, run
from atlasforge.eval.score import score_run

dataset = load_dataset("data.jsonl", "generation")
backend = OpenAIBackend(base_url="http://127.0.0.1:8000/v1", model="NCAIR1/N-ATLaS")

run(backend, dataset, "runs/base", config=RunConfig(concurrency=4))
report = score_run(
    dataset, read_results("runs/base/results.jsonl"), metrics=["exact_match", "chrf"]
)
print(report.metric("chrf@tone_aware").mean)
```

## Next

[Compare two models](compare-two-models.md): evaluate a second model the same way, then compare.
