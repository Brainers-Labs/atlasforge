# Evaluate a model

`atlasforge eval` runs a dataset through a model, scores the answers, and writes a report. This guide covers everything it can do.

```bash
atlasforge eval data.jsonl --out runs/base --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
```

## The three tasks

The `--task` option tells AtlasForge what kind of dataset it is reading and which metrics make sense.

| Task | The model is asked to | `reference` is | Default metrics |
|---|---|---|---|
| `generation` (default) | produce free text (translation, answer, rewrite) | optional | `exact_match`, `chrf` |
| `classification` | answer with one of a set of labels | **required** (the label) | `accuracy`, `macro_f1` |
| `asr` | transcribe audio | **required** (the transcript) | `wer`, `cer` |

Override the metrics with the repeatable `-m/--metric` option:

```bash
atlasforge eval data.jsonl --out runs/base --model ... -m exact_match -m chrf++ 
```

| Metric | Scale | Direction | Task |
|---|---|---|---|
| `exact_match` | 0-1 | higher is better | any |
| `chrf` | 0-100 | higher | any |
| `chrf++` | 0-100 | higher | any (adds word n-grams) |
| `wer` | fraction (can exceed 1) | **lower** is better | any, intended for `asr` |
| `cer` | fraction (can exceed 1) | **lower** | any, intended for `asr` |
| `accuracy` | 0-1 | higher | `classification` only |
| `macro_f1` | 0-1 | higher | `classification` only |

Details of how each is computed are in the [metrics reference](../reference/metrics.md).

## Connecting to a model

=== "OpenAI-compatible server"

    ```bash
    atlasforge eval data.jsonl --out runs/base \
      --backend openai --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
    ```

    `--backend openai` is the default, so you can omit it. Set `ATLASFORGE_BASE_URL` to avoid repeating the URL, and `ATLASFORGE_API_KEY` if the server needs a key.

=== "Local transformers"

    ```bash
    pip install -e ".[local]"
    atlasforge eval data.jsonl --out runs/base --backend local --model NCAIR1/N-ATLaS
    ```

    Needs enough memory for the weights. See [Run locally with transformers](local-backend.md).

## Generation settings

| Option | Default | Meaning |
|---|---|---|
| `--temperature` | `0.1` | `0` for greedy decoding |
| `--max-new-tokens` | `1000` | upper bound on generated tokens |

The repetition penalty is fixed at `1.12`. These defaults follow the settings recommended on the N-ATLaS model card (`temperature=0.1`, `repetition_penalty=1.12`, `max_new_tokens=1000`). They are written into `run.json`, and a run **cannot be resumed with different settings**.

!!! warning "The server's own defaults differ from the card"
    The model repository's `generation_config.json` ships `temperature=0.6` and no repetition penalty. AtlasForge sends its own values explicitly on every request, so what you evaluate is always the card's recommendation, regardless of server defaults. See [Verified model facts](../natlas/model-facts.md).

For the `openai` backend the request carries `temperature`, `max_tokens`, `repetition_penalty` and (if set) `top_p` and `seed`. If your server rejects `repetition_penalty`, the Python API can switch it off with `send_repetition_penalty=False`; see the [Python API guide](python-api.md).

## Resuming

A run directory is **resumable by design**. If a run stops for any reason (Ctrl-C, a crash, a dead server), re-run the exact same command:

```bash
atlasforge eval data.jsonl --out runs/base --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
# Resumed: 143 example(s) were already finished.
```

Rules:

- Examples already answered **successfully** are skipped.
- Examples that **failed** are retried. Pass `--no-retry-errors` to leave them as failures.
- AtlasForge first compares the run's identity with the existing `run.json`: task, dataset hash, model, revision, generation parameters and language. If any differ, it stops rather than mix results:

    ```text
    error: runs/base holds a different run (changed: model, dataset_sha256).
      -> Use a new output directory, or delete this one to start over.
    ```

    This protects you from the classic mistake of editing the dataset or switching the model and then comparing numbers that came from different things.

## Failures and the circuit breaker

Individual failures never abort a run; they are recorded and counted against the model. But a dead server would otherwise be retried for every remaining example, so there is a circuit breaker:

```bash
--max-consecutive-failures 20     # default; 0 disables
```

After that many failures **in a row**, `eval` stops with:

```text
error: Stopped after 20 failures in a row. Last error: BackendConnectionError: Could not reach http://127.0.0.1:8000/v1/ (ConnectError).
  -> Fix the cause, then re-run with the same --out to resume; everything already finished is kept.
```

The exit code is `2` in that case. After a normal finish, the exit code is `0`, or `1` if **every** example failed. If some failed, a warning on stderr tells you how many and where to look (`results.jsonl`).

Per-request retries are separate from this: the `openai` backend retries `429`, `500`, `502`, `503`, `504` and connection errors up to `--retries` times (default `2`) with exponential backoff, honouring a `Retry-After` header (capped at 30 seconds). Timeouts are **not** retried. Set `--timeout` (default 120 s) to suit slow models.

## Concurrency

```bash
atlasforge eval data.jsonl --out runs/base --model ... --concurrency 8
```

`--concurrency` sends that many requests in parallel (default `1`). It speeds up servers that batch requests, such as vLLM. Scores are identical to a sequential run; only the order examples finish in changes. Keep `1` for the `local` backend. If the server returns `429` (rate limited), lower the number.

## Classification datasets

For `task: classification`, each `reference` is a label. The model's free-text answer is mapped to a label by finding the label that appears **earliest** in the (normalised) answer as a whole word or phrase, with the longest label winning a tie. If no label is found, the answer counts as wrong.

```json
{"id": "s1", "input": "Classify the sentiment as positive or negative: ...", "reference": "positive"}
```

Tips:

- Ask the model to answer with the **label only**.
- Known limitation: **negation is not understood**. An answer like "not positive" would be read as `positive`. Keep prompts short and label-only.
- `macro_f1` is the average of per-class F1 over the classes present in your references. A missing label counts as a miss for its true class. Prefer it over accuracy when classes are imbalanced; [`dataset validate`](validate-dataset.md) warns when they are.

## Speech datasets

For `task: asr`, examples point to audio files and the references are transcripts. Clips over 30 seconds are chunked automatically. See [Transcribe and evaluate speech](asr.md).

## Reading the output

The console prints a table. The same numbers are in the files:

```text
runs/base/
  run.json         manifest: what was run
  results.jsonl    one record per example (prediction, latency, error)
  report.md        human-readable report
  report.json      machine-readable report, including per-example scores
```

A report looks like this. *(Generated against a stub server with synthetic data, to show the format; the numbers mean nothing.)*

```markdown
# AtlasForge evaluation report

- **Task:** generation
- **Examples:** 40 (40 ok, 0 failed, 0 missing)
- **Dataset fingerprint (sha256):** `e6337504252d466d`
- **Model:** cand-m
- **Backend:** openai

## Metrics

| Metric | View | Mean | Pooled | n |
|---|---|---|---|---|
| exact_match | tone-aware | 50.0% | - | 40 |
| chrf | tone-aware | 50.0 | 48.5 | 40 |
| exact_match | tone-insensitive | 50.0% | - | 40 |
| chrf | tone-insensitive | 50.0 | 48.5 | 40 |

## Latency

mean 1 ms | p50 1 ms | p95 1 ms | max 1 ms (successful calls only)
```

- **Mean** averages the per-example scores. **Pooled** is computed over the whole corpus (chrF, WER, CER, macro-F1); it can differ from the mean.
- Each text metric appears under both the tone-aware and tone-insensitive views.
- **Latency** counts successful calls only.
- The fingerprint is a prefix of the dataset's SHA-256 hash.

All file formats are specified in [Output files](../reference/output-files.md).

## Re-scoring without a model

Because scoring is separate from running, you can recompute a report from a finished run with different metrics, **without calling the model**:

```bash
atlasforge report runs/base --dataset data.jsonl -m exact_match -m chrf++
```

`report` needs the original dataset (to get the references) and the same `--task`. It rewrites `report.md` and `report.json` in the run directory. It does not alter `results.jsonl`.

## A complete workflow

```bash
# 1. check the data
atlasforge dataset validate data.jsonl --task generation

# 2. evaluate the base model
atlasforge eval data.jsonl --out runs/base --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS --concurrency 4

# 3. evaluate your change
atlasforge eval data.jsonl --out runs/tuned --base-url http://127.0.0.1:8001/v1 --model my-finetune --concurrency 4

# 4. compare
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --out cmp --slice domain
```

Continue with [Compare two models](compare.md).

!!! warning "Results can contain sensitive text"
    `results.jsonl` stores the model's answers, which may echo your data. Do not commit runs made from sensitive datasets. See [Security and privacy](../operations/security-privacy.md).
