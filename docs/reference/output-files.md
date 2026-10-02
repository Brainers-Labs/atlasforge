# Output files

Everything AtlasForge writes is plain UTF-8 text (JSON or Markdown), so you can read it, diff it, and process it with any tool. Non-ASCII text is written readably, not as `\u` escapes.

!!! note "Example data"
    The samples on this page come from a run against a stub server with synthetic data, to show exact structure. The numbers mean nothing.

## A run directory

Created by `atlasforge eval --out runs/base`:

```text
runs/base/
  run.json         manifest: what was run
  results.jsonl    one record per example, appended as each finishes
  report.md        human-readable report (rewritten by `eval` and `report`)
  report.json      machine-readable report (rewritten by `eval` and `report`)
```

`run.json` and `results.jsonl` are the **source of truth**. The two reports are derived from them and can be regenerated any time with `atlasforge report`.

### `run.json`

Written once when the run starts (atomically). On resume it is compared with the new run's identity instead of being overwritten.

```json
{
  "schema_version": 1,
  "atlasforge_version": "0.1.0.dev0",
  "task": "generation",
  "dataset_sha256": "e6337504252d466d3223c7a766272cdb22fa0d21f0989b42aae5fdf82a985290",
  "n_examples": 40,
  "backend": "openai",
  "model": "cand-m",
  "revision": null,
  "device": null,
  "dtype": null,
  "capabilities": ["generate", "transcribe"],
  "gen_params": {
    "temperature": 0.1,
    "repetition_penalty": 1.12,
    "max_new_tokens": 1000,
    "top_p": null,
    "seed": null
  },
  "lang": null,
  "started_at": "2026-10-02T07:36:57+00:00"
}
```

| Field | Meaning |
|---|---|
| `schema_version` | format version of this file (currently `1`) |
| `atlasforge_version` | the version that produced the run |
| `task` | `generation`, `classification` or `asr` |
| `dataset_sha256` | fingerprint of the exact dataset bytes |
| `n_examples` | number of examples in the dataset |
| `backend`, `model`, `revision`, `device`, `dtype` | what actually ran. `revision`, `device` and `dtype` are filled by backends that know them (the `local` backend) |
| `capabilities` | what the backend advertised |
| `gen_params` | generation settings |
| `lang` | the run-wide default language (ASR), or `null` |
| `started_at` | UTC time the run began |

**Identity keys.** A run directory can only be resumed if `task`, `dataset_sha256`, `model`, `revision`, `gen_params` and `lang` are all unchanged. Everything else (timestamps, versions, device) may differ.

### `results.jsonl`

One JSON object per line, appended and flushed as each example finishes. Exactly one of `prediction` and `error` is set.

```json
{"id": "e0", "prediction": "ans0", "latency_ms": 1.04, "error": null, "request_id": "chatcmpl-test"}
{"id": "e7", "prediction": null, "latency_ms": null, "error": "BackendHTTPError: Endpoint returned HTTP 500.", "request_id": null}
```

| Field | Meaning |
|---|---|
| `id` | the example id |
| `prediction` | the model's text (the transcript, for ASR) |
| `latency_ms` | time for the successful call, in milliseconds |
| `error` | `ErrorType: message` for AtlasForge errors; for anything unexpected only `UnexpectedError: <ExceptionType>`, so prompt text can never leak through an exception message |
| `request_id` | the server's request id (`x-request-id` header or response `id`), when available |

If an example appears more than once (a failure that was retried), **the latest record wins**. A truncated last line from a crash is ignored and that example reruns; a corrupt line anywhere else is an error.

!!! warning "Contains model output"
    Predictions can echo your input data. Do not commit run directories made from sensitive datasets.

### `report.json`

The scored run: summary metrics plus per-example values.

```json
{
  "task": "generation",
  "dataset_sha256": "e6337504…",
  "n_total": 40, "n_ok": 40, "n_failed": 0, "n_missing": 0,
  "metrics": [
    {"key": "chrf@tone_aware", "name": "chrf", "view": "tone_aware",
     "n": 40, "mean": 50.0, "corpus": 48.49, "higher_is_better": true}
  ],
  "per_example": {
    "e0": {"exact_match@tone_aware": 1.0, "chrf@tone_aware": 100.0,
           "exact_match@tone_insensitive": 1.0, "chrf@tone_insensitive": 100.0}
  },
  "latency_ms": {"mean": 0.74, "p50": 0.72, "p95": 0.92, "max": 1.04},
  "normalization": {
    "tone_aware":       {"tones": "keep",  "lowercase": true, "punctuation": "strip"},
    "tone_insensitive": {"tones": "strip", "lowercase": true, "punctuation": "strip"}
  }
}
```

| Field | Meaning |
|---|---|
| `n_total`, `n_ok`, `n_failed`, `n_missing` | outcome counts; failed and missing are scored as wrong |
| `metrics[].key` | `name@view`, e.g. `wer@tone_insensitive` |
| `metrics[].mean` / `corpus` | mean of per-example values / pooled value (`null` if none) |
| `metrics[].n` | examples scored (those with a reference) |
| `per_example` | per-example scores keyed by id and metric key; `macro_f1` has none |
| `latency_ms` | mean, p50, p95, max over successful calls; `null` when there were none |
| `normalization` | the exact settings of each view |

### `report.md`

The same content as a Markdown document: header (task, counts, dataset fingerprint, model, backend), a metrics table, latency, and a short method note. See the full sample in [Evaluate a model](../guides/evaluate.md#reading-the-output).

## A comparison directory

Created by `atlasforge compare --out cmp`:

```text
cmp/
  comparison.md    Markdown report
  comparison.json  machine-readable report
```

`comparison.json` fields:

| Field | Meaning |
|---|---|
| `task`, `dataset_sha256`, `n_total` | identity of the compared dataset |
| `base`, `candidate` | `{model, revision, backend, n_failed, n_missing}` for each run |
| `primary` | the metric key used for slices |
| `metrics[]` | per metric: `key`, `name`, `view`, `higher_is_better`, `n`, `base_mean`, `candidate_mean`, `delta` (candidate minus base), `low`/`high` (95% interval), `verdict`, `wins`/`ties`/`losses`, `base_corpus`/`candidate_corpus`, `mcnemar` |
| `metrics[].verdict` | `improved`, `regressed`, `no clear change`, or `not tested (pooled metric)` |
| `metrics[].mcnemar` | `{base_only, candidate_only, p_value}` for 0/1 metrics, else `null` |
| `slices[]` | per slice value: `field`, `value`, `n`, `base_mean`, `candidate_mean`, `delta`, `low`, `high`, `status` |
| `slices[].status` | `improved`, `regressed`, `no clear change` or `insufficient data` |
| `settings` | `n_boot`, `seed`, `confidence`, `min_slice_n`, `slice_fields` |

A full sample is shown in [Compare two models](../guides/compare.md#using-the-json-in-scripts-and-ci). `delta`, `low` and `high` use the metric's own scale (fractions for accuracy-like metrics, points for chrF).

## Other machine-readable outputs

| Command | Output |
|---|---|
| `doctor --json` | a list of `{"name", "status", "detail", "hint"}` objects, `status` being `ok`, `warn` or `fail` |
| `run --json` | `{text, model, revision, latency_ms, finish_reason, request_id, usage}` |
| `transcribe --json` / `--out` | one object per file: `{file, lang, text, latency_ms, chunks: [{start_s, end_s, text}]}` |
| `dataset validate --json` | `{path, task, sha256, n_lines, n_examples, issues, stats}`; see [Validate a dataset](../guides/validate-dataset.md#statistics-in-the-json-report) |
