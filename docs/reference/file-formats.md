# File formats

Everything AtlasForge reads or writes is plain UTF-8 text (JSON or Markdown) in your own folders. The examples below are **real files** produced by the offline demo while these docs are built, trimmed for reading.

## Dataset: `*.jsonl`

One JSON object per line. Full field reference in [Datasets](../concepts/datasets.md).

{{ file_example("dataset") }}

## Run directory

| File | Written by | Purpose |
|---|---|---|
| `run.json` | `eval` | The manifest: what produced the results |
| `results.jsonl` | `eval` | One line per example: the raw answer or the error |
| `report.json`, `report.md` | `eval`, `report` | Scored metrics |

See [Run directories](../concepts/run-directories.md) for behaviour.

### `run.json`

{{ file_example("run") }}

| Field | Meaning |
|---|---|
| `schema_version` | Format version of this file |
| `task` | `generation`, `classification` or `asr` |
| `dataset_sha256`, `n_examples` | Fingerprint and size of the dataset |
| `backend`, `model`, `revision`, `device`, `dtype` | What ran. `revision` is the model's commit when known. |
| `gen_params` | Temperature, repetition penalty, token limit, `top_p`, seed |
| `lang` | Default language (speech) |
| `capabilities` | What the backend can do |
| `started_at`, `atlasforge_version` | When and with which version |

Resuming is refused if `task`, `dataset_sha256`, `model`, `revision`, `gen_params` or `lang` change.

### `results.jsonl`

{{ file_example("results") }}

| Field | Meaning |
|---|---|
| `id` | The example's id |
| `prediction` | The raw answer. Null on failure. |
| `latency_ms` | Time for the call. Null on failure. |
| `error` | `ExceptionName: message`, or just the exception type for unexpected errors. Null on success. |
| `request_id` | The server's request id, when it provided one |

If an id appears more than once (a retry), the **last** line wins.

### `report.json`

{{ file_example("report") }}

| Field | Meaning |
|---|---|
| `n_total`, `n_ok`, `n_failed`, `n_missing` | Counts. Failed and missing are scored as wrong. |
| `metrics[]` | One entry per metric and view: `key` (`name@view`), `mean`, `corpus` (pooled), `n`, `higher_is_better` |
| `per_example` | Every example's score for every per-example metric, used for paired statistics |
| `latency_ms` | mean, p50, p95, max of successful calls |
| `normalization` | The exact normalisation applied in each view |
| `dataset_sha256` | The dataset fingerprint |

## Comparison: `comparison.json`

Written by `compare`, next to `comparison.md`.

{{ file_example("comparison") }}

| Field | Meaning |
|---|---|
| `base`, `candidate` | Model, revision, backend, failed and missing counts for each run |
| `metrics[]` | Per metric: base and candidate means, `delta`, interval `low`/`high`, `verdict`, wins/ties/losses, pooled values, McNemar result |
| `primary` | The metric the slices were computed for |
| `slices[]` | Per slice value: `n`, means, `delta`, interval, `status` |
| `settings` | `n_boot`, `seed`, `confidence`, `min_slice_n`, `slice_fields` |

`status` and `verdict` are one of `improved`, `regressed`, `no clear change`, `insufficient data` (slices only) or `not tested (pooled metric)` (macro-F1).

## `training_run.json`

Written next to the adapter by `finetune`; read by `card`.

```json
{
  "method": "qlora",
  "base_model": "NCAIR1/N-ATLaS",
  "base_revision": null,
  "n_train_examples": 1200,
  "n_truncated": 0,
  "dataset_sha256": "...",
  "atlasforge_version": "...",
  "started_at": "...",
  "finished_at": "...",
  "output_dir": "adapters/my-adapter",
  "hyperparameters": { "lora_r": 16, "learning_rate": 0.0002 }
}
```

Illustrative: `finetune` has not yet been run on a GPU, so this shows the structure, not a real run.

## Transcripts: `transcribe --out`

One JSON object per line:

```json
{"file": "voice.ogg", "lang": "ha", "text": "...", "latency_ms": 1234.5,
 "chunks": [{"start_s": 0.0, "end_s": 28.0, "text": "..."}]}
```

`start_s` and `end_s` are AtlasForge's own chunk boundaries, **not** word timestamps from the model.

## Model card: `MODEL_CARD.md`

Markdown with Hugging Face front matter. See [Publish a model card](../guides/publish-a-model-card.md).
