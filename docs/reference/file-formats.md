# File formats

Everything AtlasForge reads or writes is plain UTF-8 text (JSON, Markdown or HTML) in your own folders. The examples below are **real files** produced by the offline demo while these docs are built, trimmed for reading.

## Dataset: `*.jsonl`

One JSON object per line. Full field reference in [Datasets](../concepts/datasets.md).

{{ file_example("dataset") }}

## Run directory

| File | Written by | Purpose |
|---|---|---|
| `run.json` | `eval` | The manifest: what produced the results |
| `results.jsonl` | `eval` | One line per example: the raw answer or the error |
| `report.json`, `report.md` | `eval`, `report` | Scored metrics |
| `report.html` | `eval`, `report` | The same numbers as one self-contained page |

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
| `flags` | Failure-mode flag counts over the answers that came back, keyed by flag name |
| `n_flagged` | How many answers raised at least one flag |
| `per_example_flags` | The flags on each flagged answer, by example id (absent for `classification`) |
| `asr` | For a `task` of `asr`: the alignment-based error analysis — hit/substitution/deletion/insertion counts, `tone_only_substitutions`, the top errors with example ids, and `by_length` buckets. `null` for other tasks. |
| `latency_ms` | mean, p50, p95, max of successful calls |
| `normalization` | The exact normalisation applied in each view |
| `dataset_sha256` | The dataset fingerprint |

### `report.html`

The same report as a page you can open in a browser, attach to an email, or read from an air-gapped machine. It is **one file**: no JavaScript, no stylesheet link, no font or image fetched from anywhere. The charts are inline SVG, and everything that came out of a report — a metric name, a slice value, a transcript word — is escaped on the way in, so an answer containing markup can only ever appear as text.

{{ file_example("report_html") }}

Everything after that is the stylesheet and the charts. The page repeats the Markdown's numbers rather than recomputing them: both renderers read the same `ScoreReport`, so they cannot disagree.

## Comparison: `comparison.json`

Written by `compare`, next to `comparison.md` and `comparison.html`.

{{ file_example("comparison") }}

| Field | Meaning |
|---|---|
| `base`, `candidate` | Model, revision, backend, failed, missing and flagged counts for each run |
| `metrics[]` | Per metric: base and candidate means, `delta`, interval `low`/`high`, `verdict`, wins/ties/losses, pooled values, McNemar result |
| `flags[]` | Per [failure-mode flag](../concepts/failure-modes.md): how often each run raised it, and the difference |
| `primary` | The metric the slices were computed for |
| `slices[]` | Per slice value: `n`, means, `delta`, interval, `status` |
| `settings` | `n_boot`, `seed`, `confidence`, `min_slice_n`, `slice_fields` |

`status` and `verdict` are one of `improved`, `regressed`, `no clear change`, `insufficient data` (slices only) or `not tested (pooled metric)` (macro-F1).

### `comparison.html`

The comparison as a page, written the same way as `report.html` and just as self-contained. The bars here are the *change* per metric — candidate minus base — with the 95% interval drawn as a whisker, so a verdict you cannot see in a table is visible at a glance.

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

## Benchmark outputs: `bench afrobench --out`

Three files in the output directory. The first two are written by AtlasForge; the third is the
harness's own and is kept verbatim, because it is the source the other two are read from.

```json
{
  "suite": "afrobench-lite",
  "model": "NCAIR1/N-ATLaS",
  "revision": null,
  "harness_version": "0.4.13",
  "missing_families": ["sib", "injongo", "afrimgsm"],
  "few_shot": null,
  "raw_path": "harness/results_2026-10-07.json",
  "tasks": [
    {
      "task": "afrixnli_yo",
      "primary_metric": "acc_norm,none",
      "primary": 0.3361,
      "metrics": { "acc,none": 0.3122, "acc_norm,none": 0.3361 }
    }
  ]
}
```

| File | What it is |
|---|---|
| `bench.json` | The above: every figure the harness reported per task, with the run's settings |
| `bench.md` | The same as a table, with one headline metric per task |
| `harness/results*.json` | The harness's own output, unmodified |

Two fields need a word. `revision: null` means the model's default revision was used, not that a
revision was pinned; `few_shot: null` means `--num_fewshot` was not passed, so each task ran with
whatever the harness defaults to for it — which is **not** necessarily zero-shot. The file name
under `harness/` is the harness's choice: `--output_path` is a request, some versions treat it as a
directory and invent a name inside it, so `raw_path` records the file that was actually found.

Illustrative: these figures came from a stand-in harness, so this shows the shape of the file and
not a measurement of any model. The AfroBench-LITE wrapper has not been run against the real harness
or the real weights — see [Status](../help/status.md).
