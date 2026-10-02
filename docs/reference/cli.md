# CLI reference

```text
atlasforge [OPTIONS] COMMAND [ARGS]...
```

| Command | Purpose |
|---|---|
| [`doctor`](#atlasforge-doctor) | check Python, GPU, disk, ffmpeg, Hugging Face token and extras |
| [`run`](#atlasforge-run) | send one prompt to the LLM and print the answer |
| [`transcribe`](#atlasforge-transcribe) | transcribe audio with an official speech model |
| [`eval`](#atlasforge-eval) | run a dataset through a model, score it, write a report |
| [`report`](#atlasforge-report) | re-score a finished run, no model needed |
| [`compare`](#atlasforge-compare) | compare two finished runs statistically |
| [`dataset validate`](#atlasforge-dataset-validate) | find problems in a dataset |

Global options:

| Option | Meaning |
|---|---|
| `--version`, `-V` | print the version and exit |
| `--help` | show help (works on every command) |

`python -m atlasforge ...` is equivalent to `atlasforge ...`.

## Options shared by commands that talk to a model

These apply to `run`, `transcribe` and `eval`, except where noted.

| Option | Short | Default | Meaning |
|---|---|---|---|
| `--backend` | `-b` | `openai` | `openai`: any OpenAI-compatible server. `local`: `transformers` in-process |
| `--base-url` | | env `ATLASFORGE_BASE_URL` | the server URL, e.g. `http://127.0.0.1:8000/v1`. **Required** for the `openai` backend |
| `--model` | | `NCAIR1/N-ATLaS` | model name as the server knows it, or a Hugging Face repo id |
| `--quantize` | | `none` | `local` backend only (`run`, `eval`): `none`, `4bit` or `8bit` (NVIDIA GPU) |
| `--device` | | `auto` | `local` backend only: `auto`, `cpu`, `cuda`, `mps` |
| `--timeout` | | `120.0` | seconds to wait per request |
| `--retries` | | `2` | retries on `429`, `5xx` and connection errors (`openai` backend) |
| `--allow-insecure-http` | | off | permit plain `http://` to a non-local host |

The API key is read from the environment variable `ATLASFORGE_API_KEY`. There is deliberately no `--api-key` flag.

---

## `atlasforge doctor`

Check that your environment is ready.

```bash
atlasforge doctor [--json]
```

| Option | Meaning |
|---|---|
| `--json` | print the checks as JSON: a list of `{name, status, detail, hint}` |

**Exit code:** `1` if any check has status `fail`, otherwise `0` (warnings do not fail). See [Installation](../getting-started/installation.md#check-your-setup) for the checks.

---

## `atlasforge run`

Send one prompt and print the answer.

```bash
atlasforge run PROMPT [options]
```

| Argument / option | Default | Meaning |
|---|---|---|
| `PROMPT` | required | the prompt, or `-` to read it from standard input |
| `--system` | none | an optional system prompt |
| `--temperature` | `0.1` | `0` for greedy decoding |
| `--max-new-tokens` | `1000` | maximum tokens to generate |
| `--seed` | none | sampling seed |
| `--json` | off | print a JSON object instead of just the text |
| shared model options | | see above |

Examples:

```bash
atlasforge run "Ina kwana?" --base-url http://127.0.0.1:8000/v1
echo "Translate to English: Sannu" | atlasforge run - --base-url http://127.0.0.1:8000/v1
atlasforge run "Hello" --system "Reply in Yoruba." --json --base-url http://127.0.0.1:8000/v1
```

`--json` output:

```json
{
  "text": "…",
  "model": "NCAIR1/N-ATLaS",
  "revision": null,
  "latency_ms": 812.4,
  "finish_reason": "stop",
  "request_id": "chatcmpl-…",
  "usage": {"prompt_tokens": 12, "completion_tokens": 20}
}
```

---

## `atlasforge transcribe`

Transcribe one or more audio files.

```bash
atlasforge transcribe FILE... --lang ha|yo|ig|en [options]
```

| Argument / option | Short | Default | Meaning |
|---|---|---|---|
| `FILE...` | | required | audio files: wav, mp3, m4a, ogg/opus, and anything ffmpeg reads |
| `--lang` | `-l` | **required** | `ha`, `yo`, `ig`, `en` (or `hausa`, `yoruba`, `igbo`, `english`) |
| `--out` | `-o` | none | write one JSON record per file to this JSONL file |
| `--json` | | off | print JSONL instead of plain text |
| shared model options (no `--quantize`) | | | see above |

Long audio is split into overlapping 28-second windows automatically. With one file and no `--json`/`--out`, only the text is printed; with several files, or `--json`, one JSON record per line. A failing file prints `error: <file>: ...` to stderr and the rest continue.

**Exit code:** `1` if any file failed, otherwise `0`. Details: [Transcribe and evaluate speech](../guides/asr.md).

---

## `atlasforge eval`

Run a dataset through a model, score it, and write `report.md` and `report.json`. Resumable.

```bash
atlasforge eval DATASET --out RUN_DIR [options]
```

| Argument / option | Short | Default | Meaning |
|---|---|---|---|
| `DATASET` | | required | the JSONL dataset |
| `--out` | `-o` | **required** | the run directory (created if absent; resumable) |
| `--task` | `-t` | `generation` | `generation`, `classification` or `asr` |
| `--metric` | `-m` | task defaults | repeatable. `exact_match`, `chrf`, `chrf++`, `wer`, `cer`, `accuracy`, `macro_f1` |
| `--concurrency` | | `1` | parallel requests (HTTP backends) |
| `--lang` | `-l` | none | default language for ASR examples without their own `lang` |
| `--temperature` | | `0.1` | `0` for greedy decoding |
| `--max-new-tokens` | | `1000` | maximum tokens to generate |
| `--retry-errors` / `--no-retry-errors` | | retry | re-run failed examples when resuming |
| `--max-consecutive-failures` | | `20` | stop after this many failures in a row; `0` never stops |
| shared model options | | | see above |

For `--task asr` the backend is wrapped so clips over 30 seconds are chunked. Re-running into the same `--out` resumes, after checking that the manifest matches.

**Exit code:** `0` normally (with a stderr warning if some examples failed); `1` if **every** example failed; `2` on an error such as a bad dataset, a mismatched run directory or an aborted run (circuit breaker). See [Evaluate a model](../guides/evaluate.md).

---

## `atlasforge report`

Re-score a finished run without calling the model.

```bash
atlasforge report RUN_DIR --dataset DATASET [--task TASK] [-m METRIC]...
```

| Argument / option | Short | Default | Meaning |
|---|---|---|---|
| `RUN_DIR` | | required | a directory made by `atlasforge eval` |
| `--dataset` | `-d` | **required** | the dataset it was run on |
| `--task` | `-t` | `generation` | must match the run |
| `--metric` | `-m` | task defaults | repeatable |

Rewrites `report.md` and `report.json` in the run directory and prints the table. `results.jsonl` is not touched.

---

## `atlasforge compare`

Compare two runs of the same dataset.

```bash
atlasforge compare DATASET --base RUN_A --candidate RUN_B [options]
```

| Argument / option | Short | Default | Meaning |
|---|---|---|---|
| `DATASET` | | required | the dataset both runs used |
| `--base` | | **required** | run directory of the base model |
| `--candidate` | | **required** | run directory of the candidate |
| `--out` | `-o` | `comparison` | output directory (created) |
| `--task` | `-t` | `generation` | must match the runs |
| `--metric` | `-m` | task defaults | repeatable |
| `--primary` | | first tone-aware metric | metric for slice analysis, e.g. `chrf@tone_aware` |
| `--slice` | | `lang`, `length`, `has_number` | repeatable; add a `meta` key |
| `--n-boot` | | `1000` | bootstrap resamples |
| `--seed` | | `0` | bootstrap seed |
| `--min-slice-n` | | `30` | smaller slices are "insufficient data" |

Writes `comparison.md` and `comparison.json`. **Exit code** is `0` when a report is produced (even if it shows regressions) and `2` on errors such as runs from a different dataset. See [Compare two models](../guides/compare.md).

---

## `atlasforge dataset validate`

Find malformed lines, duplicates, leakage, broken Unicode and stripped diacritics.

```bash
atlasforge dataset validate FILE [--task TASK] [--against OTHER] [--json]
```

| Argument / option | Short | Default | Meaning |
|---|---|---|---|
| `FILE` | | required | the JSONL dataset |
| `--task` | `-t` | `generation` | `generation`, `classification` or `asr` |
| `--against` | | none | another split, to check for train/test leakage |
| `--json` | | off | print the full report as JSON |

**Exit code:** `1` if there are errors, `0` otherwise (warnings never fail). See [Validate a dataset](../guides/validate-dataset.md).

---

## Exit codes at a glance

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the command ran but the result is a failure: `doctor` check failed, every `eval` example failed, a `transcribe` file failed, `dataset validate` found errors |
| `2` | an AtlasForge error was raised (bad input, configuration, backend or run problem); the message and hint are printed to stderr |
| `2` (from the CLI framework) | invalid command line usage |

See also [Errors and exit codes](errors.md).
