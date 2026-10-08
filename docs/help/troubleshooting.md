# Troubleshooting

Find your symptom, usually the exact text AtlasForge printed, and follow the fix. Every error message also includes a `->` hint with the next step. If you are still stuck, see the [FAQ](faq.md) or open an issue.

!!! tip "Start with `atlasforge doctor`"
    It checks Python, GPU, disk, ffmpeg, your Hugging Face token, your access to each gated model, your project's `atlasforge.toml` if it has one, and the optional extras, and tells you the fix for each warning.

## Installing and environment

### `atlasforge.toml has unknown setting(s): ...`
A key in your [project file](../reference/configuration.md) is misspelled or is not one AtlasForge
reads. The error names the valid keys; a misspelled setting is an error rather than a silent no-op.
Point of confusion: the file says `base_url`, not `base-url`.

### My run used a model or endpoint I did not ask for
An `atlasforge.toml` is being picked up from the current directory or one above it. Run
`atlasforge doctor` and read the `config` row: it names the file and the settings it applies.

### `ffmpeg was not found on PATH.`
Speech needs ffmpeg to decode audio. Install it (`brew install ffmpeg`, `winget install Gyan.FFmpeg`, or `sudo apt install ffmpeg`) and **open a new terminal**, because `PATH` is read when a terminal starts.

### `zsh: no matches found: atlasforge[local]`
zsh expands square brackets. Quote the argument: `pip install -e ".[local]"`.

### `The local backend needs 'torch', which is not installed.`
You used `--backend local` without the extra. Run `pip install -e ".[local]"` (or `[asr]` for speech, `[finetune]` for training).

### PowerShell: a multi-line command breaks
In PowerShell, continue a line with a backtick `` ` ``, not a backslash. The guides show one-line commands where possible.

## Access to the models

### `hf-token ... no token found`
You have not given the shell a Hugging Face token. See [Access and licences](../get-started/access-and-licences.md). After setting `HF_TOKEN`, run `atlasforge doctor` again.

### `model-access ... licence not accepted on NCAIR1/...`
Having a token is not the same as having accepted the licence on a given model page — that is a separate click, per model. Open the page `doctor` names, agree to the terms while logged in, and run `doctor` again. Approval can take a little while.

### `model-access ... could not check`
`doctor` could not reach the Hub, so it is telling you it does not know rather than guessing. Usually there is no network, or `huggingface_hub` is not installed (`pip install "brainers-atlasforge[local]"`). Open the model page yourself to confirm access.

### `Cannot access NCAIR1/...`
The model is gated. Accept the licence on that model's Hugging Face page, make sure the token belongs to the same account, and set `HF_TOKEN`. Approval can take a little while.

## Talking to a server

### `The openai backend needs a server URL.`
Pass `--base-url http://127.0.0.1:8000/v1` or set `ATLASFORGE_BASE_URL`. The default backend is `openai`; use `--backend local` to load the model yourself.

### `Could not reach http://... (ConnectError).`
Nothing is answering at that address. Is the server running? Is the port right? Does the URL end in `/v1`? AtlasForge retries connection errors before giving up (`--retries`).

### `No response within 120s.`
The server is too slow or overloaded. Raise `--timeout`, lower `--concurrency`, or check the server. Timeouts are not retried, because a retry would double the wait.

### `Endpoint returned HTTP 404.`
Usually a wrong base URL or model name. The URL should end in `/v1`, and `--model` must match what the server serves.

### `Endpoint returned HTTP 401` or `403`
The server wants an API key. Put it in the `ATLASFORGE_API_KEY` environment variable.

### `Endpoint returned HTTP 400.`
The server rejected the request. If it complains about `repetition_penalty` (llama.cpp and Ollama use different names), add `--no-send-repetition-penalty`.

### `Endpoint returned HTTP 429` or `5xx`
Rate-limited or a server error. AtlasForge already retries these with backoff (`--retries`, default 2). Lower `--concurrency`, or look at the server's logs.

### `Refusing plain http to a non-local host`
AtlasForge only allows `http://` to `localhost`, `127.0.0.1` and `::1`. Use `https://`, or pass `--allow-insecure-http` if you trust the network.

## Running evaluations

### `Stopped after 20 failures in a row. Last error: ...`
The circuit breaker. Something is persistently failing, usually the server. Fix it, then run the **same command again**: everything finished is kept. Use `--max-consecutive-failures 0` to turn the breaker off.

### `... holds a different run (changed: model, ...)`
You reused an `--out` directory for a different dataset, model or settings. AtlasForge refuses to mix runs. Use a new `--out`.

### Many examples show as failed
Open `runs/<name>/results.jsonl` and look at the `error` field on failed lines. They name the error type.

### Every metric is identical in both tone views
Neither your data nor the model output contains tone marks, so the views cannot differ. That is normal for English. For Yoruba, it may mean the data has had its diacritics stripped: run [`dataset validate`](../guides/validate-your-data.md).

### `Unknown metric(s): ...` / `accuracy and macro_f1 need a classification dataset.`
Pick from the [metrics list](../reference/metrics.md). Accuracy and macro-F1 only work with `--task classification`.

### `Example 'x' has no language for ASR.`
Speech needs a language. Add `lang` to each example, or pass `--lang ha` as a default.

## Datasets

### `line 12: invalid JSON (...)`
That line is not valid JSON. Each line must be one complete JSON object. `dataset validate` lists all bad lines at once.

### `unknown key(s): refrence`
A misspelled field name. Allowed keys are `id`, `input`, `messages`, `audio`, `reference`, `lang`, `meta`. Put your own columns under `meta`.

### `provide exactly one of 'input' or 'messages'`
Each example needs a prompt as `input` (text) **or** `messages` (a list of turns), not both and not neither.

### `duplicate id 'x' (first seen on line N)`
Two examples share an id. Make them unique, or leave `id` out so one is derived from the content.

### `... is not valid UTF-8`
Re-save the file as UTF-8. Excel's "CSV" export is a common cause.

### `audio file not found: ...`
Speech paths are relative to the **dataset file**, and every file must exist when the dataset is loaded.

## Comparing

### `... was not produced from this dataset and task.`
`compare` pairs examples, so both runs must come from exactly the dataset file you passed. Even an edited copy has a different fingerprint. Re-run, or pass the original file.

### `... is not a run directory (no run.json).`
Point `--base` and `--candidate` at the folder you gave `eval --out`, not at a file inside it.

### `Primary metric '...' has no per-example values to compare.`
`--primary` must be a per-example metric such as `chrf@tone_aware`. Macro-F1 is pooled only, so it cannot be primary.

### Everything says `insufficient data`
Slices smaller than 30 examples are not judged. Use a larger dataset, a coarser slice, or (carefully) a lower `--min-slice-n`.

## Speech

### `Audio is 45s; the ASR models take at most 30s.`
That came from calling the local backend directly. The `transcribe` and `eval --task asr` commands split long audio for you.

### `ffmpeg could not decode the audio: ...`
The file is not valid audio or is damaged. Try converting it yourself first (`ffmpeg -i in.ogg out.wav`).

## Fine-tuning

### `QLoRA (4-bit) training needs an NVIDIA GPU, and none was found.`
QLoRA needs CUDA. Use a GPU machine (Colab, a cloud GPU, a cluster). It does not work on Macs or CPUs.

### `... also appear in test.jsonl`
Your training data overlaps your held-out file. Remove the overlap, or your evaluation will measure memorisation. See [Validate your data](../guides/validate-your-data.md#checking-for-leakage).

### `Only N training example(s); at least 20 are required.`
Fine-tuning on a handful of examples mostly teaches repetition. Add data, or lower `min_examples` if you truly mean it.

### `N of M training examples exceed max_seq_len ...`
They would be cut off mid-answer. Raise `max_seq_len` (needs more memory) or shorten or remove the long examples.

### `The tokenizer ... has no chat template`
Training text must be built the same way the model is prompted, and AtlasForge will not guess the format. This needs support added; please report it.

### `Unknown setting(s) in finetune.yaml: ...`
A typo in a setting name. The hint lists every valid name. See [Fine-tune settings](../reference/fine-tune-settings.md).

### `Reading a YAML config needs PyYAML.`
Install the extra (`pip install -e ".[finetune]"`) or use a `.json` config.

### `--adapter only works with the local backend.`
An adapter is loaded in-process. With a server, load the adapter there and pass its served name as `--model`.

### `Ran out of memory during training.`
Lower `max_seq_len` or `lora_r`, keep `batch_size` at 1 and raise `grad_accum`, or use a bigger GPU.

## Benchmarks

### `lm-evaluation-harness is not installed, so there are no tasks to run.`
The `bench` command does not reimplement the AfroBench-LITE tasks, it drives the harness that
defines them. Install it: `pip install "brainers-atlasforge[bench]"`. See the
[guide](../guides/benchmark-with-afrobench.md).

### `The installed harness has no task matching the afrobench-lite families.`
The harness is installed but names its tasks differently from the families in the study. Run
`atlasforge bench afrobench --list` to see what this installation has, then pass the exact names
with `--tasks afrixnli_yo --tasks belebele_hau` (repeatable).

### `The harness does not have '...'.`
A name given to `--tasks` is not one the installed harness offers — usually a language suffix that
this version spells differently. `atlasforge bench afrobench --list` prints the real ones.

### `The harness exited with status 1.`
The harness failed before writing results, so nothing was written. Its own output is printed above
this message and says why: a missing dependency, an out-of-memory kill, or a model it could not
load (check the gated licence and `HF_TOKEN`). Add `--dry-run` to print the exact command and run it
yourself with the harness's own verbosity.

### `The harness exited cleanly but wrote no results under ...`
It ran but put its file somewhere else. Its output names the path it used; this is why the report
records the file it actually read rather than the one it asked for. A harness run that reports "no
results" for a task usually means the task was not found — see `--list` above.

### Every benchmark number looks like a percentage, or chrF looks enormous
The report scales accuracy-like metrics (0-1 fractions) to percentages and leaves chrF and BLEU on
the harness's own 0-100 scale. Both scales are named under the headline table, and
`bench.json` holds every figure exactly as the harness reported it, with `harness/results*.json`
beside it — compare against that file before reporting anything as wrong.

## Still stuck?

Run `atlasforge doctor --json` and include its output, the exact command, and the error message when you ask for help. **Remove tokens and any private data first.**
