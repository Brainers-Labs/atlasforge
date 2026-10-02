# Troubleshooting

Start with `atlasforge doctor`. It catches most environment problems and prints the fix. For everything else, find your symptom below.

## Installation and environment

### `command not found: atlasforge`

The virtual environment is not active. Run `. .venv/bin/activate` (Windows: `.venv\Scripts\activate`), or call it as `python -m atlasforge`.

### `ffmpeg was not found on PATH`

Install it: `brew install ffmpeg` (macOS), `sudo apt install ffmpeg` (Linux), `winget install Gyan.FFmpeg` (Windows). Open a new terminal afterwards so `PATH` refreshes.

### `ffmpeg could not decode the audio`

The file is corrupt, empty or not audio. Convert it yourself to check: `ffmpeg -i input.m4a test.wav`. An `AudioError: The audio decoded to zero samples` means the file is empty or silent.

### `doctor` says "no NVIDIA GPU detected" on a Mac

Expected. `doctor` only recognises NVIDIA GPUs. Apple Silicon's GPU (MPS) still works for the speech models; see [Run on a Mac](../guides/mac.md).

### `Could not load NCAIR1/... (ImportError)` on macOS

*Symptom:* loading a speech model fails with `ImportError`, and importing `transformers` shows `dlopen(...scipy/sparse/linalg/_propack/_spropack...) ... section '__DATA/__thread_bss' has a zero-fill section type`.

*Cause:* the default `scipy` 1.15 wheel cannot load on some recent macOS releases on Apple Silicon. `transformers` imports it while loading, even though speech recognition does not use it. This is a `scipy` packaging problem, not an AtlasForge bug.

*Fix:*

```bash
pip install "scipy<1.15"
```

The `asr` and `local` extras already pin this on macOS, so a fresh install of those extras avoids it.

To see the real error behind the generic message, import the library directly: `python -c "import transformers; transformers.pipeline('automatic-speech-recognition', model='NCAIR1/Hausa-ASR')"`.

### `ModuleNotFoundError: No module named '_lzma'`

Your Python was built without the `xz` library (common with `pyenv`). It only affects the optional `datasets` package. Install `xz` (`brew install xz`) and reinstall Python.

### `The local backend needs 'torch', which is not installed.`

Install an extra: `pip install -e ".[asr]"` for the speech models or `pip install -e ".[local]"` for the LLM.

## Access and downloads

### `Cannot access NCAIR1/...` / `403 GatedRepoError`

Your token is valid but the licence is not accepted **on that specific repository**, or there is no token. Accept the terms on each of the five model pages while signed in, then set `HF_TOKEN`. A quick manual check: open the model page; if it still asks you to accept terms, you have not. See [Models and access](../getting-started/models-and-access.md).

### `hf-token` warning in `doctor`

No `HF_TOKEN`, no `HUGGING_FACE_HUB_TOKEN`, and no cached login. Set the variable or run `hf auth login`. `doctor` only checks that a token **exists**, not that your licences are accepted.

### Download fails with `CAS Client Error ... error sending request`

Hugging Face's Xet transfer path failed. Retry with plain HTTP:

```bash
HF_HUB_DISABLE_XET=1 atlasforge transcribe clip.wav --lang ha --backend local
```

(For manual downloads, set the same variable before your Python command.) Interrupted downloads resume from the cache.

### Disk fills up

The LLM is 16 GB and Hugging Face caches it under `~/.cache/huggingface/hub`. Keep at least 25 GB free (`doctor` warns below that). Remove models you no longer need from that cache.

## Connecting to a model

### `The openai backend needs a server URL.`

Pass `--base-url http://127.0.0.1:8000/v1` or `export ATLASFORGE_BASE_URL=...`.

### `Could not reach http://.../v1/ (ConnectError)`

Nothing is answering at that address. Check that the server is running, the port is right, and the URL includes `/v1`. AtlasForge retries connection errors `--retries` times before giving up.

### `Refusing plain http to a non-local host`

By design: it prevents sending an API key unencrypted. Use `https://`, or pass `--allow-insecure-http` if you trust the network.

### `HTTP 404`

The URL must end in `/v1` and `--model` must equal the name the server serves (for Ollama, the name you gave in `ollama create`).

### `HTTP 400` mentioning `repetition_penalty`

The server does not accept that field. From Python, create the backend with `send_repetition_penalty=False`.

### `HTTP 429` (rate limited)

Lower `--concurrency`. AtlasForge already retries `429` with backoff and honours `Retry-After`.

### `No response within 120s`

The model is slow or the server is overloaded. Raise `--timeout`. Timeouts are not retried.

### Replies look unformatted, odd, or ignore the roles

The server may not be applying the chat template. This happens with Ollama after importing safetensors weights. See [Serve the models](../guides/serve-models.md#3-restore-the-chat-template-important).

## Running evaluations

### `eval` stops with `Stopped after 20 failures in a row`

The circuit breaker tripped, usually because the server died. Everything finished is kept. Fix the cause and run the **same command** again to resume. Use `--max-consecutive-failures 0` to disable the breaker, though a dead server will then be tried for every example.

### `... holds a different run (changed: model, dataset_sha256)`

You re-used an `--out` directory with a different model, dataset, parameters or language. AtlasForge refuses to mix results. Use a new directory, or delete the old one to start over. If `dataset_sha256` is listed, the dataset file changed (even whitespace counts).

### `eval` exits `1` and says every example failed

Check `results.jsonl`: each record's `error` field says why (for example `BackendConnectionError` or `BackendHTTPError ... 401`).

### Scores are low and `n_failed` is high

Failures count as wrong answers, which drags scores down. Fix the failures first (check the `error` values), then resume the run to retry them; successful answers are kept.

### Both tone views are low

The model is getting the **words** wrong, so normalisation cannot help. If only the tone-aware view is low, the model's tone marking differs from the references. See [Tone-aware scoring](../concepts/tone-aware-scoring.md#reading-the-numbers).

### `compare` says the run "was not produced from this dataset and task"

The runs and the dataset you passed do not share a hash. Pass the exact file the runs used, with the same `--task`.

### Every slice says "insufficient data"

Slices under 30 examples never get a verdict. Use broader slices, add examples, or lower `--min-slice-n` knowing the conclusions get weaker.

### Classification accuracy is far lower than expected

The label extractor looks for a label word in the answer. If the model writes long sentences, or negates ("not positive"), it can pick the wrong label or none. Ask for the label only. Check `results.jsonl` to see the actual answers.

## Datasets

### `line N: unknown key(s): ...`

Only `id`, `input`, `messages`, `audio`, `reference`, `lang` and `meta` are allowed. Check for a typo; put extra columns under `meta`.

### `provide exactly one of 'input' or 'messages'`

A line has both, or neither. ASR datasets take `audio` instead.

### `is not valid UTF-8`

Re-save the file as UTF-8. Excel's "CSV" export and some Windows editors are common culprits.

### `dataset validate` warns that diacritics are missing

None of a language's texts contain the expected marks, which usually means they were stripped during export or copy-paste. Fix the source. Stripped diacritics change meaning.

## Speech

### `Audio is 45s; the ASR models take at most 30s`

You called the backend directly. Use `atlasforge transcribe` or `eval`, or `transcribe_long` in Python, which split long audio.

### A word is repeated or missing at a 28-second boundary

A known limitation of fixed windows (see [Speech](../guides/asr.md#how-long-audio-is-handled)). The merge removes duplicates only when at least two consecutive words match, to avoid deleting real repeats.

### Many `[transformers]` warnings on first run

Normal. They concern the fine-tuned models' stored settings and do not indicate a failure.

## Still stuck?

Run `atlasforge doctor --json` and open an issue with its output (it contains no secrets), the exact command, and the full error message including the `->` hint line. Never paste tokens, API keys, prompts or audio from sensitive datasets.
