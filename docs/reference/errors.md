# Errors and exit codes

Every error AtlasForge raises on purpose derives from `AtlasForgeError` and carries a **hint**: the concrete next step. The CLI prints both:

```text
error: Could not reach http://127.0.0.1:8000/v1/ (ConnectError).
  -> Is the server running, and is base_url correct?
```

and exits with code `2`. From Python, use `exc.message`, `exc.hint`, or `exc.format()`.

## Hierarchy

```text
AtlasForgeError
├── ConfigError              bad or missing configuration, or a missing optional extra
├── ModelAccessError         a gated model is not accessible (licence not accepted, bad token)
├── ResourceError            not enough VRAM, RAM or disk, or a required tool is missing
├── UnsupportedFeatureError  the backend or model lacks the requested capability
├── AudioError               audio is unreadable or in an unsupported format
├── DatasetError             a dataset file is malformed (has a 1-based `line` when known)
├── RunAborted               an eval run stopped early after too many failures in a row
└── BackendError             a backend failed to produce a result
    ├── BackendTimeout           the call exceeded its timeout
    ├── BackendConnectionError   the backend could not be reached
    └── BackendHTTPError         an HTTP backend answered with an error status
                                 (has `status_code`, `request_id`, `body_excerpt`)
```

## Common messages and what to do

| Message (abridged) | Class | Fix |
|---|---|---|
| `The openai backend needs a server URL.` | `ConfigError` | pass `--base-url` or set `ATLASFORGE_BASE_URL` |
| `base_url must be an http(s) URL` | `ConfigError` | include `http://` or `https://` |
| `Refusing plain http to a non-local host` | `ConfigError` | use `https://`, or `--allow-insecure-http` |
| `Unknown task` / `Unknown metric(s)` | `ConfigError` | the message lists the valid choices |
| `Unsupported language` | `ConfigError` | use `ha`, `yo`, `ig`, `en` |
| `accuracy and macro_f1 need a classification dataset.` | `ConfigError` | use `--task classification` or other metrics |
| `... holds a different run (changed: ...)` | `ConfigError` | use a new `--out`, or delete the old directory |
| `... was not produced from this dataset and task.` | `ConfigError` | `compare` needs runs from the same dataset file |
| `The local backend needs 'torch', which is not installed.` | `ConfigError` | `pip install "atlasforge[local]"` |
| `Cannot access NCAIR1/...` | `ModelAccessError` | accept the licence on that model page and set `HF_TOKEN` |
| `Not enough memory to load ...` | `ResourceError` | quantise, or serve the model elsewhere |
| `ffmpeg was not found on PATH.` | `ResourceError` | install ffmpeg |
| `line N: invalid JSON (...)` | `DatasetError` | fix that line; `dataset validate` lists all of them |
| `line N: unknown key(s): ...` | `DatasetError` | check spelling; extra columns go under `meta` |
| `cannot read ... is not valid UTF-8` | `DatasetError` | re-save the file as UTF-8 |
| `Audio is 45s; the ASR models take at most 30s.` | `AudioError` | use `transcribe` or `eval`, which chunk long audio |
| `Stopped after N failures in a row.` | `RunAborted` | fix the server, re-run the same command to resume |
| `No response within 120s.` | `BackendTimeout` | raise `--timeout`, or check server load |
| `Could not reach ... (ConnectError).` | `BackendConnectionError` | is the server running; is the URL right |

## HTTP errors

For an HTTP backend, `BackendHTTPError` messages read `Endpoint returned HTTP <status>.` plus a hint:

| Status | Hint |
|---|---|
| `400` | the server rejected the request; if it mentions `repetition_penalty`, create the backend with `send_repetition_penalty=False` |
| `401` | check the API key (`ATLASFORGE_API_KEY`) |
| `403` | the key works but is not allowed to do this |
| `404` | check that the base URL ends in `/v1` and the model name matches what the server serves |
| `429` | rate limited: lower the concurrency or retry later |
| `5xx` | server-side error; check the server logs |

`429`, `500`, `502`, `503` and `504` and connection failures are **retried automatically** up to `--retries` times. Timeouts and other statuses are not.

## Privacy of error messages

- Error messages **never contain response bodies**, because servers sometimes echo the prompt. A short excerpt (300 characters) is kept on `BackendHTTPError.body_excerpt` for debugging, and the CLI does not print it.
- In `results.jsonl`, an unexpected exception is recorded by its **type only**.
- Tokens are never part of any message.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the command ran, but the outcome is a failure: a `doctor` check failed; every `eval` example failed; a `transcribe` file failed; `dataset validate` found errors |
| `2` | an AtlasForge error (above), or invalid command-line usage |

Warnings, such as `doctor` warnings or `eval` runs where only some examples failed, do not change the exit code.
