# Configuration

AtlasForge has **no configuration file**. Everything is set by command-line options, environment variables or function arguments. This keeps runs reproducible: what you typed is what ran, and the important parts are recorded in each run's `run.json`.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `ATLASFORGE_BASE_URL` | `run`, `transcribe`, `eval` | default for `--base-url` |
| `ATLASFORGE_API_KEY` | the `openai` backend | the bearer token sent as `Authorization: Bearer ...` |
| `HF_TOKEN` | the `local` backend, `doctor` | Hugging Face access token for the gated models |
| `HUGGING_FACE_HUB_TOKEN` | the `local` backend, `doctor` | older name for the same thing |
| `HF_HOME` | `doctor`, Hugging Face libraries | where Hugging Face keeps its cache and login; `doctor` looks for a cached login here, else in `~/.cache/huggingface` |
| `HF_HUB_DISABLE_XET` | Hugging Face libraries | set to `1` to avoid the Xet transfer path if downloads fail with `CAS Client Error` |

**Precedence:** a command-line option always beats the environment variable.

!!! tip "Why there is no `--api-key` flag"
    Command-line arguments appear in shell history and in the process list. Keys come from the environment so they stay out of both. AtlasForge never prints a key; tokens shown by `doctor` are masked (`hf_****abcd`).

The API key variable name can be changed from Python (`OpenAIBackend(api_key_env="MY_VAR")`), or you can pass the key directly with `api_key=`.

## Defaults

| Setting | Default | Source |
|---|---|---|
| model | `NCAIR1/N-ATLaS` | the official LLM |
| backend | `openai` | |
| temperature | `0.1` | N-ATLaS model card |
| repetition penalty | `1.12` | N-ATLaS model card |
| max new tokens | `1000` | N-ATLaS model card |
| request timeout | `120` seconds | |
| retries | `2` | with exponential backoff, starting at 0.5 s and capped at 8 s; a `Retry-After` header is honoured up to 30 s |
| concurrency | `1` | |
| circuit breaker | `20` consecutive failures | |
| ASR window / overlap | `28 s` / `2 s` | the models accept at most 30 s |
| bootstrap resamples / seed | `1000` / `0` | |
| minimum slice size | `30` | |
| confidence level | `95%` | fixed |

!!! warning "The model repository's own generation defaults differ"
    The model's `generation_config.json` has `temperature=0.6` and no repetition penalty. AtlasForge sends its own explicit values on every request, so servers' defaults never affect an evaluation. See [Verified model facts](../natlas/model-facts.md).

## What gets recorded

The `run.json` manifest stores the model, revision, backend, device, dtype, generation parameters, language and dataset hash, so any result can be traced back to the settings that produced it. It never stores tokens, keys or URLs.

## Network safety

- Plain `http://` is refused for any host other than `localhost`, `127.0.0.1` and `::1`, unless `--allow-insecure-http` is given. This stops an API key being sent unencrypted by accident.
- Error messages from the HTTP backend never include the response body, because servers sometimes echo the prompt back.
