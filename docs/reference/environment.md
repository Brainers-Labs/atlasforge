# Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `HF_TOKEN` | local backend, `finetune`, `doctor` | Hugging Face access token for the gated `NCAIR1` models. Shown only masked. |
| `HUGGING_FACE_HUB_TOKEN` | same | Alternative name for the same token |
| `HF_HOME` | `doctor` | Where Hugging Face keeps its cache and saved login. `doctor` looks for a saved login here. |
| `ATLASFORGE_BASE_URL` | `run`, `transcribe`, `eval` | Default for `--base-url` |
| `ATLASFORGE_API_KEY` | `openai` backend | Bearer token sent to your server, if it needs one. Never printed. |

## Precedence

For options that have an environment variable, the command-line flag wins:

1. the flag you type (`--base-url ...`)
2. the environment variable (`ATLASFORGE_BASE_URL`)
3. the built-in default

## Setting them

=== "macOS / Linux"

    ```bash
    export HF_TOKEN=hf_your_token_here
    export ATLASFORGE_BASE_URL=http://127.0.0.1:8000/v1
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:HF_TOKEN = "hf_your_token_here"
    $env:ATLASFORGE_BASE_URL = "http://127.0.0.1:8000/v1"
    ```

These last for the current terminal session only. Never put a token in a file you commit.

## What AtlasForge does not read

There is no configuration file and no telemetry setting: nothing is collected, so there is nothing to opt out of.
