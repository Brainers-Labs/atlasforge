# Security and privacy

AtlasForge handles API keys, model access tokens, prompts and sometimes people's voices. This page describes what it does to protect them, and what you must do yourself.

## Secrets

| Secret | How AtlasForge treats it |
|---|---|
| Hugging Face token | read from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or the cached login; shown only masked (`hf_****abcd`); never written to reports, manifests or logs |
| Endpoint API key | read from `ATLASFORGE_API_KEY`; there is **no** `--api-key` flag, so it never lands in shell history or the process list; sent only as an `Authorization` header |
| Anything else | not collected |

Masking reveals at most a short prefix and the last four characters, and reveals nothing for short values.

**What you should do:** never commit tokens or keys, never paste them into chat or screenshots, and revoke a token immediately at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) if exposed. The repository runs `gitleaks` as a pre-commit hook and in CI.

## Network

- Plain `http://` is refused for any host except `localhost`, `127.0.0.1` and `::1`, so a key cannot be sent unencrypted by accident. `--allow-insecure-http` overrides this knowingly.
- AtlasForge's own code talks only to the server **you** give it and adds no telemetry. The `local` backend uses the Hugging Face libraries, which contact Hugging Face to download gated models and have their own network behaviour and offline switches (see the Hugging Face documentation, for example `HF_HUB_OFFLINE`).
- A shared model endpoint should use HTTPS, a key per user (so keys can be revoked individually), a rate limit, a hard spending cap on the GPU provider, an auto-shutdown schedule, and no logging of prompt content.

## Logs, errors and reports

- Default output contains **no prompts, completions, audio or tokens** apart from what you explicitly asked for.
- Error messages never include HTTP response bodies, because servers sometimes echo the prompt. A short excerpt is kept on the exception object for debugging but is not printed.
- In `results.jsonl`, unexpected exceptions are recorded by **type only**.
- **`results.jsonl` does contain model predictions**, which is its purpose, and predictions can echo your inputs. Do not commit run directories made from sensitive data, and delete them when you are done.
- `report.json` holds per-example **scores**, not text.

## Voice and personal data

Voice recordings and transcripts are personal data. Under the Nigeria Data Protection Act 2023 (which applies to data you collect):

- get explicit consent before recording anyone,
- keep transcripts and delete raw audio after use where you can,
- never commit raw recordings or put them in a public dataset,
- do not log audio content.

## Model files

AtlasForge never bundles, mirrors or uploads model weights. See [Licence and compliance](../natlas/licence.md).

## Dependencies and the supply chain

- The **core install is small** (`typer`, `rich`, `httpx`, `numpy`, `jiwer`, `sacrebleu`) and imports no ML framework. A CI job installs only the core and fails if `torch`, `transformers` or `librosa` appear.
- CI runs `pip-audit` for known vulnerabilities and `gitleaks` for secrets.
- Heavy dependencies live in optional extras and are imported lazily.
- Audio is decoded by calling `ffmpeg` with a **fixed argument list** (no shell), and a decode timeout of 300 seconds by default.

## Reporting a vulnerability

Email the address in `SECURITY.md` (do not open a public issue). The goal is to acknowledge within 72 hours.

!!! note "Placeholder"
    The security contact in `SECURITY.md` is still a placeholder to be replaced before the repository is made public.
