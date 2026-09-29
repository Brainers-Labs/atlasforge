# Security, Privacy and Licence Compliance

## Secrets

Never commit, print, screenshot or log: `HF_TOKEN`, endpoint API keys, cloud GPU credentials.
- `.env` in `.gitignore` from the first commit. `gitleaks` in CI and as a pre-commit hook.
- Tokens are shown in `doctor` only as `hf_****abcd`.

## Logs and data

Default logs contain no prompts, completions, audio or tokens. `--log-samples` is explicit opt-in.
`results.jsonl` does contain predictions (that's its purpose), so the docs warn users not to commit results that come from sensitive datasets.

## Audio

- Treat voice data as personal data (Nigeria Data Protection Act 2023 applies to anything we collect ourselves).
- Beta and demo recordings need explicit consent. Keep transcripts, delete raw audio after use, and never commit raw tester audio.

## Shared beta endpoint (temporary infrastructure)

- One vLLM instance behind an API key per tester (a reverse proxy or vLLM `--api-key`; per-tester keys via a small proxy is preferred so keys can be revoked individually).
- HTTPS only. Rate limit per key. **Hard spending cap** on the GPU provider. Auto-shutdown schedule.
- No request logging of prompt content. Only counts and latency.
- Decommission after Round 2 and record the date.

## Dependencies

- Pin versions in lock files for CI. Keep the core install small and put heavy dependencies in extras.
- `pip-audit` in CI. Licence check of all dependencies (all must be compatible with Apache-2.0).

## N-ATLaS licence compliance (VERIFIED terms from model cards, re-check before release)

- AtlasForge **never redistributes** NCAIR1 weights. Users download them under their own accepted licence.
- Docs state clearly: 1,000 active end-user cap; commercial use needs a separate agreement with Awarri / the Ministry; attribution required.
- `atlasforge card` (P1) writes the required attribution and "Powered by Awarri" naming guidance into every adapter model card.
- We do **not** publish quantized or merged weights in v0.1.
- Check whether Llama 3 Community License obligations also flow through (base model). INFERRED likely; to verify.

## Open-source hygiene

`SECURITY.md` with a private disclosure email and 72-hour acknowledgement target, Dependabot enabled, signed release tags.
