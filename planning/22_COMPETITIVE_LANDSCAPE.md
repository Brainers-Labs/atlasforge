# Competitive Landscape

Checked 28 Sep 2026. Re-check weekly. Other NAIC teams are shipping in parallel.

## Community "N-ATLAS Kit" (another NAIC 2026 Problem 01 entry)

- Repo: https://github.com/Kambah123/N-ATLAS-Kit (states it targets NAIC 2026 Problem Statement 1)
- PyPI: `natlas` v0.1.0 (25 Sep 2026), npm: `n-atlas`, docs: natlas-docs.vercel.app, playground: natlas-playground.vercel.app
- Author: OneDev Studioo. Apache-2.0.

What they have (per their README/PyPI, not independently tested):
- Python + JS SDKs (httpx, pydantic), chat with streaming, transcription, translation/summarisation helpers, voice chat
- `/serve` gateway: vLLM (OpenAI-compatible LLM) + FastAPI ASR (`/v1/audio/transcriptions`, 4-model routing, ffmpeg conversion, silence-aware chunking >30 s). Docker Compose / Modal deploy.
- Playground, English + Hausa docs, example apps

What they don't have (as of 28 Sep):
- CLI ("planned")
- Local `transformers` backend ("later milestone")
- **Any evaluation, benchmarking, comparison, or Nigerian-language-aware metrics**
- **Fine-tuning tooling**
- Licence-aware model cards

## Implications for us

1. **Don't build a second SDK/gateway/playground.** We would be later and weaker, and judges would see a duplicate.
2. **Our lane is evaluation, comparison and fine-tuning.** It's listed in Problem 01, unserved, and needed by every Problem 02/03 team.
3. **Interoperate.** Our `openai` backend can target their gateway. Document it. This makes AtlasForge useful to their users too, and shows ecosystem thinking to judges (Scalability & Sustainability).
4. **Distinct name** (AtlasForge), so nobody confuses the two projects.
5. Stay professional: no disparaging comparisons in public docs or the video. A factual "works with" section only.

## General tools (not competitors, building blocks)

- `lm-evaluation-harness` (EleutherAI): research benchmarks. We wrap it for AfroBench-LITE and don't replace it.
- `vLLM`, HF TGI: serving
- `jiwer`, `sacrebleu`: metric primitives
- `peft`, `trl`: fine-tuning primitives

AtlasForge's value is the N-ATLaS-specific, Nigerian-language-aware glue and the evidence-quality reports on top of these.
