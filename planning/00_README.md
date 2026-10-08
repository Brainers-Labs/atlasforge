# AtlasForge — Brainers Labs

**Working name:** AtlasForge (formerly "N-ATLAS Kit" — renamed, see [22](22_COMPETITIVE_LANDSCAPE.md))
**Python package / CLI:** `atlasforge` (availability to confirm on Day 1)
**Company:** Brainers Labs
**NAIC track:** Problem Statement 01 — Developer Infrastructure ("Build the tools that make N-ATLAS easy to build with")
**Deadline:** Monday 12 October 2026, 11:59 PM WAT. Internal target: submit Sunday 11 October.

## Product in one sentence

> AtlasForge is an open-source toolkit to **run, evaluate, fine-tune and prove** N-ATLaS models — LLM and ASR — from one CLI and Python library.

## Why this shape

1. **N-ATLaS is open weights, not an API.** It is published as gated Hugging Face models (`NCAIR1/N-ATLaS` plus four Whisper-Small ASR models). NAIC gives API credentials only to *shortlisted* teams. Every developer today has to download, load, serve and evaluate the models themselves. Details: [21](21_NATLAS_DISCOVERY.md).
2. **Another team has already shipped the "SDK + gateway + playground" lane**, under the name N-ATLAS Kit and the PyPI name `natlas`. Competing there means doing the same thing, later. Details: [22](22_COMPETITIVE_LANDSCAPE.md).
3. **Nobody has built the evaluation or fine-tuning tooling yet.** NAIC lists both as Problem 01 examples. Every Problem 03 team (Sectoral Fine-Tuning) must fine-tune N-ATLAS *and* show improvement over the base model on 500+ examples. Every Problem 02 team (Voice-First) must use the official ASR. AtlasForge is the tooling those teams need. They are also our beta testers.

## v0.1 scope (14 days)

| Priority | Component | What it does |
|---|---|---|
| P0 | Backends | Run N-ATLaS via local `transformers` (fp16 / 4-bit) or any OpenAI-compatible endpoint (vLLM, HF Endpoint, third-party gateways) |
| P0 | ASR | Official NCAIR1 ASR models with >30 s chunking, audio conversion, language routing |
| P0 | Eval engine | JSONL tasks, Nigerian-language-aware metrics (tone-aware and tone-insensitive WER/CER/chrF), latency, JSON/Markdown reports |
| P0 | Compare | Base vs candidate with paired bootstrap CIs, **slice analysis** (where it improved, where it regressed, with n≥30 rule) and failure-mode flags, for Problem 03 "measurable improvement" evidence |
| P0 | Dataset validate | Schema, duplicates, train/test leakage, Unicode/diacritic health |
| P0 | CLI | `doctor`, `run`, `transcribe`, `eval`, `compare`, `report` |
| P1 | Fine-tune starter kit | QLoRA recipe for N-ATLaS, fine-tune recipe for the Whisper-Small ASR models, Colab notebooks |
| P1 | License-aware model cards | Auto-generate attribution / "Powered by Awarri" / user-cap notices on fine-tuned outputs |
| P1 | AfroBench-LITE runner | Thin wrapper over `lm-evaluation-harness` to reproduce the published N-ATLaS numbers. Delivered 7 Oct as `atlasforge bench afrobench`: it measures them here rather than copying the study's figures, which stay in [21](21_NATLAS_DISCOVERY.md) |
| Out | Playground, JS SDK, gateway | Already exist elsewhere, so we interoperate instead |

## Non-negotiables

- Only official NCAIR1 weights in the core path. No other foundation model, ever.
- Never invent model behaviour, metrics, limits or validation. Label facts VERIFIED / INFERRED / UNKNOWN.
- Never redistribute N-ATLaS weights. Users accept the gated license with their own Hugging Face account.
- Real external beta testers, with dated evidence.

## Start here

- Execution plan (today onward): [11_14_DAY_EXECUTION_PLAN.md](11_14_DAY_EXECUTION_PLAN.md)
- Verified model facts: [21_NATLAS_DISCOVERY.md](21_NATLAS_DISCOVERY.md)
- Full index: [INDEX.md](INDEX.md)
