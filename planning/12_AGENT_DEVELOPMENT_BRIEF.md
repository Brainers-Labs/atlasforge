# Agent Development Brief

The full context for an AI coding agent (or a new engineer) working on AtlasForge. [13](13_MASTER_AGENT_PROMPT.md) is the short prompt that points here.

## Mission

Build AtlasForge v0.1: an open-source Python toolkit and CLI to run, evaluate, compare and fine-tune the official N-ATLaS models (`NCAIR1/*` on Hugging Face), with Nigerian-language-aware metrics.

## Read before coding

1. [21_NATLAS_DISCOVERY.md](21_NATLAS_DISCOVERY.md): what is VERIFIED about the models
2. [03_PRD.md](03_PRD.md): requirements and scope
3. [04_ARCHITECTURE.md](04_ARCHITECTURE.md) + [05_SDK_API_DESIGN.md](05_SDK_API_DESIGN.md): structure and interfaces
4. [10_TESTING_STRATEGY.md](10_TESTING_STRATEGY.md): what must be tested
5. [11_14_DAY_EXECUTION_PLAN.md](11_14_DAY_EXECUTION_PLAN.md): which milestone you are serving

## Non-negotiables

- Only `NCAIR1/*` models in any core or example path. Tiny public models are allowed **only** in CI wiring tests and must never be presented as N-ATLaS results.
- Never invent model behaviour (timestamps, confidence, language detection, limits, metrics). If it's not in [21](21_NATLAS_DISCOVERY.md) as VERIFIED, verify it and update the doc, or don't expose it.
- Never download, bundle or redistribute weights in the package or repo.
- Core install must not import torch. Heavy dependencies sit behind extras, imported lazily.
- No secrets, prompts or audio in logs by default.
- Every public function has type hints and a test. CLI stays thin over the Python API.
- Do not add features outside the current milestone. Scope creep is the main risk.

## Stack

Python 3.10–3.13 · hatch · typer + rich · httpx · pydantic v2 · jiwer · sacrebleu · (extras) torch, transformers, accelerate, bitsandbytes, librosa/soundfile, peft, trl · pytest · ruff · mypy · mkdocs-material.

## Unknowns protocol

1. Check [21](21_NATLAS_DISCOVERY.md). 2. Check the model card / `config.json` / `tokenizer_config.json`. 3. Test on the GPU box. 4. Record the evidence in [21](21_NATLAS_DISCOVERY.md) with date and status. 5. Only then implement.

## Stop and ask a human if

- gated access or GPU is unavailable
- a licence term is unclear or seems to forbid something we plan (e.g. the endpoint, publishing adapters)
- a requirement would need a non-N-ATLaS model
- a milestone is going to slip by more than 1 day

## Engineering standard

`simple → tested → documented → extensible` over `complex → impressive-looking → unfinished`.
