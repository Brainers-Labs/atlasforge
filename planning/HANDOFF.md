# AtlasForge - Handoff (updated 29 Sep 2026)

## What this is
Brainers Labs' NAIC 2026 Problem 01 entry. Open-source Python toolkit + CLI to **run, evaluate, compare and fine-tune the official N-ATLaS models** (`NCAIR1/*` on Hugging Face), with tone-aware and tone-insensitive scoring for Hausa/Yoruba/Igbo/Nigerian English.
Answers: "I changed this N-ATLaS model. Did I actually make it better on my task, and where did it get worse?"
Deadline: **12 Oct 2026, 11:59 PM WAT** (internal submit 11 Oct). Track: Developer Infrastructure, needs >=2 real external beta testers.

## Why this shape (short)
- N-ATLaS has **no public API**: gated open weights on HF. NAIC gives API credentials only to shortlisted teams. Licence: Awarri custom (1,000 active-user cap, attribution, "Powered by Awarri" on derivatives, commercial use needs agreement).
- Another NAIC team already ships "N-ATLAS Kit" (`natlas` on PyPI, GitHub Kambah123/N-ATLAS-Kit): SDKs, vLLM+FastAPI gateway, playground. It has **no** eval, compare or fine-tune tooling. So we renamed to AtlasForge and own that gap. We interoperate through an OpenAI-compatible backend, we do not build an SDK/gateway/playground.
- No LLM-as-judge and no "hallucination rate" (a non-N-ATLaS judge risks disqualification).

## Where everything is
ONE git repo: https://github.com/im-aderm/atlasforge (private).
- `planning/` - planning docs. Start with `INDEX.md`, then `11_14_DAY_EXECUTION_PLAN.md`, `15_DECISION_LOG.md`, `21_NATLAS_DISCOVERY.md` (verified model facts + a blank verification log to fill on a machine with the models). Inside those docs, `docs/NN_...` paths mean `planning/NN_...`.
- `src/atlasforge/`, `tests/` - the product.

## Code state (29 Sep): 558 tests, 98% coverage; ruff + mypy strict clean
Built and tested on Windows / Python 3.12; the CI matrix covers Linux, macOS and Windows on Python 3.10 and 3.13 (it caught one 3.10-only numpy typing issue, fixed).

**Works and is tested (no model needed):**
- `eval/`: strict JSONL loader, resumable runner (manifest guard, crash-safe, circuit breaker, bounded threading), metrics (EM, accuracy, macro-F1, chrF/chrF++, WER, CER), scoring under tone-aware + tone-insensitive views (failures count as wrong), reports, `validate` (duplicates, leakage, Unicode, diacritics, class balance), `normalize`.
- `compare/`: paired bootstrap, exact McNemar, slice analysis (n>=30 rule), Markdown/JSON report.
- `backends/openai.py`: OpenAI-compatible HTTP backend, tested end-to-end against a real (fake-model) HTTP server.
- `asr/`: ffmpeg decode (incl. real opus/ogg), 30 s windowing, seam-aware merge, `LongAudioBackend`.
- CLI: `doctor`, `run`, `transcribe`, `eval`, `report`, `compare`, `dataset validate`.

**Written but NEVER run against real weights (do this on the Mac first):**
- `backends/local.py` (transformers). Only tested with stand-in torch/transformers modules.
- The official ASR models (loading them, real WER).
- Open questions for `planning/21`: chat template present? real context length? `return_timestamps` behaviour? VRAM/speed?

**Added 2 Oct (tested with stand-ins, 679 tests total):** `finetune/` (QLoRA config, data checks, `--dry-run`, TRL argument-name detection, `training_run.json`), `cards/` + `atlasforge card` (licence-aware model card from `comparison.json` / `training_run.json`), and `--adapter` on the local backend. **The training run itself has never executed: it needs an NVIDIA GPU.**

**Not written yet:** `bench afrobench` wrapper, HTML report, Colab notebooks, the user docs site (deferred by the team: "we will document later"), `scripts/live_smoke.py`, Hausa quickstart, `examples/` and benchmark packs.

## First thing to do on the new machine (Mac M1 16 GB)
```bash
git clone https://github.com/im-aderm/atlasforge && cd atlasforge
python3 -m venv .venv && . .venv/bin/activate      # Python 3.10-3.13
pip install -e ".[dev]"
brew install ffmpeg
pytest && ruff check . && ruff format --check . && mypy
atlasforge doctor
```
Then, to get real evidence:
1. Accept the licence on all 5 NCAIR1 repos (Hugging Face), `export HF_TOKEN=...`.
2. **ASR (fits in memory, CPU/MPS):** `pip install -e ".[asr]"`, then try `atlasforge transcribe some.ogg --lang ha --backend local`. Record the result in `planning/21`.
3. **LLM on the Mac:** fp16 (16 GB) will not fit; 4-bit via bitsandbytes needs CUDA. Serve a quantised model with llama.cpp or Ollama (INFERRED to work for a Llama-3-8B fine-tune, untested) and point `--backend openai --base-url http://127.0.0.1:PORT/v1` at it. Never publish quantised N-ATLaS weights (licence).
4. Run `atlasforge eval` on a real dataset and `compare` two runs. Fill in the verification log in `planning/21`.

## Blockers and TODOs
1. Placeholders: security email in `SECURITY.md`; GitHub org in `pyproject.toml`; confirm `atlasforge` is free on PyPI (could not check).
2. Human-only Day-1 actions, none done: secure a GPU with a spend cap (needed for the fine-tune demo and the shared beta endpoint); every member accepts the licence on all 5 NCAIR1 repos; email NAIC the open questions (team-size 2-5 vs 1-6, tester evidence format, whether HF-weights integration satisfies verification, public repo required?); post the beta-tester call (target 5 recruits, need >=2 complete); assign stream owners.
3. D026: find a small, clean-licence domain dataset for the flagship compare + fine-tune demo. Do not invent data.
4. Beta testers need something to run: they need either a GPU box/endpoint or the Mac-style llama.cpp path documented in a quickstart.

## Status vs plan
Phase 1 milestone M1 (29 Sep) is NOT met on the model side (no GPU, no model run yet) but the code side is well ahead of plan: the D3-D8 code (eval, metrics, compare, openai backend, ASR audio, validate, CLI) exists and is tested. Remaining code for v0.1: fine-tune recipe, model cards, docs site.

## Rules to keep
Only NCAIR1 models in core paths. Never invent model behaviour (label VERIFIED/INFERRED/UNKNOWN). Never redistribute weights. No secrets/prompts/audio in logs. Core install must not import torch. Never fabricate validation.
