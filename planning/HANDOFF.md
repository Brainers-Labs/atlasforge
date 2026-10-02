# AtlasForge - Handoff (updated 2 Oct 2026)

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

**Verified against real weights on the Mac M1 16 GB (2 Oct; evidence in `planning/21` verification log):**
- All 4 ASR models load and transcribe real FLEURS clips through `atlasforge transcribe --backend local` (Hausa, Yoruba, Igbo; English model loads/runs but was only tested on US-accent audio). `return_timestamps` works (`True` = coarse segments, `"word"` = per-word).
- The LLM generates in en/ha/yo/ig through `atlasforge run --backend openai` against an int4 Ollama import (local only, never publish). Quality of that int4 serve is NOT evidence of official quality.
- Facts that contradict the model card: real context length is 131072 (card says 8,092); repo `generation_config.json` has temperature 0.6 and no repetition_penalty (card recommends 0.1 / 1.12). Ollama's safetensors import drops the chat template; we re-add it (see the log row).

**Still never run:**
- `backends/local.py` for the **LLM** (fp16 does not fit in 16 GB; needs a >=24 GB GPU). Only the ASR path of `local.py` has run for real.
- vLLM serving, fp16/4-bit VRAM and speed numbers (needs the GPU box).
- A real `atlasforge eval` / `compare` on a real dataset (blocked on D026, no data invented).
- A formal ASR WER (only single-sample smoke tests so far).

**Not written yet:** fine-tune recipes (`finetune/`), `atlasforge card` (licence-aware model card), `bench afrobench` wrapper, HTML report, Colab notebooks, docs site, `scripts/live_smoke.py`, Hausa quickstart.

## First thing to do on the new machine (Mac M1 16 GB)
```bash
git clone https://github.com/im-aderm/atlasforge && cd atlasforge
python3 -m venv .venv && . .venv/bin/activate      # Python 3.10-3.13
pip install -e ".[dev]"
brew install ffmpeg
pytest && ruff check . && ruff format --check . && mypy
atlasforge doctor
```
Mac recipe that worked (2 Oct):
1. Accept the licence on all 5 NCAIR1 repos, then `huggingface_hub.login(token=...)`. Steps 2-3 are done; they are kept as the tester recipe.
2. ASR: `pip install -e ".[asr]"` then `atlasforge transcribe clip.wav --lang ha --backend local --model NCAIR1/Hausa-ASR`. On recent macOS the scipy wheel can fail to load; the extras now pin `scipy<1.15` on darwin.
3. LLM: `HF_HUB_DISABLE_XET=1` for the 16 GB download (Xet CDN failed twice), then `ollama create <name> -f Modelfile --quantize int4` with `FROM <snapshot dir>` and re-add the Llama-3.1 `TEMPLATE` (the import drops it), then `--backend openai --base-url http://127.0.0.1:11434/v1 --model <name>`. Never publish quantised N-ATLaS weights (licence).
4. Still to do: run `atlasforge eval` on a real dataset and `compare` two runs (needs D026).

## Blockers and TODOs
1. Placeholders: security email in `SECURITY.md`; GitHub org in `pyproject.toml`; confirm `atlasforge` is free on PyPI (could not check).
2. Human-only Day-1 actions, none done: secure a GPU with a spend cap (needed for the fine-tune demo and the shared beta endpoint); every member accepts the licence on all 5 NCAIR1 repos; email NAIC the open questions (team-size 2-5 vs 1-6, tester evidence format, whether HF-weights integration satisfies verification, public repo required?); post the beta-tester call (target 5 recruits, need >=2 complete); assign stream owners.
3. D026: find a small, clean-licence domain dataset for the flagship compare + fine-tune demo. Do not invent data.
4. Beta testers need something to run: they need either a GPU box/endpoint or the Mac-style llama.cpp path documented in a quickstart.

## Status vs plan
Phase 1 milestone M1 (29 Sep) is now PARTLY met on the model side: all 4 ASR models and the LLM (int4 only) run for real on the Mac; fp16/vLLM still need a GPU. The code side is also the code side is well ahead of plan: the D3-D8 code (eval, metrics, compare, openai backend, ASR audio, validate, CLI) exists and is tested. Remaining code for v0.1: fine-tune recipe, model cards, docs site.

## Rules to keep
Only NCAIR1 models in core paths. Never invent model behaviour (label VERIFIED/INFERRED/UNKNOWN). Never redistribute weights. No secrets/prompts/audio in logs. Core install must not import torch. Never fabricate validation.
