# AtlasForge - Handoff (29 Sep 2026)

## What this is
Brainers Labs' NAIC 2026 Problem 01 entry. Open-source Python toolkit + CLI to **run, evaluate, compare and fine-tune the official N-ATLaS models** (`NCAIR1/*` on Hugging Face), with tone-aware and tone-insensitive scoring for Hausa/Yoruba/Igbo/Nigerian English.
Answers: "I changed this N-ATLaS model. Did I actually make it better on my task, and where did it get worse?"
Deadline: **12 Oct 2026, 11:59 PM WAT** (internal submit 11 Oct). Track: Developer Infrastructure, needs >=2 real external beta testers.

## Why this shape (short)
- N-ATLaS has **no public API**: gated open weights on HF. NAIC gives API credentials only to shortlisted teams. Licence: Awarri custom (1,000 active-user cap, attribution, "Powered by Awarri" on derivatives, commercial use needs agreement).
- Another NAIC team already ships "N-ATLAS Kit" (`natlas` on PyPI, GitHub Kambah123/N-ATLAS-Kit): SDKs, vLLM+FastAPI gateway, playground. It has **no** eval, compare or fine-tune tooling. So we renamed to AtlasForge and own that gap. We interoperate through an OpenAI-compatible backend, we do not build an SDK/gateway/playground.
- No LLM-as-judge and no "hallucination rate" (a non-N-ATLaS judge risks disqualification). Deterministic failure flags instead.

## Where everything is
Everything now lives in ONE git repo (this folder's parent).
- `planning/` - 23 planning docs (was `docs/`; that name is reserved for user docs). Start with `planning/INDEX.md`, then `11_14_DAY_EXECUTION_PLAN.md` (day-by-day, milestones M1-M6), `15_DECISION_LOG.md` (D001-D026), `21_NATLAS_DISCOVERY.md` (verified model facts + a blank verification log to fill on a GPU). Inside those docs, `docs/NN_...` paths mean `planning/NN_...`.
- `src/atlasforge/`, `tests/` - the product (src layout, ruff/mypy strict, pytest, CI, pre-commit+gitleaks).

## Code state (updated 29 Sep): VERIFIED on Windows, Python 3.12
`pytest` 207 passed, 96% coverage; `ruff check`, `ruff format --check` and `mypy` (strict) all clean. Not yet run on Linux/macOS or Python 3.10/3.13 (CI matrix will do it once pushed).
Written: `errors.py`, `types.py`, `security.py`, `backends/base.py` (Protocol), `eval/normalize.py`, `eval/dataset.py`, `eval/runner.py`, `doctor.py`, `cli.py` (`doctor`, `--version`), and 8 test files under `tests/unit/`.
Not written yet: metrics (EM, accuracy, chrF, WER/CER), scoring/report, `compare`, backends (`local`, `openai`), ASR, `dataset validate`, fine-tune recipe.

## First thing to do on the new machine
```bash
git clone <repo-url> && cd atlasforge
python -m venv .venv          # Python 3.10-3.13
# activate: .venv\Scripts\activate (Windows) or . .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy
```
Also install ffmpeg (needed for ASR audio): `winget install Gyan.FFmpeg` / `brew install ffmpeg` / `apt install ffmpeg`.

## Blockers and TODOs
1. LICENSE (Apache-2.0) is now in the repo. Still placeholders: security email in `SECURITY.md`; GitHub org in `pyproject.toml`; confirm `atlasforge` is free on PyPI (could not check).
2. Day-1 actions only humans can do, none done yet: secure a >=24 GB GPU with a spend cap; every member accepts the licence on all 5 NCAIR1 HF repos; email NAIC the open questions (team-size 2-5 vs 1-6, tester evidence format, whether HF-weights integration satisfies verification, public repo required?); post the beta-tester call (target 5 recruits, need >=2 complete); assign stream owners A/B/C.
3. Then Stream B: run all 5 models and fill the log in `docs/21_NATLAS_DISCOVERY.md` (chat template, real context length, VRAM fp16 vs 4-bit, vLLM serve, ASR `return_timestamps`, commit SHAs).
4. D026: find a small, clean-licence domain dataset for the flagship compare + fine-tune demo. Do not invent data.

## Status vs plan
Phase 1 (Foundation), Day 1-2, milestone M1 (Tue 29 Sep): NOT met. Needs GPU + models running, doc 21 unknowns resolved, green CI, >=5 testers contacted.
Next code (Day 3-4): metrics + scoring + report, then `openai`/`local` backends.

## Rules to keep
Only NCAIR1 models in core paths. Never invent model behaviour (label VERIFIED/INFERRED/UNKNOWN). Never redistribute weights. No secrets/prompts/audio in logs. Core install must not import torch. Never fabricate validation.
