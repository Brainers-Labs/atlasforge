# Execution Plan — 28 Sep → 12 Oct 2026

**Day 1 = today, Monday 28 September. Internal submission: Sunday 11 October (Day 14). Monday 12 October is buffer only.**

## Workstreams

Assumes 3 people. If you have fewer, merge streams in this order: C into A first, then B into A.

| Stream | Owner | Scope |
|---|---|---|
| **A — Core** | _TBD_ | package, backends, eval engine, metrics/normalization, compare, CLI, tests, CI |
| **B — Models & infra** | _TBD_ | GPU box, live verification, ASR module, vLLM beta endpoint, fine-tune recipes, live evidence |
| **C — Product & evidence** | _TBD_ | NAIC comms, beta recruitment and sessions, docs site, benchmark packs + provenance, Hausa quickstart, video, submission |

## Milestones (these are the dates that matter)

| # | Date | Exit criteria |
|---|---|---|
| **M1** | Tue 29 Sep (D2) | N-ATLaS + all 4 ASR models run on our GPU. [21](21_NATLAS_DISCOVERY.md) UNKNOWNs resolved. Repo public-ready skeleton with green CI. ≥5 beta candidates contacted. |
| **M2** | Fri 2 Oct (D5) | `atlasforge eval` end-to-end on `local` and `openai` backends with a real starter pack. Beta endpoint live with per-tester keys. |
| **M3** | Sun 4 Oct (D7) | Tag `v0.1.0a1`. Quickstart docs good enough for a stranger. **Beta Round 1 starts.** |
| **M4** | Wed 7 Oct (D10) | `compare`, `transcribe` with chunking, ASR eval, Round 1 fixes merged, fine-tune LLM recipe smoke-tested. Tag `v0.1.0rc1`. |
| **M5** | Fri 9 Oct (D12) | Beta Round 2 complete (≥2 testers with full evidence). Live evidence file generated. |
| **M6** | Sun 11 Oct (D14) | `v0.1.0` on PyPI + GitHub, video uploaded, **submission sent**. |

## Day by day

### D1 — Mon 28 Sep (today)
- **All (first hour):** read [00](00_README.md), [15](15_DECISION_LOG.md), [22](22_COMPETITIVE_LANDSCAPE.md). Confirm the name and owners. Each person creates a Hugging Face account and **accepts the licence on all 5 NCAIR1 repos** (it can take time to be approved).
- **A:** check `atlasforge` on PyPI and GitHub. Create the repo (private until D7). Set up `pyproject` (hatch), src layout, typer CLI stub, ruff/mypy/pytest, GitHub Actions matrix, gitleaks, Apache-2.0, `.gitignore`. Write `types.py` and the `Backend` protocol.
- **B:** **secure a GPU today** (≥24 GB: RunPod / Lambda / Modal / Colab Pro+ / university; set a budget cap). Download the weights. Run the model-card snippets for the LLM and all 4 ASR models. Record the results in [21](21_NATLAS_DISCOVERY.md).
- **C:** email NAIC the open questions in [02](02_NAIC_REQUIREMENTS.md). Draft the beta call. Post it in NAIC participant channels, DSN community, university NLP groups and X/LinkedIn. Build the tester tracker sheet and feedback form.

### D2 — Tue 29 Sep
- **A:** `openai` backend + contract fixtures. `local` backend (lazy import, chat template, 4-bit option). `atlasforge run`.
- **B:** verify the chat template, context length (`config.json`), fp16 vs 4-bit VRAM, and speed. Try `vllm serve NCAIR1/N-ATLaS` and record whether it works as-is. Test ASR `return_timestamps`. → **M1**
- **C:** follow up with candidates. Aim for 5 committed. Collect FLORES-200 / SIB / FLEURS licence details and write `benchmarks/*/PROVENANCE.md`.

### D3 — Wed 30 Sep
- **A:** `eval/dataset.py` (JSONL + validation errors with line numbers), `eval/runner.py` (resumable, latency), `results.jsonl`.
- **B:** `asr/audio.py` (ffmpeg → 16 kHz mono), model routing, `atlasforge transcribe` for inputs ≤30 s.
- **C:** build starter packs: FLORES translation 100 items × (ha, yo, ig) ↔ en, plus one classification pack. **Start sourcing the flagship domain dataset** (a small, clean-licence Hausa/Yoruba/Igbo domain QA or classification set for the compare + fine-tune demo, D026). Report candidates and licences by D5.

### D4 — Thu 1 Oct
- **A:** `eval/normalize.py` + metrics (EM, accuracy, chrF, WER/CER, tone-aware/insensitive) with the must-have tests from [10](10_TESTING_STRATEGY.md).
- **B:** stand up the beta vLLM endpoint: HTTPS, per-tester keys, rate limit, spend cap, auto-shutdown.
- **C:** mkdocs skeleton. Write the Quickstart (Path A endpoint / Path B local) and the Licences & access page.

### D5 — Fri 2 Oct
- **A:** `report.py` (JSON, Markdown, rich terminal). `atlasforge eval` CLI. End-to-end on both backends. → **M2**
- **B:** first real baseline eval runs on all starter packs. Save the outputs as example reports.
- **C:** schedule Round 1 sessions for D7–D8. Send testers pre-reading (accept the HF licences early).

### D6 — Sat 3 Oct
- **A:** `atlasforge doctor` (env, GPU, disk, ffmpeg, token, gated status per repo, endpoint). Error hierarchy and hints. `atlasforge dataset validate` (schema, duplicates, leakage, Unicode health, diacritic stats).
- **B:** `asr/chunking.py` (>30 s, overlap merge) + tests. ASR eval with a FLEURS subset.
- **C:** dry-run the Quickstart yourself on a clean machine (Windows + Linux). Fix the docs.

### D7 — Sun 4 Oct
- **A + B:** bug bash, clean install tests, tag `v0.1.0a1`, make the repo public. → **M3**
- **C:** **Round 1 sessions begin.** Log every friction point as a GitHub issue.

### D8 — Mon 5 Oct
- **A:** `compare/` (paired bootstrap + McNemar, slice analysis with per-slice CIs and n≥30 rule, failure-mode flags, Markdown/JSON report) + CLI.
- **B:** `finetune/llm_qlora.py` + YAML config. Start a small QLoRA smoke run (e.g. 1–2k examples, one epoch) on the GPU box.
- **C:** finish Round 1. Triage issues into must-fix / later.

### D9 — Tue 6 Oct
- **A:** fix Round 1 must-fix issues. Resume/robustness.
- **B:** run `compare` of base vs smoke adapter as the flagship example. `finetune asr` recipe (config + notebook; a full training run only if GPU time allows).
- **C:** Problem 03 guide + Problem 02 guide. Hausa Quickstart translation with a named reviewer.

### D10 — Wed 7 Oct
- **A:** `atlasforge card` (license-aware model card). API reference docs.
- **B:** `scripts/live_smoke.py` → `evidence/live_<date>.md`. Colab notebooks J1/J2.
- **C:** tag `v0.1.0rc1`. Brief Round 2 testers. → **M4**

### D11 — Thu 8 Oct
- **All:** **Beta Round 2** (J1 + J2 or J3 on rc1). A/B on call for fixes, with no scope additions.
- **C:** write the video script (see [19](19_DEMO_STORY.md)).

### D12 — Fri 9 Oct
- **All:** finish Round 2. Write up `validation/testers/T0X.md` with real metrics. → **M5**
- **B:** record the demo terminal segments on the GPU box.

### D13 — Sat 10 Oct
- **A:** release candidate fixes only. Final docs pass. Troubleshooting page from real beta errors.
- **C:** edit the video (3–5 min). Submission docs: integration evidence, architecture, validation summary, team profile, CAC/endorsement.

### D14 — Sun 11 Oct — ship
- Fresh-machine install test → tag `v0.1.0` → publish to PyPI → docs live → secret scan → run through the [16](16_SUBMISSION_CHECKLIST.md) checklist → **submit**.
- Shut down or restrict the beta endpoint as recorded in [09](09_SECURITY.md).

### D15 — Mon 12 Oct
Buffer only. Deadline 11:59 PM WAT.

## Cut line (if behind schedule, drop from the bottom up)

1. Real N-ATLaS integration + live evidence (**never cut**)
2. `eval` with NG normalization + reports
3. External beta ×2 with evidence (**never cut**)
4. CLI `doctor` / `run` / `eval`
5. Quickstart docs
6. `transcribe` + ASR eval
7. `compare`
8. Fine-tune LLM recipe (a documented config and notebook is acceptable without a big training run)
9. `card`
10. Hausa quickstart
11. `dataset validate` (cut before the items above it if forced)
12. Failure-mode flags
13. ASR fine-tune recipe
14. HTML report
15. AfroBench harness wrapper

## Daily rhythm

- 09:00 WAT 15-minute stand-up: yesterday / today / blockers, measured against the next milestone.
- End of day: each stream pushes, updates [15](15_DECISION_LOG.md) if a decision was made, and moves the milestone checkboxes.
- Any milestone missed by more than 1 day → invoke the cut line immediately. Don't hope.

## Budget to approve today

GPU: ~24 GB class for ~14 days, part-time (dev plus beta endpoint). Check the providers' current prices. Set a hard cap in the provider console before starting.
