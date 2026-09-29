# Evaluation Engine and External Validation

Two separate topics: (A) the evaluation *product* we ship, and (B) how we validate *AtlasForge itself* with external testers for NAIC.

## A. Evaluation engine (the product)

### Why it's the core

A published AfroBench-LITE study (see [22](22_COMPETITIVE_LANDSCAPE.md)) found N-ATLaS improves strongly over Llama-3-8B-Instruct on Hausa/Yoruba/Igbo, but with a persistent 24–32 point gap to English. It also noted that chrF misses semantic accuracy. Developers need to measure *their own* task, not rely on a leaderboard.

### Metrics shipped in v0.1

| Metric | Library | Use |
|---|---|---|
| exact_match | built-in | QA, extraction |
| accuracy | built-in + label parser | classification / intent |
| chrF, chrF++ | sacrebleu | translation, generation |
| WER, CER | jiwer | ASR |
| latency p50/p95, errors | built-in | operations |
| custom | user callable | domain rules |

### Nigerian-language normalization (differentiator)

`eval/normalize.py`, one rule set per language, all unit-tested:
- Unicode NFC. Yoruba tone marks and underdots can arrive as combining or precomposed characters. Without NFC, identical strings score as different.
- Lowercase and punctuation handling. Keep Hausa hooked letters (ɓ ɗ ƙ ƴ) and apostrophe forms (ʼy).
- **Two views of every text metric**: `tone_aware` (all diacritics kept) and `tone_insensitive` (combining marks stripped after NFD). Always report both, and never silently pick one.
- Normalization config is written into every report.

### Statistical honesty

- `compare` uses paired bootstrap (default 1,000 resamples, fixed seed) and reports the 95% CI of the delta.
- Reports state n, dataset hash, model revision, backend, generation parameters and package version.
- Nothing is called "significant" when the CI crosses zero.

### Starter benchmark packs (`benchmarks/`)

Small, licence-clean subsets (for example 100–200 items per language) to make J1 work out of the box. Each pack has a `PROVENANCE.md` with source, licence and sampling method. Candidates, all to verify:
- FLORES-200 devtest (hau, yor, ibo ↔ eng), CC-BY-SA-4.0: translation / chrF
- SIB-200 or Injongo intent: classification
- FLEURS: ASR / WER

The full AfroBench-LITE suite is run through `lm-evaluation-harness` (P1 wrapper). We don't reimplement it.

## B. External validation of AtlasForge (NAIC requirement: ≥2 testers)

### Who

Target **5 recruits → at least 3 completions → minimum 2 with full evidence.** Priority:
1. NAIC Problem 03 teams (they need `compare` now)
2. NAIC Problem 02 teams (they need `transcribe` + WER)
3. University NLP/ML researchers
4. Independent Nigerian developers

Testers must be outside Brainers Labs and have no ownership stake.

### Rounds

- **Round 1 (Day 7–8)**: J1 on the shared beta endpoint plus local ASR. Goal: find friction while there is still time to fix it.
- **Round 2 (Day 11–12)**: J1 + J2 or J3 on the release candidate. Goal: clean evidence.

### Protocol per tester

1. Receives the repo link, release tag and a beta API key. No live help for the first 20 minutes: they follow the docs alone.
2. Install → `doctor` → `run` → `transcribe` → `eval` on a starter pack → `eval` on **their own** 20+ examples.
3. Optional: `compare` two prompts or models.
4. Fill the feedback form. 15-minute debrief call (recorded only with consent).

### Record per tester (`validation/testers/T0X.md`)

Date, tester role/org (with consent), OS, Python version, GPU/CPU, release tag, time-to-first-success per step, errors hit (verbatim), doc gaps, SUS-style 5-question usability score, "would you use this in your NAIC project?" (Y/N + why), issue links.

### Metrics we report

Time to first `run`, time to first eval report, setup success rate, number of issues filed and fixed between rounds, reuse intent. Real numbers only, including the bad ones.

### Evidence kept

Anonymised terminal logs, screenshots (no keys visible), feedback forms, GitHub issues, report files produced by testers, and the exact release tag tested. Never fabricate or "tidy up" validation.
