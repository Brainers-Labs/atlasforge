# Decision Log

Status: ✅ active · 🔁 superseded · ❓ pending

## Original decisions

| ID | Decision | Status | Note |
|---|---|---|---|
| D001 | Apply to NAIC Problem 01 — Developer Infrastructure | ✅ | |
| D002 | Keep Breno separate from the NAIC project | ✅ | Commercial product vs open infrastructure |
| D003 | Open source | ✅ | Refined by D016 |
| D004 | Working name "N-ATLAS Kit" | 🔁 by D012 | Name taken |
| D005 | "Whisper for N-ATLAS" as internal analogy only | 🔁 by D015 | ASR models *are* Whisper fine-tunes |
| D006 | Python first, JS/TS later | ✅ | JS now out of scope (D014) |
| D007 | Include evaluation | 🔁 by D013 | Evaluation is now the core, not an add-on |
| D008 | ≥2 external beta testers | ✅ | Refined by D018 |
| D009 | Verify before coding | ✅ | |
| D010 | Scope ruthlessly | ✅ | Cut line in [11](11_14_DAY_EXECUTION_PLAN.md) |

## 28 Sep 2026 — pivot decisions

**D011 — Integrate via official Hugging Face weights, not an API.** ✅
No public API exists, and NAIC API credentials go to shortlisted teams only. Backends: `local` (transformers) and `openai` (any OpenAI-compatible server, e.g. vLLM). An `official` backend gets added post-shortlist.

**D012 — Rename to AtlasForge (`atlasforge`).** ✅ (pending name availability check)
"N-ATLAS Kit" and `natlas` are used by another NAIC entry. A clearly distinct name also avoids implying government endorsement.

**D013 — Evaluation, comparison and fine-tuning are the product core.** ✅
This is the unfilled gap in the ecosystem, it's explicitly listed in Problem 01 examples, and it directly serves every Problem 02/03 team.

**D014 — Out of scope for v0.1: playground, JS SDK, gateway.** ✅
They already exist in the community. We interoperate through the OpenAI-compatible backend.

**D015 — ASR = practical tooling over the official Whisper-Small checkpoints** (conversion, chunking, routing, WER). No new ASR model, and no invented timestamps or confidence. ✅

**D016 — Apache-2.0 for our code. Never redistribute weights. No quantized/merged weight publishing in v0.1.** ✅
The Awarri licence governs weights. Users accept it themselves. Adapters produced by users get licence-aware model cards.

**D017 — Tone-aware and tone-insensitive scoring always reported together.** ✅
Normalization choices change Yoruba/Igbo scores materially. Hiding the choice would be misleading.

**D018 — Beta: recruit 5 from Day 1, two rounds (D7–8, D11–12), priority NAIC Problem 02/03 teams.** ✅

**D019 — Temporary shared vLLM beta endpoint** with per-tester keys, spend cap, and decommissioning after Round 2. ✅

**D020 — Planning docs move to `planning/` when the product repo is created;** `docs/` is reserved for user documentation. ❓

## 28 Sep 2026 — refinement after external review

**D021 — Product question: "I changed this N-ATLaS model. Did I actually make it better on my task?"** ✅
The pipeline is Prepare (validate data) → Fine-tune (one recipe) → Evaluate → Compare → Understand. Speech is a specialised evaluation pipeline, not a separate product.

**D022 — Slice analysis is a P0 part of `compare`, with per-slice CIs and a 30-example minimum.** ✅
Slices under 30 examples are reported as "insufficient data". A headline regression claim on a tiny slice would mislead.

**D023 — No "hallucination rate" and no LLM-as-judge.** ✅
It would need a judge model. A non-N-ATLaS judge risks NAIC disqualification and N-ATLaS judging itself is unreliable. We ship deterministic failure-mode flags and label them as flags.

**D024 — `dataset validate` added (P0-lite): schema, duplicates, train/test leakage, Unicode health, diacritic statistics.** ✅
No language-identification claims.

**D025 — HTML report demoted to P1; ASR analysis limited to alignment-based top errors and WER by length.** ✅

**D026 — Flagship demo needs a real, clean-licence domain dataset.** ❓ Stream C to source or build it by D5. No invented data.

## Pending (decide by end of D1)

- ❓ Team roster and stream owners (A/B/C)
- ❓ GPU provider and budget cap
- ❓ Final name after the PyPI/GitHub/trademark check
- ❓ GitHub org: `brainerslabs/atlasforge`?
