# Conversation Record

## Session 1 — initial direction (before 28 Sep 2026)

- The user shared the NAIC 2026 page and received a briefing on the three problem statements, validation thresholds, submission components, criteria and timeline.
- Brainers Labs wants to build an open-source developer tool. A "Whisper for N-ATLAS" analogy was discussed.
- Recommendation: build **N-ATLAS Kit** (SDK, CLI, ASR interface, evaluation, playground) as reusable infrastructure rather than bolting N-ATLAS onto Breno or iSchool.
- Planning docs 00–20 were written, assuming a hosted N-ATLAS API (`api_key`, endpoints, rate limits).

## Session 2 — doc review and pivot (28 Sep 2026)

Review of all planning docs, with research against primary sources. Findings:

1. **No public N-ATLAS API.** N-ATLaS is gated open weights on Hugging Face (`NCAIR1/N-ATLaS`, Llama-3 8B). The NAIC page confirms API credentials go to *shortlisted* teams only. The API-client architecture was built on a false premise.
2. **The official ASR models are Whisper-Small fine-tunes** (Hausa, Yoruba, Igbo, Nigerian-accented English), with a 30 s input limit. The "not a Whisper clone" framing no longer applied.
3. **Custom licence**: Awarri Open-Source Research and Innovation License. 1,000 active end-user cap, commercial use needs an agreement, attribution, "Powered by Awarri" suffix for derivatives, gated access.
4. **A competing NAIC entry already exists** named "N-ATLAS Kit" (GitHub `Kambah123/N-ATLAS-Kit`), with `natlas` on PyPI (v0.1.0, 25 Sep 2026) and `n-atlas` on npm. It ships Python/JS SDKs, a vLLM + FastAPI gateway with ASR chunking, a playground and bilingual docs. It has no CLI, no local backend and no evaluation tooling.
5. A published AfroBench-LITE evaluation shows large gains over Llama-3-8B-Instruct but a persistent gap to English, and notes that chrF has limits.

Decisions taken: rename to **AtlasForge**; reposition around evaluation, comparison and fine-tuning; backend-agnostic design; interoperate with, rather than compete against, existing gateways; start beta recruitment on Day 1; fold Problem 02/03 teams in as target users and testers. See [15](15_DECISION_LOG.md) D011–D020.

All docs rewritten the same day. Execution starts 28 Sep 2026.
