# Product Vision

## Problem

N-ATLaS exists as open weights. Getting from "the weights exist" to "I shipped something and can prove it works" still means:

1. Accept five gated licenses, download ~16 GB of LLM weights plus the ASR models, and find a GPU.
2. Work out the chat template, generation settings and memory limits (fp16 vs 4-bit).
3. Get around the ASR models' 30-second input limit and fiddly audio formats (WhatsApp `.ogg/opus`, `.m4a`).
4. Decide whether the model is actually good enough for *your* task in Hausa, Yoruba, Igbo or Nigerian English, with metrics that don't break on Yoruba tone marks.
5. Fine-tune it for a domain, and prove the fine-tune beats the base model.
6. Stay within a custom license: 1,000 active end-user cap, attribution, "Powered by Awarri" naming for derivatives.

Steps 1–3 are partly addressed by existing community tools (see [22](22_COMPETITIVE_LANDSCAPE.md)). **Steps 4–6 have no tooling.** That is where AtlasForge starts.

## Product thesis

> Access is solved by downloading. Confidence is not. AtlasForge gives developers confidence: run it, measure it, improve it, prove it.

## Target users (in priority order for v0.1)

1. **NAIC 2026 Problem 03 teams**: must fine-tune and benchmark 500+ examples against base N-ATLAS. They need `compare` and the fine-tune kit *this month*.
2. **NAIC 2026 Problem 02 teams**: must use official ASR. They need `transcribe` with chunking and ASR evaluation (WER).
3. University researchers evaluating Nigerian-language models.
4. Startups and developers deciding whether N-ATLaS fits their use case.

Groups 1–2 are reachable right now and have urgent need. They are our beta-tester pool.

## Product boundary

AtlasForge is NOT:
- a new foundation model or a redistribution of N-ATLaS weights
- a chatbot or end-user application
- a wrapper around GPT/Claude/Gemini or any non-N-ATLaS model
- a hosted SaaS (a shared beta endpoint is temporary and for testing only)
- a competitor to other community N-ATLAS SDKs. Our OpenAI-compatible backend works *with* their gateways.

## Long-term stack

```text
N-ATLaS (NCAIR1 weights)            [official]
   |
   +-- Serving (vLLM / transformers / future official API)   [existing tools]
   |
   +-- AtlasForge
        +-- backends: local | openai-compatible | official API (when available)
        +-- ASR: chunking, conversion, routing
        +-- eval: tasks, NG-language metrics, reports
        +-- compare: base vs candidate, significance
        +-- finetune: LLM QLoRA, ASR Whisper-Small recipes
        +-- cards: license-aware model cards
        +-- benchmarks: community benchmark packs (future)
        +-- leaderboard: public N-ATLaS derivative leaderboard (future)
```

v0.1 ships the first six boxes. Everything else waits until after 12 October.
