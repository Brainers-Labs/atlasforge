# Demo Story (3–5 minute video)

## One sentence

> AtlasForge lets any developer run, measure, improve and prove N-ATLaS on their own task, in Hausa, Yoruba, Igbo and Nigerian English.

## Core message

> Downloading N-ATLaS is easy. Knowing whether it works for *your* problem, and proving your fine-tune beats the base model, is not. AtlasForge is that missing layer.

## Script (target 4:00)

| Time | Beat | On screen |
|---|---|---|
| 0:00–0:30 | **Problem.** N-ATLaS is open weights. Every NAIC team must load it, handle 30 s ASR limits, and prove improvement on 500+ examples, with no tooling. | Hugging Face NCAIR1 page → NAIC Problem 03 requirement text |
| 0:30–0:55 | **Install + doctor** | `pip install atlasforge`, `atlasforge doctor` showing GPU, token, gated access ✓ for all 5 models |
| 0:55–1:20 | **Real N-ATLaS request** in Yoruba | `atlasforge run "..."` with real output. The model ID and revision are visible. |
| 1:20–1:55 | **ASR on a WhatsApp voice note** (>30 s, Hausa) → transcript → N-ATLaS summary | `atlasforge transcribe note.ogg --lang ha` then a pipe into `run` |
| 1:55–2:40 | **Evaluate** a starter pack. Show why tone-aware vs tone-insensitive matters. | `atlasforge eval` → terminal table → Markdown report |
| 2:40–3:20 | **Compare** base vs a fine-tuned adapter, with confidence interval and a generated model card with Awarri attribution | `atlasforge compare`, report, `card` |
| 3:20–3:45 | **Real validation.** External testers, what broke in Round 1, what we fixed, their quotes (with consent) | tester evidence page |
| 3:45–4:00 | **Open source + ecosystem.** Works with community gateways. Apache-2.0. | GitHub repo, docs site |

## Rules

- Every output is from a real run on the recorded date. No edited numbers.
- No tokens on screen. Use a demo shell profile with masked environment variables.
- Record terminal segments on D12. Edit on D13.
