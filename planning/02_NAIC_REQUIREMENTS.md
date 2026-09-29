# NAIC 2026 Requirements

Source: https://ncair.nitda.gov.ng/naic/ (re-checked 28 September 2026).

## Key dates

| Date | Event |
|---|---|
| 22 Sep | Applications opened |
| **12 Oct, 11:59 PM WAT** | **Build/submission deadline** |
| 15–17 Oct | N-ATLAS integration verification |
| 16–20 Oct | Shortlisting / notification |
| 25 Oct–4 Nov | Mentorship |
| 8–10 Nov | Finals / showcase |

## How participants access N-ATLAS (VERIFIED from NAIC page)

- **Hugging Face**: `huggingface.co/NCAIR1/N-ATLaS` (open to everyone, gated)
- **API credentials**: *shortlisted teams only*
- **Compute credits**: through partner infrastructure, *accelerated teams only*

Implication: before 12 Oct we integrate through the Hugging Face weights and pay for our own GPU. The integration evidence must show official NCAIR1 weights being loaded and run. After shortlisting we add the official API as another backend.

## Problem 01 — Developer Infrastructure (our track)

> "Build the tools that make N-ATLAS easy to build with"

Listed example solutions include SDKs, interactive playgrounds, **fine-tuning starter kits**, training scripts, **evaluation tools** and bilingual documentation.

Validation: **at least 2 external beta testers.**

Hard rule: must genuinely integrate N-ATLAS. Wrapping other foundation models disqualifies.

## Other problem statements (they are our users)

- **Problem 02 — Voice-First:** must use official N-ATLAS ASR. Validation: 50+ real user interactions. → needs `atlasforge transcribe` and ASR WER evaluation.
- **Problem 03 — Sectoral Fine-Tuning:** fine-tune for a domain, show measurable improvement over base, deploy via documented API. Validation: 500+ example benchmark against base. → needs `atlasforge finetune` and `atlasforge compare`.

## Submission components (7)

1. Working artefact
2. N-ATLAS integration evidence ("documentation showing how the solution integrates with N-ATLAS")
3. Real-world validation
4. Technical documentation
5. 3–5 minute demo video
6. Team profile
7. Institutional endorsement or CAC certificate / ID

## Judging criteria (no published weights)

1. Working artefact & technical rigour
2. N-ATLAS integration depth
3. Real-world validation evidence
4. Impact potential
5. Scalability & sustainability
6. Team capability

## Open questions for NAIC (email on Day 1)

- Team size conflict: the page says 2–5 for both tracks, and also 1–6 for Innovation & Enterprise. Which applies?
- Accepted format for beta tester evidence?
- Does integration via official Hugging Face weights (no API) satisfy the integration verification on 15–17 Oct? (Expected yes, but get it in writing.)
- Is a public repository required, or is private access for judges acceptable?
