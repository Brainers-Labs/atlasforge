# Risks

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | **No GPU / GPU budget not approved on D1** | Med | Critical | Approve budget today. Fallbacks: Colab Pro+, university cluster, 4-bit on a 12–16 GB consumer GPU. The ASR work runs on CPU in the meantime. | B |
| R2 | Gated-access approval delayed | Low–Med | High | Every member requests access on D1. Testers request on invite, not on the session day. | All / C |
| R3 | **Judges see us as a duplicate of the existing N-ATLAS Kit** | Med | High | Distinct name. Explicit positioning as the evaluation and fine-tuning layer. Interoperate with and credit them. Don't ship a competing SDK/gateway/playground. | C |
| R4 | Fewer than 2 testers complete | Med | Critical (disqualifying) | Recruit 5 from D1. Two rounds. Problem 02/03 teams have a direct incentive. Endpoint removes the GPU barrier. | C |
| R5 | vLLM doesn't serve N-ATLaS cleanly (chat template, tokenizer quirks) | Low–Med | Med | Verify on D2. Fallback: `local` backend only on the GPU box, with testers using a thin endpoint via HF TGI or a minimal FastAPI wrapper. | B |
| R6 | Metrics produce misleading numbers (normalization bugs) | Med | High (credibility) | Normalization tests with native-speaker-checked fixtures. Always report both tone views. Record configs in reports. | A |
| R7 | Fine-tune smoke run doesn't finish in time | Med | Med | It's P1 and cuttable. Ship config + notebook + documented small run. Never claim results we didn't get. | B |
| R8 | Licence misunderstanding (endpoint for testers, adapters) | Low–Med | High | Beta endpoint stays well under 1,000 users and is non-commercial. Ask NAIC/Awarri if unclear. No weight redistribution. | C |
| R9 | Scope creep ("let's also add a playground") | High | High | Cut line in [11](11_14_DAY_EXECUTION_PLAN.md). New ideas go to a `post-NAIC` issue label. | All |
| R10 | Windows install pain (bitsandbytes, ffmpeg) | Med | Med | CI on Windows. `doctor` detects it. Docs recommend WSL/Colab for local weights. The core and `openai` backend are pure Python. | A |
| R11 | Secret leak in repo, video or screenshots | Low | High | gitleaks pre-commit + CI, masked tokens, a review pass of the video. | All |
| R12 | Team capacity (people unavailable) | Med | High | Streams merge in a defined order. Milestone slip >1 day triggers the cut line. | Lead |
