# Research Questions

Status as of 28 Sep 2026. Answers and evidence live in [21](21_NATLAS_DISCOVERY.md). ✅ answered · 🔬 verify on GPU (D1–D2) · 📨 ask NAIC · ⏳ open

## Model
| Q | Status | Answer / action |
|---|---|---|
| Official repository? | ✅ | `huggingface.co/NCAIR1/N-ATLaS` |
| Model ID / version? | ✅ / 🔬 | `NCAIR1/N-ATLaS`. Record the commit SHA we test against. |
| Licence? | ✅ | Awarri Open-Source Research and Innovation License (custom). Check the full text for endpoint/adapter terms. |
| Local inference? | ✅ | Yes, via transformers (card snippet) |
| Hardware? | 🔬 | Card says a CUDA fp16 GPU. Measure VRAM for fp16 and 4-bit. |
| Languages? | ✅ | English, Hausa, Igbo, Yoruba |
| Chat template present? | 🔬 | Check `tokenizer_config.json` |
| Context length? | 🔬 | Card states "8,092" (likely 8,192). Confirm in `config.json`. |
| Serves on vLLM unmodified? | 🔬 | Test D2 |

## API
| Q | Status | Answer |
|---|---|---|
| Public hosted API? | ✅ | No. NAIC API credentials go to shortlisted teams only. |
| Endpoints / auth / schema / streaming / limits? | ⏳ | Only relevant post-shortlist. `official` backend deferred. |
| Official SDKs? | ✅ | None from NCAIR. A community SDK exists (see [22](22_COMPETITIVE_LANDSCAPE.md)). |

## ASR
| Q | Status | Answer |
|---|---|---|
| Official ASR? | ✅ | 4 models: Hausa-ASR, Yoruba-ASR, Igbo-ASR, NigerianAccentedEnglish (Whisper-Small) |
| Audio format / rate? | ✅ | 16 kHz recommended. Anything librosa reads. |
| Max length? | ✅ | 30 s per inference |
| Timestamps? | 🔬 | Not documented. Test `return_timestamps`. |
| Confidence? | ✅ | Not provided. Won't expose. |
| Language detection? | ✅ | No. Monolingual models. |
| Published WER? | ✅ | Not on the cards. We measure our own and document the methodology. |
| Igbo / English cards match Hausa / Yoruba terms? | 🔬 | Read on D1 |

## Ecosystem
| Q | Status | Answer |
|---|---|---|
| Existing SDKs / playground / gateway? | ✅ | Yes, the community N-ATLAS Kit ([22](22_COMPETITIVE_LANDSCAPE.md)) |
| Existing evaluation tooling? | ✅ | Only general tools (lm-evaluation-harness, used in the AfroBench-LITE study). Nothing developer-facing, nothing NG-language-aware, nothing for ASR WER or base-vs-fine-tune comparison. |
| Unserved pain points? | ✅ / ⏳ | Evaluation, comparison, fine-tuning, licence-aware cards. Confirm with beta testers. |

## NAIC
| Q | Status |
|---|---|
| Team-size rule (2–5 vs 1–6)? | 📨 |
| Beta-tester evidence format? | 📨 |
| Does HF-weights integration satisfy verification? | 📨 |
| Public repo required? | 📨 |
| Required N-ATLaS version? | 📨 |
