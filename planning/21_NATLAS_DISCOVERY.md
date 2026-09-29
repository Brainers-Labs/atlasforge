# N-ATLaS Discovery

Single source of truth for model facts. Every row has a status and a source.
**VERIFIED** = read from an official source or observed in our own run · **INFERRED** = reasonable but unconfirmed · **UNKNOWN** = not yet checked

Last updated: 28 Sep 2026 (from model cards and the NAIC page. GPU verification pending, D1–D2.)

## Official sources

- NAIC: https://ncair.nitda.gov.ng/naic/
- HF org: https://huggingface.co/NCAIR1
- LLM: https://huggingface.co/NCAIR1/N-ATLaS
- ASR: https://huggingface.co/NCAIR1/Hausa-ASR · https://huggingface.co/NCAIR1/Yoruba-ASR · https://huggingface.co/NCAIR1/Igbo-ASR · https://huggingface.co/NCAIR1/NigerianAccentedEnglish

Released September 2025 by Awarri Technologies with NCAIR, NITDA and the Federal Ministry of Communications, Innovation and Digital Economy. (VERIFIED, model cards)

## Access

| Fact | Status | Source |
|---|---|---|
| All models are gated. You must accept terms on HF before download. | VERIFIED | model cards |
| No public hosted API. NAIC API credentials only for shortlisted teams. | VERIFIED | NAIC page |
| No HF Inference Provider deployment for ASR | VERIFIED | Hausa-ASR card |
| Compute credits only for accelerated teams | VERIFIED | NAIC page |

## LLM — `NCAIR1/N-ATLaS`

| Fact | Value | Status |
|---|---|---|
| Base model | Llama-3 8B | VERIFIED (card) |
| Parameters | 8B · hidden 4,096 · vocab 128,256 | VERIFIED (card) |
| Weights format | safetensors, BF16 | VERIFIED (card) |
| Languages | English, Hausa, Igbo, Yoruba | VERIFIED (card) |
| Context length | card says "8,092" | VERIFIED as stated. Likely 8,192. Confirm in `config.json` (UNKNOWN). |
| Training data | ~391.9M tokens instruction data, 4 languages | VERIFIED (card) |
| Recommended generation | `max_new_tokens=1000`, `repetition_penalty=1.12`, `temperature=0.1` | VERIFIED (card) |
| Loading | `AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS", torch_dtype=float16, device_map="auto")` | VERIFIED (card) |
| Hardware | CUDA GPU with fp16. ~16 GB VRAM fp16, ~6 GB 4-bit. | Card VERIFIED. VRAM numbers INFERRED. Measure. |
| Chat template in tokenizer config | ? | UNKNOWN |
| vLLM serving works unmodified | ? | UNKNOWN (test D2) |
| Known limitations | bias, dialect/accent bias, limited code-switching | VERIFIED (card) |

## ASR models

| Fact | Value | Status |
|---|---|---|
| Architecture | Whisper-Small fine-tunes (~244M params) | VERIFIED (Hausa, Yoruba cards). Igbo INFERRED from search. English UNKNOWN. |
| Languages | one model per language: ha, yo, ig, Nigerian-accented en | VERIFIED |
| Input | 16 kHz recommended, librosa-readable formats | VERIFIED (cards) |
| Max duration | 30 s per inference | VERIFIED (cards) |
| Usage | `pipeline("automatic-speech-recognition", model="NCAIR1/<lang>-ASR")` | VERIFIED (cards) |
| Training data | Hausa 120 h · Yoruba 627.09 h · Igbo ? · English ? | Partly VERIFIED |
| Timestamps | not documented | UNKNOWN (test `return_timestamps`) |
| Confidence scores | not provided | VERIFIED (absent) |
| Published WER | none on cards | VERIFIED (absent) |
| Known limitations | code-switching, dialects, poor audio, children's speech | VERIFIED (cards) |

## Licence (all models)

"Open-Source Research and Innovation License" (Awarri). Key terms from the cards (VERIFIED as summarised; **read the full text before release**):
- Limited to organisations with ≤1,000 active end-users (per 30 days, per a third-party summary; confirm)
- Commercial use requires a separate agreement
- Attribution to Awarri Technologies and the Federal Ministry of Communications
- Derivative works must use the suffix "Powered by Awarri"
- Whether Llama 3 Community License terms also apply: UNKNOWN

## Published evaluation

AfroBench-LITE study (HF blog, seun-ajayi) using `lm-evaluation-harness`: AfriXNLI, Belebele, AfriMMLU, FLORES, SIB, Injongo, AfriMGSM. Relative gains over Llama-3-8B-Instruct: Yoruba +57%, Hausa +53%, Igbo +40%, English +15%. It reports a persistent 24–32 point gap between English and the Nigerian languages, and Yoruba weakest in translation. (VERIFIED as reported in a third-party study, not by NCAIR)

## Our verification log (fill on D1–D2)

| Date | Who | Check | Result | Evidence |
|---|---|---|---|---|
| | | Gated access granted (5 repos) | | |
| | | LLM loads fp16 / 4-bit, VRAM used | | |
| | | Chat template present / output sane in 4 languages | | |
| | | `config.json` max_position_embeddings | | |
| | | vLLM serve + chat completion | | |
| | | Each ASR model transcribes a 10 s sample | | |
| | | ASR `return_timestamps` behaviour | | |
| | | Commit SHAs of all 5 repos | | |
