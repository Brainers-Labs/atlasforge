# Verified model facts

A single source of truth for what is known about the N-ATLaS models, with **how well each fact is known**.

| Label | Meaning |
|---|---|
| **VERIFIED** | read from an official source, or observed in our own run |
| **INFERRED** | reasonable but not confirmed |
| **UNKNOWN** | not yet checked |

Last updated: 2 October 2026. The authoritative, always-current copy lives in the repository at `planning/21_NATLAS_DISCOVERY.md`; this page is its readable summary.

!!! warning "Where the model card and the files disagree"
    Two facts differ between the model card and what we observed. Both matter in practice.

    1. **Context length.** The card says "8,092". The model's `config.json` says `max_position_embeddings = 131072`.
    2. **Generation defaults.** The card recommends `temperature=0.1`, `repetition_penalty=1.12`, `max_new_tokens=1000`. The repository's `generation_config.json` ships `temperature=0.6` and no repetition penalty.

    AtlasForge sends the card's values explicitly on every request.

## Official sources

- NAIC: <https://ncair.nitda.gov.ng/naic/>
- Hugging Face organisation: <https://huggingface.co/NCAIR1>

Released in September 2025 by Awarri Technologies with NCAIR, NITDA and the Federal Ministry of Communications, Innovation and Digital Economy. (VERIFIED, model cards)

## Access

| Fact | Status | Source |
|---|---|---|
| All five models are gated; the terms must be accepted on each repository | VERIFIED | model cards, and our own `403 GatedRepoError` before accepting |
| No public hosted API; NAIC API credentials only for shortlisted teams | VERIFIED | NAIC page |
| No Hugging Face Inference Provider deployment for ASR | VERIFIED | Hausa-ASR card |
| Compute credits only for accelerated teams | VERIFIED | NAIC page |

## The LLM: `NCAIR1/N-ATLaS`

| Fact | Value | Status |
|---|---|---|
| Architecture | `LlamaForCausalLM`, hidden size 4096, 32 layers, vocabulary 128,256 | VERIFIED (`config.json`) |
| Base model | Llama-3 8B | VERIFIED (card) |
| Parameters | about 8.0 billion | VERIFIED (card; Ollama reports 8.0B) |
| Weights | 4 safetensors shards, **16.08 GB** total, bfloat16 | VERIFIED (Hugging Face file listing) |
| Languages | English, Hausa, Igbo, Yoruba | VERIFIED (card) |
| Context length | `max_position_embeddings` = **131072** | VERIFIED (our read of `config.json`); contradicts the card's "8,092". Quality at long context was not tested |
| Training data | about 391.9 M tokens of instruction data, four languages | VERIFIED (card) |
| Recommended generation | `max_new_tokens=1000`, `repetition_penalty=1.12`, `temperature=0.1` | VERIFIED (card) |
| Repository generation defaults | `temperature=0.6`, no repetition penalty, `eos_token_id=[128001, 128008, 128009]`, `bos_token_id=128000` | VERIFIED (`generation_config.json`) |
| Chat template | present; the Llama-3.1-Instruct format with tool-calling blocks and a default "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" system block | VERIFIED (`tokenizer_config.json`) |
| Hardware | the card says a CUDA GPU with fp16, about 16 GB VRAM at fp16 and about 6 GB at 4-bit | card VERIFIED; the VRAM figures are INFERRED |
| Fits a 16 GB Apple M1 at fp16? | **No**: 16.08 GB of weights against 17.18 GB of total RAM | VERIFIED by arithmetic (a live load was deliberately not attempted) |
| Runs as int4 through Ollama on that Mac | **Yes** (6.6 GB) | VERIFIED (our run) |
| Ollama keeps the chat template on import | **No**: only `{{ .Prompt }}` is kept; the template must be restored | VERIFIED (our run) |
| vLLM serves it unmodified | not tested | UNKNOWN |
| Loads in fp16 / 4-bit through `transformers` | not tested | UNKNOWN |
| Limitations | bias, dialect and accent bias, limited code-switching | VERIFIED (card) |

## The speech models

| Fact | Value | Status |
|---|---|---|
| Architecture | `WhisperForConditionalGeneration` for all four | VERIFIED (`config.json` of each) |
| One model per language | Hausa, Yoruba, Igbo, Nigerian-accented English | VERIFIED |
| Input | 16 kHz, `chunk_length` 30 s, 80 mel bins | VERIFIED (`preprocessor_config.json`) |
| Maximum duration per call | 30 s | VERIFIED (cards) |
| Usage | `pipeline("automatic-speech-recognition", model="NCAIR1/<name>")` | VERIFIED (cards, and what we ran) |
| Training data | Hausa 120 h; Yoruba 627.09 h; Igbo and English not stated | partly VERIFIED |
| Timestamps | `return_timestamps=True` returns coarse segments; `return_timestamps="word"` returns per-word times | VERIFIED (our run, Hausa) |
| Confidence scores | not provided | VERIFIED (absent) |
| Published WER | none on the cards | VERIFIED (absent) |
| Limitations | code-switching, dialects, poor audio, children's speech | VERIFIED (cards) |

## Published evaluation by others

A third-party AfroBench-LITE study (Hugging Face blog, `lm-evaluation-harness`) reports relative gains over Llama-3-8B-Instruct of about +57% (Yoruba), +53% (Hausa), +40% (Igbo) and +15% (English), and a persistent 24 to 32 point gap between English and the Nigerian languages. This is as reported by that study, not by NCAIR and not reproduced by AtlasForge.

## Commit revisions we verified against

| Repository | Commit |
|---|---|
| `NCAIR1/N-ATLaS` | `e294476928aca9030e924ca27bb8e085e8581273` |
| `NCAIR1/Hausa-ASR` | `e635b9eda29060c6114c8f4d8b2d903f5c83a44a` |
| `NCAIR1/Yoruba-ASR` | `d1ae7b8b79c2ccd547d8761effe5057433f3fc7f` |
| `NCAIR1/Igbo-ASR` | `180732299d5cba3dc8b289260ac84b7838bb3954` |
| `NCAIR1/NigerianAccentedEnglish` | `3c52c6e6c9ec508014a7b9db6a42b503b8930dff` |

Models can change after this date. Pin a `revision` when you need reproducibility; the `local` backend records the revision in `run.json`.

## Our verification log

Machine: a MacBook with an Apple M1 and 16 GB of unified memory, macOS 27.0, Apple MPS or CPU only (no NVIDIA GPU). Date: 30 September to 2 October 2026.

| Check | Result |
|---|---|
| Gated access to all five repositories | **Passed**, after the licence was accepted on each page |
| Hausa ASR on a real 19 s clip (FLEURS `ha_ng`) | Close match; proper nouns "Hong Kong" and "Harbor" wrong; tone and hooked letters preserved. Run through `atlasforge transcribe --backend local` on Apple MPS |
| Yoruba ASR on a real 26 s clip (FLEURS `yo_ng`) | Good; tone marks mostly right; some word-boundary drift in the second half. Direct `transformers` pipeline on CPU |
| Igbo ASR on a real 11 s clip (FLEURS `ig_ng`) | Good; minor spelling and word-boundary drift. Direct pipeline on CPU |
| Nigerian-accented English ASR | Loads and runs; near-perfect on a general **US-accent** clip (FLEURS `en_us`). This does **not** test Nigerian-accented speech |
| `return_timestamps` | segment-level and word-level both work (Hausa) |
| LLM, int4 through Ollama, via `atlasforge run` | Works end to end; replies in English, Hausa, Yoruba and Igbo; latency about 1 to 12 s on the M1. Quantised, so quality is not representative. Observed: English correct; Yoruba fluent but shallow; Igbo fluent but with a hallucinated claim; the Hausa answer did not address the question |
| LLM at fp16 on the Mac | Not attempted (cannot fit) |
| vLLM | Not tested (no NVIDIA GPU) |
| Formal WER or LLM benchmark | Not done |

Environment findings recorded along the way: the default `scipy` wheel fails to load on this macOS (pin `scipy<1.15`), and Hugging Face's Xet download path failed repeatedly (use `HF_HUB_DISABLE_XET=1`). See [Troubleshooting](../operations/troubleshooting.md).
