# N-ATLaS Discovery

Single source of truth for model facts. Every row has a status and a source.
**VERIFIED** = read from an official source or observed in our own run · **INFERRED** = reasonable but unconfirmed · **UNKNOWN** = not yet checked

Last updated: 2 Oct 2026 (ASR verified with real runs on a Mac M1 16 GB, CPU-only. LLM verified via `config.json`/tokenizer and a live int4 generation through Ollama. fp16 and vLLM are still unverified and need a GPU box.)

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
| Context length | `config.json` `max_position_embeddings` = **131072** | VERIFIED (our own read of `config.json`, 30 Sep). **Card's "8,092" appears wrong** — the checkpoint is configured for Llama-3's full 128k context. Card value unverified against real generation quality at long context; treat 131072 as the ceiling, not a recommendation. |
| Training data | ~391.9M tokens instruction data, 4 languages | VERIFIED (card) |
| Recommended generation (card) | `max_new_tokens=1000`, `repetition_penalty=1.12`, `temperature=0.1` | VERIFIED (card) |
| `generation_config.json` defaults (repo) | `temperature=0.6`, `repetition_penalty` not set, `eos_token_id=[128001,128008,128009]`, `bos_token_id=128000` | VERIFIED (our own read, 30 Sep). **Differs from the card's recommended settings** (0.6 vs 0.1 temperature; no default repetition_penalty). Explicitly pass the card's params rather than trusting the repo defaults. |
| Weights on disk | 4 safetensors shards, 16.08 GB total (bf16) | VERIFIED (`HfApi.model_info(files_metadata=True)`, 30 Sep) |
| Loading | `AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS", torch_dtype=float16, device_map="auto")` | VERIFIED (card) |
| Hardware | CUDA GPU with fp16. ~16 GB VRAM fp16, ~6 GB 4-bit. | Card VERIFIED. On our Mac M1 16 GB unified memory: 16.08 GB of weights alone would consume ~93% of total system RAM with nothing left for the OS, KV cache or activations — **does not fit**, confirmed by file size vs `sysctl hw.memsize` (17.18 GB) without attempting a live load (real risk of hanging the machine). fp16/bf16 needs a real GPU (≥24 GB) or a quantised local serve. |
| Chat template in tokenizer config | Present: standard Llama-3.1-Instruct template (tool-calling blocks, `date_string` default `"26 Jul 2024"`) | VERIFIED (our own read of `tokenizer_config.json`, 30 Sep). Note the template looks like Llama-3.1's, not plain Llama-3's, despite the card saying "Llama-3 8B" — minor discrepancy, not chased further. |
| vLLM serving works unmodified | ? | UNKNOWN — untestable on this machine (no CUDA/NVIDIA GPU on Mac M1). Needs the GPU box from D1. |
| Known limitations | bias, dialect/accent bias, limited code-switching | VERIFIED (card) |

## ASR models

| Fact | Value | Status |
|---|---|---|
| Architecture | `WhisperForConditionalGeneration` for all 4 (`config.json` `architectures`), consistent with Whisper-Small fine-tunes | VERIFIED for all 4 (our own `config.json` read, 30 Sep — Igbo and English were previously inferred/unknown, now confirmed) |
| Languages | one model per language: ha, yo, ig, Nigerian-accented en | VERIFIED |
| Input | 16 kHz recommended, librosa-readable formats | VERIFIED (cards + our own `preprocessor_config.json` read: `sampling_rate=16000`, `chunk_length=30`) |
| Max duration | 30 s per inference | VERIFIED (cards) |
| Usage | `pipeline("automatic-speech-recognition", model="NCAIR1/<lang>-ASR")` | VERIFIED (cards, and this is exactly what we ran) |
| Training data | Hausa 120 h · Yoruba 627.09 h · Igbo ? · English ? | Partly VERIFIED |
| Timestamps | `return_timestamps=True` returns one segment-level chunk per utterance (`{"timestamp": (start, end), "text": ...}`); `return_timestamps="word"` returns per-word timestamps and works cleanly | VERIFIED (our own run against Hausa-ASR, 30 Sep — see evidence below) |
| Confidence scores | not provided | VERIFIED (absent) |
| Published WER | none on cards | VERIFIED (absent). We did not compute a formal WER (single-sample smoke test only); see verification log for qualitative accuracy. |
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

## Our verification log

Machine: Mac M1, 16 GB unified memory, macOS 27.0, CPU/MPS only (no NVIDIA GPU). `atlasforge[asr]` installed.

| Date | Who | Check | Result | Evidence |
|---|---|---|---|---|
| 30 Sep 2026 | Ismail (token) + Claude | Gated access granted (5 repos) | All 5 accessible after licence acceptance | `HfApi.model_info()` / `hf_hub_download("config.json")` succeeded for all 5 |
| 30 Sep 2026 | Claude | Commit SHAs of all 5 repos | N-ATLaS `e294476928aca9030e924ca27bb8e085e8581273` · Hausa-ASR `e635b9eda29060c6114c8f4d8b2d903f5c83a44a` · Yoruba-ASR `d1ae7b8b79c2ccd547d8761effe5057433f3fc7f` · Igbo-ASR `180732299d5cba3dc8b289260ac84b7838bb3954` · NigerianAccentedEnglish `3c52c6e6c9ec508014a7b9db6a42b503b8930dff` | `HfApi.model_info(repo).sha` |
| 30 Sep 2026 | Claude | `config.json` max_position_embeddings | 131072 (contradicts card's "8,092") | see LLM table above |
| 30 Sep 2026 | Claude | Chat template present | Yes, Llama-3.1-Instruct-style template with tool-calling blocks | `tokenizer_config.json` |
| 30 Sep 2026 | Claude | LLM loads fp16, VRAM/RAM used | Not attempted live — 16.08 GB of weights vs 17.18 GB total system RAM (Mac M1). Calculated from file sizes, not run, to avoid hanging the machine. Confirms the handoff's prediction. | `HfApi.model_info(files_metadata=True)` + `sysctl hw.memsize` |
| 2 Oct 2026 | Claude | LLM generation via quantised local serve (Ollama 0.34.4 importing the official safetensors snapshot `e294476…` with `--quantize int4`, local only, never published; 6.6 GB) | **Works end to end**: `atlasforge run` → `openai` backend → Ollama `/v1` → model replies in all 4 languages. Latency on M1 16 GB: ~1–12 s per short reply (first call includes model load). `ollama show` independently reports context length 131072, matching `config.json`. **Quality caveat:** this is int4-quantised, not the fp16 model, so these outputs say nothing reliable about official quality. Observed: English correct ("The capital of Nigeria is Abuja."); Yoruba fluent but shallow; Igbo fluent but contains a hallucination (invokes the "Roman Empire" for Nigerian history); Hausa answer was short and did not answer the question ("Kwam a Kano, jihar Kano."). Not enough samples to attribute any of this to the model vs the quantisation. | prompts: capital of Nigeria (en); "Ina kwana? Gaya mini game da Najeriya a taƙaice." (ha); "Bawo ni? Sọ fun mi nípa ìlú Eko ní ṣókí." (yo); "Kedu? Gwa m banyere Naịjirịa n'ụzọ dị mkpụmkpụ." (ig) |
| 2 Oct 2026 | Claude | Chat template survives Ollama's safetensors import | **No.** `ollama show --template` returned only `{{ .Prompt }}` (no role headers). We rendered the official template with `tokenizer.apply_chat_template` and re-created the model with a matching `TEMPLATE` (Llama-3.1 header format + default "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" system block) and stop tokens `<|eot_id|>`, `<|end_of_text|>`, `<|eom_id|>`. Anyone serving N-ATLaS through Ollama must do the same or get unformatted prompts. BOS handling in the Ollama template was not separately verified. | `Modelfile.chat` (not committed; scratch) |
| 2 Oct 2026 | Claude | Download reliability (not model-specific) | `snapshot_download` of the 16 GB LLM failed twice with a Hugging Face Xet CDN error (`CAS Client Error ... error sending request`); setting `HF_HUB_DISABLE_XET=1` fixed it (completed in ~9 min). Worth a troubleshooting note for testers. | hf-hub with `hf-xet 1.6.0` |
| 30 Sep 2026 | Claude | vLLM serve + chat completion | Not testable on this machine (no CUDA). Still open for the GPU box. | — |
| 30 Sep 2026 | Claude | Hausa-ASR transcribes a real 19 s sample (FLEURS `ha_ng` test #0) | Close match, a couple of proper-noun errors ("Hong Kong"→"hunkunk", "Harbor"→"habu"), tone diacritics preserved | ref: "An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor." hyp: "an kwatanta faretin gine-ginen da ke yin sararin samaniya, hunkunk, da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan victoria habu." |
| 30 Sep 2026 | Claude | Yoruba-ASR transcribes a real 26 s sample (FLEURS `yo_ng` test #0) | Good, tone marks mostly correct, some word-boundary drift in the back half | ref: "Àwọn èyàn ti mọ̀ nípa àwọn kemika pepe bí wúrà, fàdákà àti kọ́pa àtijọ́..." hyp: "àwọn èyàn tí mọ̀ nípa àwọn kẹ́míkà pépè bí wúrà fàdákà àti kọ́pà àtijọ́..." |
| 30 Sep 2026 | Claude | Igbo-ASR transcribes a real 11 s sample (FLEURS `ig_ng` test #0) | Good, minor word-boundary/spelling drift | ref: "Ka akara Rossby na-adị obere karịa, ka arụmarụ na-adịkwu obere nke kpakpando n'ikwanye ugwu nye ụmụ ntụgharị nke ihendọta." hyp: "a kara rosby na-adị obere karịa ka arụmarụ na-adịkwa obere nke gbagbando n'ịkwanye ugwu nye ụmụ ntụgharị nke ihe ndọta." |
| 30 Sep 2026 | Claude | NigerianAccentedEnglish transcribes a real 11 s sample (FLEURS `en_us` test #0 — **not actually Nigerian-accented**, a general US English sample; loads-and-runs check only, not an accuracy test for this model's real purpose) | Near-perfect | ref: "However, due to the slow communication channels, styles in the west could lag behind by 25 to 30 year." hyp: "however, due to the slow communication channels, styles in the west could lag behind by 25 to 30 years." |
| 30 Sep 2026 | Claude | ASR `return_timestamps` behaviour | `return_timestamps=True` → one segment per chunk (`(0.0, 17.0)` for our 19 s clip — coarse, not per-sentence). `return_timestamps="word"` → clean per-word timestamps | ran against Hausa-ASR, see LLM/ASR tables |
| 30 Sep 2026 | Claude | Environment note (not model-specific) | `atlasforge[asr]` pulls in `transformers>=4.44` unpinned, which resolved to 5.17.0 and transitively imports `scipy` at package-import time (via an unrelated object-detection loss module). The default PyPI `scipy==1.15.3` wheel fails to `dlopen` its `_propack` extension on this macOS 27.0 / Apple Silicon build (`zero-fill section` Mach-O error) — unrelated to N-ATLaS. Pinning `scipy==1.14.1` works around it. | Reproduced consistently; worth a pinned constraint or a documented troubleshooting note before beta testers hit it on newer macOS |
