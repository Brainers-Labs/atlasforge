# ASR

## Facts (see [21](21_NATLAS_DISCOVERY.md))

The official N-ATLaS ASR models are **Whisper-Small fine-tunes** (244M parameters), one per language:

| Lang | Model | Training data (card) |
|---|---|---|
| ha | `NCAIR1/Hausa-ASR` | 120 h |
| yo | `NCAIR1/Yoruba-ASR` | 627 h |
| ig | `NCAIR1/Igbo-ASR` | verify |
| en | `NCAIR1/NigerianAccentedEnglish` | verify |

- Gated. Licensed under the same Awarri research license as the LLM.
- Input: 16 kHz recommended. **Max 30 s per inference.**
- Model cards do not document timestamps, confidence scores or WER.
- Known weaknesses (from the cards): code-switching, dialects, poor audio, children's speech.

So the "Whisper for N-ATLAS" analogy is now literal. Our job is not a new model. Our job is to make these models practical and measurable.

## What AtlasForge adds

1. **Audio in any format**: ffmpeg-based conversion of `.ogg/.opus` (WhatsApp), `.m4a`, `.mp3`, `.wav` → 16 kHz mono float32.
2. **Long audio**: chunk into ≤30 s windows with overlap and merge the text. v0.1 uses fixed windows by default. Silence-aware splitting was a stretch goal and is now implemented as an opt-in (`--silence-aware`), still off by default because the two have not been compared on real audio.
3. **Routing**: `--lang` selects the right model. Models are cached after the first load.
4. **ASR evaluation**: WER / CER, tone-aware and tone-insensitive, per-utterance and corpus-level. This is what Problem 02 teams need for their evidence.
5. **Voice → LLM example**: transcribe → N-ATLaS prompt (for example translate, summarise, answer), as an example script and not a product.

## CLI

```bash
atlasforge transcribe voice-note.ogg --lang ha
atlasforge transcribe folder/ --lang yo --out transcripts.jsonl
atlasforge eval asr_test.jsonl --task asr --lang yo
```

## Honesty rules

- `chunks[].start_s/end_s` are **our chunk boundaries**, documented as such. They are not word timestamps.
- Test whether `return_timestamps=True` produces sensible output from the fine-tuned checkpoints (the fine-tune may have removed timestamp tokens). Expose it only if verified, and label it experimental.
- No confidence scores, no language detection, no claimed WER numbers unless we measured them. Measured WER must name the dataset, split and normalization used.

## Candidate ASR test data (licences to verify before use)

- Google FLEURS (`ha_ng`, `ig_ng`, `yo_ng` configs): CC-BY-4.0 (verify)
- Mozilla Common Voice (ha, yo, ig subsets): CC0 (verify availability and size)
- Our own recordings: only with explicit consent. Store transcripts, not raw audio, in the repo.
