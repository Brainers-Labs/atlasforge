# Transcribe and evaluate speech

AtlasForge works with the four official N-ATLaS speech recognition models: Hausa, Yoruba, Igbo and Nigerian-accented English. They are Whisper-based, small enough to run on an ordinary laptop, and accept at most **30 seconds** of audio per call. AtlasForge hides that limit: you give it audio of any length.

| Code | Language | Model |
|---|---|---|
| `ha` | Hausa | `NCAIR1/Hausa-ASR` |
| `yo` | Yoruba | `NCAIR1/Yoruba-ASR` |
| `ig` | Igbo | `NCAIR1/Igbo-ASR` |
| `en` | Nigerian-accented English | `NCAIR1/NigerianAccentedEnglish` |

## Prerequisites

- `ffmpeg` on your `PATH` (`atlasforge doctor` checks it).
- Access to the model repos: see [Models and access](../getting-started/models-and-access.md).
- For local use: `pip install -e ".[asr]"`.

## Transcribe

```bash
atlasforge transcribe note.ogg --lang ha --backend local
```

```text
an kwatanta faretin gine-ginen da ke yin sararin samaniya, hunkunk, da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan victoria habu.
```

`transcribe` accepts one or more files in any format ffmpeg can read: `.wav`, `.mp3`, `.m4a`, `.ogg`/opus (WhatsApp voice notes) and more.

| Option | Meaning |
|---|---|
| `--lang`, `-l` | **required**: `ha`, `yo`, `ig` or `en` |
| `--backend`, `-b` | `local` (load the model in this process) or `openai` (an endpoint that serves `/audio/transcriptions`) |
| `--base-url`, `--model`, `--timeout`, `--retries`, `--allow-insecure-http` | as for the `openai` backend |
| `--device` | `local` only: `auto` (default), `cpu`, `cuda`, `mps` |
| `--out`, `-o` | write one JSON record per file to this JSONL file |
| `--json` | print JSON records instead of plain text |

With `--backend local`, the model is chosen **by `--lang`** automatically. With `--backend openai`, `--model` is the name of the speech model your server serves.

### Output

Plain text by default. With several files, or `--json`, you get one JSON object per line:

```json
{
  "file": "note.ogg",
  "lang": "ha",
  "text": "an kwatanta faretin …",
  "latency_ms": 4210.5,
  "chunks": [{"start_s": 0.0, "end_s": 18.96, "text": "an kwatanta faretin …"}]
}
```

If a file fails (missing, undecodable), the error goes to stderr and the other files continue; the exit code is `1` if any file failed.

!!! note "`chunks` are our windows, not model timestamps"
    `start_s` and `end_s` are the boundaries of the **windows AtlasForge cut**, not times produced by the model. The ASR models themselves can return timestamps (see below), but the `transcribe` command does not expose them yet.

### Noisy warnings are normal

`transformers` prints several warnings on first load (about `forced_decoder_ids`, language detection, `clean_up_tokenization_spaces`). They come from the library and the fine-tuned models' stored configs, and do not indicate a problem.

## How long audio is handled

The models take at most 30 seconds. For longer audio AtlasForge:

1. **Decodes** the file to 16 kHz mono 16-bit audio with ffmpeg.
2. **Cuts it into windows** of 28 seconds that overlap by 2 seconds. Audio that fits in one window is sent whole. The last window ends exactly at the end of the audio.
3. **Transcribes each window** separately.
4. **Merges** the transcripts, removing words duplicated across each overlap.

The merge only removes a duplicate when at least **two consecutive words** at the seam match (ignoring case, tone marks and punctuation). It deliberately errs towards keeping a repeated word, because a legitimate repeat must not be deleted.

!!! warning "Known limitation: fixed windows"
    A word cut by a window boundary can be misheard. The overlap makes that less likely but cannot eliminate it. Splitting on silence is future work.

You can use the same logic from Python on any backend; see [`transcribe_long`](python-api.md#transcribe-long-audio).

## Evaluating a speech model

Make an ASR dataset where each line points to an audio file and gives the reference transcript:

```json
{"id": "c001", "audio": "clips/c001.wav", "reference": "Ina kwana", "lang": "ha"}
{"id": "c002", "audio": "clips/c002.wav", "reference": "Ẹ káàárọ̀", "lang": "yo"}
```

`audio` paths are relative to the dataset file. Every example needs a language, either its own `lang` or a run-wide default with `--lang`. Then:

```bash
atlasforge dataset validate speech.jsonl --task asr
atlasforge eval speech.jsonl --task asr --out runs/asr-base --backend local --lang ha
```

The default metrics are `wer` and `cer`, each under both the tone-aware and tone-insensitive views. **Lower is better.** The pooled WER (total errors over total reference words) is the standard figure to quote. Compare two ASR systems exactly as you would two LLMs:

```bash
atlasforge compare speech.jsonl --task asr --base runs/asr-base --candidate runs/asr-tuned --out cmp
```

Long clips are chunked automatically during `eval`. Keep `--concurrency` at `1` for the `local` backend.

## What has been verified

!!! success "Run for real"
    On a Mac M1 (16 GB), all four speech models were loaded and run through AtlasForge's local backend or the underlying `transformers` pipeline on real [FLEURS](https://huggingface.co/datasets/google/fleurs) test clips. This is a **smoke test of single clips, not a benchmark**: no formal WER has been computed.

| Language | Clip | Reference | Model output |
|---|---|---|---|
| Hausa | 19 s | *An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor.* | *an kwatanta faretin gine-ginen da ke yin sararin samaniya, hunkunk, da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan victoria habu.* |

The Hausa output is close, with errors on the proper nouns "Hong Kong" and "Harbor". The Yoruba and Igbo outputs were also close, with tone marks largely preserved and some word-boundary drift. The English model was only tried on a general US-accent clip, so its accuracy on Nigerian-accented speech is **not** tested. The full record is in [Verified model facts](../natlas/model-facts.md#our-verification-log).

Also verified: the models return timestamps when asked (`return_timestamps=True` gives one coarse segment per window; `return_timestamps="word"` gives per-word times). AtlasForge does not use them yet.

## What has not been verified

- The `openai` backend's `/audio/transcriptions` path is tested against a stub server, not against a real endpoint serving the N-ATLaS speech models. It is **INFERRED** to work with a server that implements the OpenAI transcription API for Whisper models.
- Formal WER on a real test set.
- Behaviour on children's speech, strong dialects, code-switching and poor audio, which the model cards list as limitations.
