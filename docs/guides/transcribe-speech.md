# Transcribe speech

Turn audio into text with the official N-ATLaS speech models, and measure how accurate they are.

!!! warning "Not yet run on the real speech models"
    The audio handling (decoding, splitting long audio, merging transcripts) is fully tested. Loading the official `NCAIR1` speech models and measuring their real accuracy has **not been done yet**; see [Project status](../help/status.md).

## The models

There is one speech model per language, each a Whisper-Small fine-tune of about 244 million parameters, small enough to run on a CPU.

| `--lang` | Model |
|---|---|
| `ha` | `NCAIR1/Hausa-ASR` |
| `yo` | `NCAIR1/Yoruba-ASR` |
| `ig` | `NCAIR1/Igbo-ASR` |
| `en` | `NCAIR1/NigerianAccentedEnglish` |

You choose the language: the models are monolingual and there is no automatic language detection. They provide no confidence scores, and AtlasForge does not invent any. The models are Whisper fine-tunes, so `transformers` can return timestamps for them (coarse segments, or per-word times; checked on the Hausa model), but AtlasForge does not use them: the window boundaries in its output are its own.

## Transcribe a file

```bash
atlasforge transcribe voice-note.ogg --lang ha --backend local
```

First install the `asr` extra and [get access to the model](../get-started/access-and-licences.md). The first run downloads about 1 GB.

=== "Locally"

    ```bash
    atlasforge transcribe voice-note.ogg --lang ha --backend local
    ```

=== "Through a server"

    ```bash
    atlasforge transcribe voice-note.ogg --lang ha --base-url http://127.0.0.1:8000/v1
    ```

    The server must provide `/v1/audio/transcriptions`.

## Any format, any length

- **Formats.** Anything ffmpeg reads: `.wav`, `.mp3`, `.m4a`, `.ogg`/opus (WhatsApp voice notes). You need [ffmpeg installed](../get-started/installation.md#install-ffmpeg).
- **Length.** The models accept at most 30 seconds. AtlasForge decodes to 16 kHz mono, cuts longer audio into **28-second windows with a 2-second overlap**, transcribes each, and merges the text by removing words duplicated across each seam.

!!! note "Known limitation of fixed windows"
    A word cut by a window boundary can be misheard, and merging only removes a duplicate when at least two words match at the seam (keeping a repeat is safer than deleting a real word). The window boundaries in the output are *ours*, not model timestamps.

### Cutting at a pause instead

`--silence-aware` moves each boundary up to two seconds, either way, to the quietest moment near it — so the cut lands in a breath rather than inside a word:

```bash
atlasforge transcribe interview.m4a --lang ig --backend local --silence-aware
```

A boundary only moves if there is a **pause** there (a frame at or below -40 dBFS). A stretch that is merely quieter than its surroundings does not count, so loud audio is not cut arbitrarily. The number of windows, their overlap and the 30-second limit are unchanged: a move that would push a window past the limit is simply not taken, and the boundary stays where the fixed grid put it.

The default is the fixed grid, deliberately. Both split the *same recording* differently, so transcripts and WER from a silence-aware run are not comparable with a fixed-window one — and listening to both is the only way to know which is better on your audio, which is exactly the check that cannot be done until the real models are run. In Python: `atlasforge.evaluate(..., silence_aware=True)`, or `transcribe_long(..., silence_aware=True)`.

## Several files

```bash
atlasforge transcribe a.ogg b.ogg c.ogg --lang yo --out transcripts.jsonl
```

Each line has the file name, the text, the latency and the chunks. One bad file does not stop the others, but the exit code is 1 if any failed.

## Measure accuracy

Build a speech dataset ([format](../concepts/datasets.md#speech-datasets)) and evaluate it:

```bash
atlasforge eval speech.jsonl --task asr --backend local --lang ha --out runs/asr
```

You get **WER** (word error rate) and **CER** (character error rate) under both [tone views](../concepts/tone-aware-scoring.md). Pooled WER is total word errors over total reference words, the standard figure. Long clips are split automatically during evaluation.

### What it heard instead

A WER number says how much was wrong, not *what*. Every speech report also carries an **ASR error analysis** built from the same alignment the WER uses — so its counts add up to the rate above it:

- **Most substituted** — the word pairs it confused most often, e.g. `sannú → sannu`, with the example ids to listen to.
- **Most dropped** and **Most inserted** — the words it lost and the words it invented.
- **Word error rate by reference length** — pooled WER per length bucket, so you can see whether it falls apart on long utterances.

A substitution is marked **tone-only** when the two words differ *only* in tone marks. For Hausa, Yoruba and Igbo that is the difference between a mis-hearing and a typing convention, and it is why the tone-insensitive view exists: a transcript can be right to a listener and still count as wrong to a strict scorer. Underdots and hooked letters (`ẹ`, `ọ`, `ṣ`, `ɓ`) are never forgiven, so dropping one is a real substitution.

The analysis is arithmetic over transcripts AtlasForge already has. It reads no audio and calls no model, so it costs nothing and needs no GPU.

If speech models are used inside an application, report WER on audio from your real users; the model cards note weaker performance on children's speech, noisy audio, dialects and code-switching.

## What has been run on real models

All four speech models were loaded and run on real [FLEURS](https://huggingface.co/datasets/google/fleurs) test clips (CC BY 4.0) on a Mac M1 with 16 GB. The Hausa clip went through `atlasforge transcribe --backend local` on Apple's GPU; the others were run through the `transformers` pipeline directly, on CPU. These are **single-clip smoke tests, not a benchmark: no word error rate has been computed.**

| Language | Clip | What came back |
|---|---|---|
| Hausa | 19 s | Close to the reference. The proper nouns "Hong Kong" and "Harbor" came out as "hunkunk" and "habu"; tone and hooked letters were kept |
| Yoruba | 26 s | Good, with tone marks mostly right and some word-boundary drift in the second half |
| Igbo | 11 s | Good, with minor spelling and word-boundary drift |
| English | 11 s | Near-perfect, but on a general **US-accent** clip, so this says nothing about Nigerian-accented speech |

For the Hausa clip, the reference was *An kwatanta faretin gine-ginen da ke yin sararin samaniyar Hong Kong da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan Victoria Harbor.* and the model returned *an kwatanta faretin gine-ginen da ke yin sararin samaniya, hunkunk, da ginshiƙi mai walƙiya wanda aka bayyana ta gaban ruwan victoria habu.*

`transformers` prints several warnings the first time a model loads (about `forced_decoder_ids`, language detection and tokenizer clean-up). They come from the library and the models' stored settings and are not a sign of failure.

## Privacy

Treat voice recordings as personal data, which in Nigeria includes the Data Protection Act 2023 for anything you collect yourself. AtlasForge decodes audio in memory and a temporary file, and does not store or log it. Do not commit recordings to a repository, and get consent before collecting them.
