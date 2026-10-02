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

You choose the language: the models are monolingual and there is no automatic language detection. They also do not provide word timestamps or confidence scores, and AtlasForge does not invent them.

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
    A word cut by a window boundary can be misheard, and merging only removes a duplicate when at least two words match at the seam (keeping a repeat is safer than deleting a real word). The window boundaries in the output are *ours*, not model timestamps. Silence-aware splitting is planned.

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

If speech models are used inside an application, report WER on audio from your real users; the model cards note weaker performance on children's speech, noisy audio, dialects and code-switching.

## Privacy

Treat voice recordings as personal data. AtlasForge decodes audio in memory and a temporary file, and does not store or log it. Do not commit recordings to a repository, and get consent before collecting them.
