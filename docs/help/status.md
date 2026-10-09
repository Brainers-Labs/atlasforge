# Project status

AtlasForge is **pre-alpha ({{ version }})**. This page says plainly what has been proved and what has not, so you can decide what to trust.

## What "verified" means here

<span class="status-verified">Tested</span> The behaviour is covered by automated tests that run on Linux, macOS and Windows with Python 3.10 and 3.13. Where a model is involved, the tests use a stand-in that answers from fixed rules.

<span class="status-verified">Run on real weights</span> It has been run against the real official models on real hardware, and what was seen is recorded below. These are single-sample checks, not benchmarks.

<span class="status-unverified">Not run on real weights</span> The code is written and its wiring is tested against stand-ins, but it has **never been run against the real N-ATLaS models**. It may need adjustment on first contact.

## Components

| Component | Status | Notes |
|---|---|---|
| Datasets: loading, strict validation, fingerprinting | <span class="status-verified">Tested</span> | |
| `dataset validate` | <span class="status-verified">Tested</span> | Duplicates, conflicts, leakage, Unicode, diacritics, balance |
| Tone-aware / tone-insensitive normalisation | <span class="status-verified">Tested</span> | |
| Metrics: exact match, accuracy, macro-F1, chrF, chrF++, WER, CER | <span class="status-verified">Tested</span> | |
| Your own metrics (`--metric module:function`) | <span class="status-verified">Tested</span> | No pooled figure for a custom metric; per-example values only |
| `eval`: resumable runs, circuit breaker, concurrency | <span class="status-verified">Tested</span> | End to end against a real HTTP server with a fake model |
| `report`, `compare`: bootstrap, McNemar, slices, failure-mode flags | <span class="status-verified">Tested</span> | |
| HTML reports (`report.html`, `comparison.html`) | <span class="status-verified">Tested</span> | Self-contained: no JavaScript, no external reference. Escaping is tested with hostile text in a manifest field and in a transcript. |
| `openai` backend (any OpenAI-compatible server) | <span class="status-verified">Tested</span> | Retries, errors, auth, https guard. **Not tested against N-ATLaS served by vLLM/llama.cpp/Ollama.** |
| `demo` data | <span class="status-verified">Tested</span> | Synthetic by design |
| Audio decoding, windowing, transcript merging | <span class="status-verified">Tested</span> | With real ffmpeg, including opus `.ogg` |
| ASR error analysis (substitutions, deletions, insertions, length buckets) | <span class="status-verified">Tested</span> | Arithmetic over transcripts; reads no audio and calls no model |
| Optional `atlasforge.toml` project settings | <span class="status-verified">Tested</span> | Read from the nearest file at or above the working directory; unknown keys are refused |
| `card` | <span class="status-verified">Tested</span> | |
| `finetune --dry-run` and data checks | <span class="status-verified">Tested</span> | |
| `local` backend, speech models | <span class="status-verified">Run on real weights</span> | All four transcribed real FLEURS clips on a Mac M1 (Apple MPS and CPU) |
| `local` backend, LLM | <span class="status-unverified">Not run on real weights</span> | Needs a GPU of 24 GB or more: the fp16 weights (16.08 GB) do not fit a 16 GB machine |
| Official ASR models | <span class="status-verified">Run on real weights</span> | Close transcriptions of single Hausa, Yoruba and Igbo clips; the English model was only tried on a US-accent clip. **No formal WER has been computed** |
| `finetune` training run | <span class="status-unverified">Not run on real weights</span> | Needs an NVIDIA GPU |
| `bench afrobench` (AfroBench-LITE wrapper) | <span class="status-unverified">Not run against the harness</span> | Wiring tested against stand-ins. Tasks are discovered from the installed `lm-evaluation-harness`, never hard-coded; needs the `bench` extra, a GPU and the gated weights |
| `--adapter` evaluation | <span class="status-unverified">Not run on real weights</span> | |
| Serving N-ATLaS with Ollama (int4) | <span class="status-verified">Run on real weights</span> | Imported from the official safetensors on a Mac M1 and answered through `atlasforge run`. Ollama drops the chat template, which has to be restored ([Serve N-ATLaS](../guides/serve-n-atlas.md#option-3-ollama-simplest-on-a-laptop)) |
| Serving N-ATLaS with vLLM, llama.cpp | <span class="status-unverified">Not run on real weights</span> | Commands follow each tool's docs |

## What running the real models has answered

Checked on 30 September to 2 October 2026, on a Mac M1 with 16 GB, with no NVIDIA GPU:

- **Does the tokenizer ship a chat template?** Yes: the Llama-3.1-Instruct format, with a default system block. Ollama's import drops it.
- **What is the true context length?** The repository's `config.json` says **131072**; the model card says 8,092. Which is right in practice was not tested.
- **How much memory does it need?** The fp16 weights are 16.08 GB, so they do not fit a 16 GB machine. An int4 import through Ollama is 6.6 GB and runs. 4-bit through `bitsandbytes` needs an NVIDIA GPU and has not been measured.
- **Do the speech models return timestamps?** `transformers` returns them for `NCAIR1/Hausa-ASR`: coarse segments with `return_timestamps=True` and per-word times with `"word"`. AtlasForge does not use them.
- **Do the speech models load and transcribe?** Yes, all four, on real clips ([Transcribe speech](../guides/transcribe-speech.md#what-has-been-run-on-real-models)).

The full record, with the evidence for each row, is `planning/21_NATLAS_DISCOVERY.md`. See also [how AtlasForge integrates N-ATLaS](../concepts/n-atlas-integration.md#what-the-repositories-actually-say).

## Still open

- Does vLLM serve N-ATLaS unmodified?
- How does the LLM behave at fp16 and 4-bit through `transformers`, and how fast is it, on a GPU?
- What is the word error rate of each speech model on a real test set?
- Which task identifiers does the harness actually expose for the seven AfroBench-LITE families, and what does N-ATLaS score on them here? (`atlasforge bench afrobench --list` then answers the first half.)

## The run that answers them

If you have the hardware and the licence, `scripts/live_smoke.py` performs the checks above — plus
gated access on all five repositories and a commit SHA per repository — and writes the result to
`evidence/live_<date>.md`:

```bash
python scripts/live_smoke.py --out evidence
python scripts/live_smoke.py --quantize 4bit --audio-dir clips --base-url http://127.0.0.1:8000/v1
```

It needs `HF_TOKEN`, the Awarri licence accepted on every `NCAIR1` model page, and `pip install
"brainers-atlasforge[local]"`. `--audio-dir` wants one clip per language named `ha`, `yo`, `ig`,
`en`; `--base-url` is any OpenAI-compatible server, so a model served by vLLM, llama.cpp or Ollama
can be checked without loading it here.

**It has not been run yet, and this page is the record of that.** (The answers above came from manual runs, not from this script, so its evidence file does not exist yet.) Every row it writes is something
it observed; a check it could not perform is written down as `not run` with the reason, and stays an
open question above. The file quotes only the script's own fixed probe sentences and the model's
answers to them — no audio, no prompt of yours and no credential is in it.

## Known limitations

- **Classification** matches the label named in the answer. The default match ignores negation, so `"not positive"` counts as `positive`; `accuracy_strict` and `macro_f1_strict` require the whole answer to be a label and reject it. Neither understands "this is not negative, it is neutral" — use the loose metric beside the strict one and read the gap.
- **Speech** uses fixed 28-second windows by default; a word cut at a seam can be misheard. `--silence-aware` moves each cut to a nearby pause, but which splitter transcribes a given recording better is exactly what the live run above has not yet measured, so the fixed grid stays the default. The length buckets in the error analysis are words in the reference, not seconds: measuring duration would mean decoding audio during every `report`.
- **Normalisation** is rule-based and not configurable; spelling variants and dialects are not modelled.
- **Slices** are not adjusted for multiple comparisons.
- **Failure-mode flags** are conservative rules, not a quality score: they miss wrong answers that break no rule, and can flag a right answer. See [what they do not claim](../concepts/failure-modes.md).
- **HTML reports** are static: the charts are inline SVG with no zoom, tooltips or filtering, and the page exports nothing. It is a readable page, not a dashboard.
- **No language identification** anywhere. Declared languages are trusted, and the script check is script statistics only.

## Support matrix

| | Linux | macOS | Windows |
|---|---|---|---|
| Python 3.10 and 3.13 (CI) | ✓ | ✓ | ✓ |
| Evaluation, comparison, data checks, demo | ✓ | ✓ | ✓ |
| Local model loading / QLoRA | NVIDIA GPU | not for QLoRA | NVIDIA GPU |

## Reporting what you find

If you run the unverified parts on real hardware, the most useful thing you can do is share what happened, including failures. Open an issue with the output of `atlasforge doctor --json` and the exact command.
