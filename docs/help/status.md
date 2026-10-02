# Project status

AtlasForge is **pre-alpha ({{ version }})**. This page says plainly what has been proved and what has not, so you can decide what to trust.

## What "verified" means here

<span class="status-verified">Tested</span> The behaviour is covered by automated tests that run on Linux, macOS and Windows with Python 3.10 and 3.13. Where a model is involved, the tests use a stand-in that answers from fixed rules.

<span class="status-unverified">Not run on real weights</span> The code is written and its wiring is tested against stand-ins, but it has **never been run against the real N-ATLaS models**. It may need adjustment on first contact.

## Components

| Component | Status | Notes |
|---|---|---|
| Datasets: loading, strict validation, fingerprinting | <span class="status-verified">Tested</span> | |
| `dataset validate` | <span class="status-verified">Tested</span> | Duplicates, conflicts, leakage, Unicode, diacritics, balance |
| Tone-aware / tone-insensitive normalisation | <span class="status-verified">Tested</span> | |
| Metrics: exact match, accuracy, macro-F1, chrF, WER, CER | <span class="status-verified">Tested</span> | |
| `eval`: resumable runs, circuit breaker, concurrency | <span class="status-verified">Tested</span> | End to end against a real HTTP server with a fake model |
| `report`, `compare`: bootstrap, McNemar, slices | <span class="status-verified">Tested</span> | |
| `openai` backend (any OpenAI-compatible server) | <span class="status-verified">Tested</span> | Retries, errors, auth, https guard. **Not tested against N-ATLaS served by vLLM/llama.cpp/Ollama.** |
| `demo` data | <span class="status-verified">Tested</span> | Synthetic by design |
| Audio decoding, windowing, transcript merging | <span class="status-verified">Tested</span> | With real ffmpeg, including opus `.ogg` |
| `card` | <span class="status-verified">Tested</span> | |
| `finetune --dry-run` and data checks | <span class="status-verified">Tested</span> | |
| `local` backend (Transformers) | <span class="status-unverified">Not run on real weights</span> | Needs the 8B model |
| Official ASR models | <span class="status-unverified">Not run on real weights</span> | Needs the `NCAIR1` speech models |
| `finetune` training run | <span class="status-unverified">Not run on real weights</span> | Needs an NVIDIA GPU |
| `--adapter` evaluation | <span class="status-unverified">Not run on real weights</span> | |
| Serving N-ATLaS with vLLM, llama.cpp, Ollama | <span class="status-unverified">Not run on real weights</span> | Commands follow each tool's docs |

## Open questions about the real models

These are the facts that can only be settled by running the models, and are being recorded as they are verified:

- Does the N-ATLaS tokenizer ship a chat template, and what does it look like?
- What is the true context length? (The model card states 8,092; 8,192 is likely.)
- How much memory does it need in fp16 and in 4-bit?
- Do the speech models return usable timestamps? (Not assumed.)
- Does vLLM serve N-ATLaS unmodified?

## Known limitations

- **Classification** matches the label named in the answer and ignores negation.
- **Speech** uses fixed 28-second windows; a word cut at a seam can be misheard.
- **Normalisation** is rule-based and not configurable; spelling variants and dialects are not modelled.
- **Slices** are not adjusted for multiple comparisons.
- **Custom metrics** are not supported yet.
- **No HTML report or web viewer** yet; reports are Markdown and JSON.
- **No language identification** anywhere. Declared languages are trusted.

## Support matrix

| | Linux | macOS | Windows |
|---|---|---|---|
| Python 3.10 and 3.13 (CI) | ✓ | ✓ | ✓ |
| Evaluation, comparison, data checks, demo | ✓ | ✓ | ✓ |
| Local model loading / QLoRA | NVIDIA GPU | not for QLoRA | NVIDIA GPU |

## Reporting what you find

If you run the unverified parts on real hardware, the most useful thing you can do is share what happened, including failures. Open an issue with the output of `atlasforge doctor --json` and the exact command.
