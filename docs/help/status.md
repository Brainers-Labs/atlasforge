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
| `local` backend (Transformers) | <span class="status-unverified">Not run on real weights</span> | Needs the 8B model |
| Official ASR models | <span class="status-unverified">Not run on real weights</span> | Needs the `NCAIR1` speech models |
| `finetune` training run | <span class="status-unverified">Not run on real weights</span> | Needs an NVIDIA GPU |
| `bench afrobench` (AfroBench-LITE wrapper) | <span class="status-unverified">Not run against the harness</span> | Wiring tested against stand-ins. Tasks are discovered from the installed `lm-evaluation-harness`, never hard-coded; needs the `bench` extra, a GPU and the gated weights |
| `--adapter` evaluation | <span class="status-unverified">Not run on real weights</span> | |
| Serving N-ATLaS with vLLM, llama.cpp, Ollama | <span class="status-unverified">Not run on real weights</span> | Commands follow each tool's docs |

## Open questions about the real models

These are the facts that can only be settled by running the models, and are being recorded as they are verified:

- Does the N-ATLaS tokenizer ship a chat template, and what does it look like?
- What is the true context length? (The model card states 8,092; 8,192 is likely.)
- How much memory does it need in fp16 and in 4-bit?
- Do the speech models return usable timestamps? (Not assumed.)
- Does vLLM serve N-ATLaS unmodified?
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

**It has not been run yet, and this page is the record of that.** Every row it writes is something
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
