# Status and roadmap

AtlasForge is **pre-alpha** (`0.1.0.dev0`), built for the NAIC 2026 competition, Problem 01 (Developer Infrastructure). This page is deliberately blunt about what works, what has only been tested with stand-ins, and what does not exist yet.

*Last updated: 2 October 2026.*

## What the labels mean

| Label | Meaning |
|---|---|
| **Verified** | exercised for real: against real model weights, or by the automated test suite where no model is involved |
| **Tested with stand-ins** | the logic is tested, but never run against the real thing |
| **Not written** | does not exist yet |

## Feature status

### Evaluation and comparison (no model needed to test)

| Feature | Status |
|---|---|
| JSONL dataset loader with line-numbered errors, strict keys | Verified (tests) |
| Resumable runner: manifest guard, crash-safe results, circuit breaker, bounded threading | Verified (tests, including end-to-end against a real HTTP server) |
| Metrics: exact match, accuracy, macro-F1, chrF/chrF++, WER, CER, latency percentiles | Verified (tests) |
| Tone-aware and tone-insensitive scoring; failures count as wrong | Verified (tests) |
| Reports (Markdown and JSON) | Verified (tests) |
| `report`: re-score without a model | Verified (tests) |
| `compare`: paired bootstrap, exact McNemar, slice analysis with the 30-example rule | Verified (tests) |
| `dataset validate`: duplicates, conflicts, leakage, Unicode, diacritics, class balance | Verified (tests) |
| `doctor` | Verified (tests; run on a real Mac) |

### Backends and models

| Feature | Status |
|---|---|
| `openai` backend (any OpenAI-compatible server) | Verified: tests against a real HTTP server; and **run for real** against an int4 N-ATLaS served by Ollama |
| `local` backend, **speech models** | **Verified for real** on a Mac M1 (Apple MPS and CPU) |
| `local` backend, **LLM** | Tested with stand-ins only. Needs a GPU of 24 GB or more |
| ASR: ffmpeg decoding, 30 s windowing, seam-aware merging | Verified (tests, real opus/ogg decoding, and real clips) |
| ASR over the `openai` backend (`/audio/transcriptions`) | Tested against a stub server only |
| vLLM serving of N-ATLaS | Not tested (no NVIDIA GPU yet) |
| llama.cpp serving | Not tested |

### Not written yet

| Feature | Notes |
|---|---|
| Fine-tuning recipes (`finetune/`) | LLM QLoRA and ASR recipes planned; see [Fine-tuning](../guides/fine-tuning.md) |
| `atlasforge card` | licence-aware model card for adapters |
| `bench afrobench` wrapper | |
| HTML report | Markdown and JSON exist |
| Colab notebooks | |
| `scripts/live_smoke.py` | a script to generate dated live-evidence files |
| Hausa (and Yoruba) translation of the quickstart | needs a named human reviewer; machine translation without review is not acceptable |
| Publication to PyPI | the package name has not been confirmed free |

## Quality bar

- 559 automated tests, about 98% line coverage.
- `ruff` (lint and format) and `mypy --strict` are clean.
- CI runs Linux, macOS and Windows on Python 3.10 and 3.13, a "core install stays light" job, and a secrets and dependency audit.
- The default test suite needs no GPU, credentials or gated weights.

## Known limitations

- Long-audio chunking uses fixed 28 s windows; a word on a boundary can be misheard.
- Classification label extraction does not understand negation.
- Slice intervals are not corrected for multiple comparisons (the report says so).
- `compare` cannot compare prompt variants (different datasets).
- `doctor` only detects NVIDIA GPUs.
- The `transcribe` command does not expose model timestamps; its `chunks` are AtlasForge's own windows.
- No language identification; `dataset validate` trusts the declared `lang`.

## Open items before a release

1. Replace placeholders: the security contact in `SECURITY.md`, and the GitHub organisation in `pyproject.toml`.
2. Confirm the package name is available on PyPI.
3. Secure a GPU to verify fp16 loading and vLLM, and to smoke-test fine-tuning.
4. Choose a small, clean-licence domain dataset for a flagship compare demo. No data is invented for this.
5. Run `eval` and `compare` on a real dataset.
6. Recruit at least two external beta testers and record real evidence.

## Roadmap

| Milestone | Scope |
|---|---|
| **v0.1.0a1** | quickstart good enough for a stranger; first beta round |
| **v0.1.0rc1** | `compare` and speech evaluation fixed from beta feedback; fine-tune recipe smoke-tested; `card` |
| **v0.1.0** | released to PyPI and GitHub; docs live |

Dates and the day-by-day plan are in the repository's `planning/` folder.
