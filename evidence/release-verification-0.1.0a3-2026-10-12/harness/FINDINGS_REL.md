## Summary

**AtlasForge 0.1.0a3 is released and verified.** It was published to PyPI on 12 October 2026 and installed from there, in fresh virtual environments with a plain `pip install` and no workarounds, then put through the same black-box suite that found the problems in 0.1.0a2, plus new checks for each fix. **Every check that can run on this machine passes (199 of 199 distinct checks). The 3 that cannot run need an NVIDIA GPU.** Each defect found in 0.1.0a2 is fixed, and the fix is shown by a check on the released package, not only by a unit test.

## What was wrong in 0.1.0a2, and what is true now

| # | Finding in 0.1.0a2 | Severity | State in 0.1.0a3 | Proof on the release |
|---|---|---|---|---|
| F1 | `pip install "brainers-atlasforge[asr]"` failed to load on Apple Silicon macOS (scipy 1.15.3), so every speech command broke | High | **Fixed.** The `scipy<1.15` pin is released; a plain install resolves `scipy 1.14.1` | 5.01 passes on a fresh install with no workaround, and all 23 speech checks pass |
| F2 | Speech runs recorded `NCAIR1/N-ATLaS` (the LLM) as the model in `run.json` and the report | Medium | **Fixed.** A speech run names the speech model per language | 5.20 passes; 11.05 on a real mixed Hausa and Yoruba run records `NCAIR1/Hausa-ASR, NCAIR1/Yoruba-ASR` |
| F3 | The PyPI description and the README said the speech models were never run on real weights | Medium | **Fixed.** The new description states what was run and what was not, and links the verification report | 8.06 passes: none of the stale phrases appear on the PyPI page |
| F4 | `dataset validate` printed the line number twice | Low | **Fixed** | 3.04 passes on the release |
| F5 | `finetune` without a GPU hung or crashed on a Mac when `quantize: none` | Low | **Fixed.** It refuses at once with a clear message for every setting, before the other heavy imports | 9.03 and 9.03b: exit 2 in 1.0 s with the NVIDIA message; 7.05 passes |
| F6a | The macOS scipy pin made `pip install` fail on Python 3.14 | Info | **Fixed.** The pin stops at Python 3.13 | 11.04: `[finetune]` installs on macOS Python 3.14 |
| F6b | `run -` sent the pipe's trailing newline to the model | Info | **Fixed** | 11.01 |
| F6c | An all-401 abort did not mention the API key | Info | **Fixed.** The hint names `ATLASFORGE_API_KEY` | 11.02 |
| F6d | `doctor` said only "no NVIDIA GPU" on an Apple Silicon Mac | Info | **Improved.** It notes Apple Silicon and says the speech models can use the Apple GPU | 11.03 |

The first run of this release also exposed a mistake of mine, which is worth recording: a new end-to-end test needed `ffmpeg`, which CI runners lack, so CI failed on a push. It was caught before tagging, fixed by giving the test the same skip marker as its neighbours, and the suite is now run both with and without `ffmpeg` before every push.

## What was proved on the release

- **Install:** `pip install brainers-atlasforge`, `[asr]` and `[finetune]` (Python 3.14) all succeed from PyPI with no workaround; the core install pulls in no ML framework.
- **Statistics and scoring:** unchanged and still matching independent recomputation (means, win/tie/loss counts, exact McNemar p-value, bootstrap interval, tone-aware and tone-insensitive scoring, WER, CER, chrF, accuracy and macro-F1).
- **Resilience and secrets:** resume after `kill -9`, the circuit breaker, retries, timeouts, concurrency, no secret or response body in any output, hostile text escaped in the HTML reports.
- **Real models:** all four official speech models, the long-audio chunking, and the real LLM (int4 import) through `run`, `eval` and `compare`.
- **The public surface:** the live docs (42 pages, 2,488 internal links, none broken), the PyPI page, CI green on Linux, macOS and Windows for Python 3.10 and 3.13, and the quickstart notebook run from the published package.

Real speech results (10 FLEURS clips per language, indicative only) are unchanged from 0.1.0a2: Hausa WER 0.291, Yoruba 0.612 (0.457 ignoring tone marks), Igbo 0.409, English 0.085 on US-accent audio.

## What is still not tested

Real LoRA or QLoRA training and 4-bit loading (no NVIDIA GPU here), N-ATLaS at full precision, vLLM and llama.cpp serving, the AfroBench-LITE wrapper with the real harness, Nigerian-accented speech, speech over HTTP against a real endpoint (a stub was used), HTTPS/TLS, Linux and Windows locally (CI only), Python 3.11 to 3.13 locally, the Colab runtime itself, and any real-user (beta tester) evidence. The README and docs state the same limits.

## Before announcing

1. Get at least two external beta testers to run the quickstart and record their sessions (the NAIC beta-test requirement is still unmet).
2. Optionally verify fine-tuning, vLLM and full precision on a machine with an NVIDIA GPU.
