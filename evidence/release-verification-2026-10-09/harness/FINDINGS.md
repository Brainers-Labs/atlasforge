## Summary

**Is it ready to release to the public? Almost, but not as published today.** The tool's core is sound and well proved: the statistics, scoring, resilience, secret handling, data validation, the real LLM and real speech models all behaved as claimed under independent checking. However, the published package has one release-blocking defect for Mac users, one provenance bug that exists in both the published and unreleased builds, and public text that is now out of date. None of these needs more than a small change, but they should be fixed and re-tested before announcing publicly.

What was proved, with real models and independent recomputation (details in the tables below):

- **Statistics are correct.** Means, win/tie/loss counts, the exact McNemar p-value and the 95% bootstrap interval all match independent re-computations; results are deterministic for a seed; a run compared with itself gives a zero-width interval.
- **Tone-aware scoring does what it claims.** Precomposed and combining Unicode score identically; tone marks matter in one view and not the other; Yoruba and Igbo underdots and Hausa hooked letters are never stripped; WER, CER, chrF, accuracy and macro-F1 match hand calculations.
- **It is resilient.** After a `kill -9` it resumes without re-running finished examples; it refuses to mix runs from a different model, dataset or settings; a dead server trips the circuit breaker after 5 requests instead of 40; failed examples count as wrong; timeouts and retries behave as documented; concurrency gives identical scores.
- **Secrets stay secret.** An API key and a Hugging Face token never appeared in any output, manifest, result or report; server error bodies (which can echo prompts) never reached output; hostile text in a model name, answer or slice value is escaped in the HTML reports; plain `http://` to a non-local host is refused.
- **The real models work.** All four official speech models transcribed real audio through `atlasforge`. One Hausa clip was also run as ogg/opus, mp3, m4a and flac and gave exactly the same transcript as the wav (WER 0.0 between them). Recordings over the 30 second limit (a 79 second file and three 31 to 34 second Igbo clips) were split and merged with no duplicated passages at the seams. The real N-ATLaS LLM (int4 import) answered in English, Hausa, Yoruba and Igbo, completed a 60 question evaluation with no failures, and a real comparison of two runs gave verdicts and per-domain slices with real statistics.
- **The package is clean.** The wheel is 134 KB and contains no weights or audio; only the five official `NCAIR1` identifiers appear in the shipped code; no LLM judge or third party model API is referenced.
- **The public surface is up.** The live docs site (42 pages, 2,488 internal links, none broken), the PyPI page, the public repository, the Colab link, and CI green on Linux, macOS and Windows for Python 3.10 and 3.13. The quickstart notebook ran top to bottom in a clean environment using the published package.

## Findings, most important first

### F1. HIGH: the published `[asr]` extra is broken on this Mac (fixed in the repo, not yet released)

`pip install "brainers-atlasforge[asr]"` from PyPI installs `scipy 1.15.3`, which fails to load on this machine (macOS 27.0, Apple Silicon), so **every speech command and `--backend local` run fails** with `Could not load NCAIR1/Hausa-ASR (ImportError)` (check 5.01, build "PyPI `[asr]` as installed"). The root cause was reproduced directly (`dlopen ... _propack ... zero-fill section`, log `pypi-asr__root_cause.log`). With the documented workaround (`pip install "scipy<1.15"`) 22 of the 23 speech checks on the published package pass (the one failure is F2), and the **repo build installs `scipy 1.14.1` automatically and passes 5.01 on a fresh install**, so the fix already exists in the repository. **It is not in the published 0.1.0a2.**

- *Caveat:* observed on one very new macOS release on Apple Silicon. It is not known whether older macOS versions are affected.
- *Action:* publish a new version containing the pin (0.1.0a3) before announcing; re-run this suite against it from PyPI.

### F2. MEDIUM: speech runs record the wrong model in `run.json` and the reports (both builds)

For `eval --task asr --backend local`, `run.json` and the report header say **Model: NCAIR1/N-ATLaS** (the LLM), although `NCAIR1/Hausa-ASR` actually transcribed (check 5.20 fails on both the published and the repo build). A reader of the report would conclude the wrong model produced the transcripts, which undermines the "every report names the weights that produced it" provenance claim. The numbers themselves are correct (they matched an independent computation).

- *Action:* record the speech repository (and revision) actually used, per language, in the manifest and the report header. This is a code change; **it has not been made**.

### F3. MEDIUM: out of date public text

- The **PyPI project description** (frozen at 0.1.0a2) and the **repository README** still say the speech models and the local backend "have never been run against real weights". That is no longer true. PyPI's text can only change by publishing a new release; the README is corrected in the same commit as this report.
- The live docs were also stale until the push of 28030de; after its deploy the status and integration pages reflect the verified facts (check 8.04 passes).

### F4. LOW: fixes that exist only in the repository

`dataset validate` prints the line number twice (`line 2: line 2: ...`) in the published 0.1.0a2 (check 3.04 fails there, passes in the repo build). Released with the next version.

### F5. LOW: `finetune` without an NVIDIA GPU can hang or crash with no message

With `quantize: none` on a Mac, `atlasforge finetune` hung or died inside `torch` while loading the model (check 9.03b). The default 4-bit path has a clear refusal ("needs an NVIDIA GPU"); the non-4-bit path has none. The same crash reproduces with plain `transformers`, so the cause is the environment, but a guard that refuses non-CUDA machines for every setting would turn a hang into a message. The documentation already says training needs an NVIDIA GPU.

### F6. INFO: smaller observations

- The `scipy<1.15` macOS pin has no wheels for Python 3.14, so `pip install` of `[asr]` or `[local]` on macOS with 3.14 tries to build scipy from source and fails. Python 3.14 is outside the supported 3.10 to 3.13, but a marker such as `python_version < '3.14'` would avoid the surprise.
- `atlasforge run -` sends the trailing newline of standard input to the model (`'hello\n'`).
- When a server returns HTTP 401 for every request, the circuit-breaker abort message names HTTP 401 but not the `ATLASFORGE_API_KEY` hint that a single 401 error would show.
- `doctor` only recognises NVIDIA GPUs, so it always warns on Apple Silicon (documented).

## Real model results

**Speech (formal, `atlasforge eval --task asr`, 10 real FLEURS clips per language, indicative only):** see the table in this report. The Yoruba model shows why the two tone views matter in practice: WER is 0.612 when tone marks must match but 0.457 when they are ignored, a 15 point gap that a single-number scorer would hide. English was measured on US-accent audio, so it says nothing about Nigerian-accented speech. Pooled WER from the tool equals an independent `jiwer` computation for all four languages once the documented apostrophe rule is applied.

**Long audio:** a 79 second Hausa recording was split into 3 overlapping windows and merged with no repeated passage (WER 0.378 against the concatenated reference); `--silence-aware` placed both interior cuts inside the silent gaps (WER 0.396).

**LLM (int4 import of the official weights, served by Ollama):** 60 real questions (30 African capitals, 30 additions) completed with 0 failures, mean latency 323 ms. Comparing the build **without** the chat template against the build **with** it, the paired comparison reported exact match improving by 96.7 points (verdict *improved*, both 30 question domain slices *improved*), which independently confirms the documentation's warning that an Ollama import must have its chat template restored. A 12 review classification run scored accuracy 1.0 and macro-F1 1.0. These are tests of the tool on a quantised copy, not an assessment of official model quality.

## What was not tested

Real LoRA/QLoRA training and 4-bit loading (no NVIDIA GPU; an attempt crashed inside `torch`); N-ATLaS at full precision; vLLM and llama.cpp serving of N-ATLaS; the AfroBench-LITE wrapper with the real harness; Nigerian-accented speech; speech over HTTP against a real endpoint (a stub was used); HTTPS/TLS; Linux and Windows locally (CI only); Python 3.11, 3.12 and 3.13 locally; the Colab runtime itself (the notebook ran locally); `scripts/live_smoke.py`; external beta testers and any real-user usability evidence; and security testing beyond secret-handling and escaping checks.

## Before announcing publicly

1. Publish a release that contains the macOS `scipy` pin, the `validate` fix and the refreshed README (F1, F3, F4), then re-run this suite against the new release **from PyPI**.
2. Fix the speech provenance bug (F2) in the same release if possible.
3. Get at least two external beta testers to run the quickstart, and record their sessions.
4. Optionally verify fine-tuning, vLLM and full precision on a machine with an NVIDIA GPU.
