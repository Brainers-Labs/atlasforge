# AtlasForge — what it does, and where it stands

*8 October 2026 · version `0.1.0a1` (pre-alpha, unreleased) · distribution `brainers-atlasforge`*

## The problem

N-ATLaS, Nigeria's open language and speech model for Hausa, Yoruba, Igbo and Nigerian-accented
English, is published as **gated open weights on Hugging Face. There is no API to call.** You can
download the model, but there is nothing to measure it with. A team that fine-tunes it has no way
to answer the only question that matters:

> *I changed this model. Did it actually get better on my task, and where did it get worse?*

AtlasForge is the tooling that answers it.

## What it is

Not a training framework, and not a leaderboard. It is an **evidence machine**: the unit of output
is an argued comparison with confidence intervals, not a single number. Every step leaves a file
that the next step reads, so a result can be traced back to the answers it came from.

```
dataset → validate → eval (run) → score  → report.md / .json / .html
                                 ↘ compare ─────────────────────────→ comparison.md / .json / .html
finetune → LoRA adapter → ──────────────────┘                              ↘ model card
```

## The pipeline

**1. Dataset** — JSONL, one example per line: `id`, `input` (or `messages`), `reference`, `lang`,
`meta`; speech uses `audio` instead of `input`. The loader is deliberately strict: **unknown keys
are rejected**, so a typo'd field fails loudly rather than quietly becoming a missing input. Errors
are line-numbered, ids must be unique, and the file gets a content fingerprint.

**2. `dataset validate`** — finds malformed lines, duplicates, conflicting labels, train/test
leakage, broken Unicode, stripped diacritics and class imbalance *before* a model is involved.

**3. `eval`** — sends each example to a model and records the raw answer. The run directory is the
unit of work and it is resumable: `run.json` is a manifest, `results.jsonl` is one line per example.
It appends crash-safely, isolates per-example errors instead of dying on one, and stops after 20
consecutive failures rather than hammering a dead server. Two backends:

- `openai` — any OpenAI-compatible server: vLLM, llama.cpp, Ollama, HF Endpoints, a gateway.
- `local` — loads the weights directly with transformers.

**4. Scoring** — every text metric is computed **twice**, under a tone-aware view and a
tone-insensitive view. This is the part that is specific to these languages: Yoruba and Igbo
underdots and Hausa hooked letters are never silently stripped, so `ẹ` against `e` is a measurable
difference rather than an invisible one. Failed and missing answers count as **wrong**, not as
omitted.

**5. Reports** — `report.md`, `report.json` and `report.html`. The HTML is **one self-contained
file**: inline SVG, no JavaScript, no font, image or stylesheet fetched from anywhere, with escaping
tested against hostile text. It opens from a filesystem, air-gapped, or as an email attachment.

**6. `compare`** — the pay-off. Paired bootstrap confidence intervals, an exact McNemar test for
right/wrong metrics, per-slice breakdowns (language, length, has-number, or any `meta` field), and
failure-mode flags compared run by run. A change is called *improved* or *regressed* only when the
**whole interval** sits on one side of zero. A slice with fewer than 30 examples is reported as
*insufficient data* rather than given a verdict.

Around that loop: `finetune` (QLoRA to a LoRA adapter, evaluated back through `--adapter`), `card`
(a licence-aware model card carrying attribution, "Powered by Awarri" and the 1,000-user cap),
`transcribe` (speech, with automatic splitting past the models' 30-second limit), `bench afrobench`,
`demo`, and `doctor`.

## What you can run today

```bash
atlasforge doctor                                   # Python, GPU/VRAM, disk, ffmpeg, HF token, extras
atlasforge dataset validate data.jsonl --task generation [--against test.jsonl]
atlasforge eval data.jsonl --out runs/base --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
atlasforge eval data.jsonl --out runs/tuned --base-url http://127.0.0.1:8001/v1 --model my-finetune
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --out cmp --slice domain
atlasforge report runs/base --dataset data.jsonl     # re-score with no model
atlasforge demo                                      # synthetic data and two finished runs
```

The `demo` command writes no model and needs no network, so a reader can reach a real comparison in
about two minutes. A notebook runs the same workflow on the same data.

## Three other surfaces

- **Python API** — `atlasforge.evaluate()`, `compare_runs()`, `score_finished_run()`,
  `write_reports()`. Importing `atlasforge` is cheap and pulls in **no** machine-learning framework.
- **Configuration** — an optional `atlasforge.toml`; precedence is flag → environment → file →
  default. Seven keys, and an unknown key is an error.
- **Documentation** — 41 pages. The CLI reference, the metrics, flags, config and error tables and
  every terminal transcript are **generated from the code**, and the test suite fails if a documented
  command or flag does not exist.

## The design spine: what it refuses to do

These are the rules that hold everywhere in the code, and they are the most distinctive thing about
it:

- **An absent number is not a zero.** A benchmark family the harness has no task for is reported as
  a gap, in as many words.
- **A timeout is not a denial.** The gated-access check reports *unknown*, never *denied*.
- **Not overridden is not zero-shot.** `few_shot: null` means `--num_fewshot` was not passed, so each
  task ran at the harness's own default — a different claim from "zero".
- **The report names the file it actually read**, not the one it asked for.
- **A check that cannot run is recorded as `not run`, with the reason** — never as a plausible
  figure.
- **Nothing is redistributed.** Weights are never copied, and another study's published figures are
  never copied into a report of ours.

## Current state

**Green:** 1,332 tests, 98.64% statement coverage, `ruff` and strict `mypy` clean, `mkdocs build
--strict` passing, on Python 3.10–3.13 across Linux, macOS and Windows. 46 source files (~9,500
lines), 33 test files (~9,800 lines).

**Verified by automated test** — datasets and validation; both normalisation views; all built-in
metrics plus user-written ones; the `eval` runner end to end against a real HTTP server with a fake
model; paired statistics, slices and failure-mode flags; all three report formats; audio decoding,
windowing and transcript merging with real ffmpeg; speech error analysis; the config file; model
cards; and `finetune --dry-run`. Where a model is involved, the tests use a stand-in that answers
from fixed rules.

**Never run against the real weights.** The evidence log is empty — 0 of 8 checks:

| Not yet run | Why it cannot run here |
|---|---|
| `--backend local` | needs the 8B model in memory |
| The official speech models | needs the gated `NCAIR1` weights |
| `finetune` training | needs an NVIDIA GPU |
| `--adapter` evaluation | needs a real adapter |
| `bench afrobench` against the real harness | needs the `bench` extra, a GPU and the weights |
| Serving N-ATLaS with vLLM, llama.cpp or Ollama | follows each tool's docs; unconfirmed |

The instrument that closes this is written and tested — a script runs the eight checks and writes a
dated log, recording a check it could not perform as `not run` with its reason. **It has never been
executed.** Closing this needs the hardware, a token and the licence accepted on all five
repositories. No further code closes it.

### Open gaps

- **Real-model evidence** — the run above.
- **External testers** — at least two are required for the submission; none recruited yet.
- A Hausa quickstart needs a named native-speaker reviewer.
- The benchmark-pack licence question is a share-alike decision inside an Apache-2.0 repository,
  deliberately not guessed.
- The notebook has not been opened in Colab. The documentation is not hosted. No release tag exists,
  because the tag waits on the first item.

## Licence

AtlasForge's own code is Apache-2.0. The models are not ours and are not redistributed: you download
them from Hugging Face after accepting the Awarri licence yourself, which carries attribution
requirements, a "Powered by Awarri" suffix for derivatives, and a 1,000 active-end-user cap.
