# AtlasForge

Run, evaluate, compare and fine-tune the official **N-ATLaS** models (Hausa, Yoruba, Igbo, Nigerian-accented English).

> **Status: pre-alpha (`0.1.0a2`, published to PyPI as `brainers-atlasforge`).** Built for NAIC 2026, Problem 01 (Developer Infrastructure). Everything under "Works today" is tested. The four official speech models, the local backend for speech, and the LLM (as an int4 copy served by Ollama) have been run for real on a Mac; the LLM at full precision, vLLM, and the fine-tuning training run have **not**, because they need an NVIDIA GPU. The full record, with every check and its evidence, is the [release verification report](https://github.com/Brainers-Labs/atlasforge/blob/main/evidence/release-verification-2026-10-09/REPORT.md). Nothing is claimed to work until it is listed here.

## Documentation

Full documentation (quickstart, concepts, guides, reference, troubleshooting) lives in [`docs/`](https://github.com/Brainers-Labs/atlasforge/blob/main/docs/index.md) and builds into a searchable site:

```bash
pip install -e ".[docs]"
mkdocs serve        # live preview
```

No model? Start with the offline demo: `atlasforge demo`. A Jupyter/Colab walkthrough of the whole workflow on that demo data is in [`notebooks/`](https://github.com/Brainers-Labs/atlasforge/blob/main/notebooks/README.md), and the output it produces — a report and a comparison in Markdown, JSON and HTML — is committed in [`examples/reports/`](https://github.com/Brainers-Labs/atlasforge/blob/main/examples/README.md).

## The question AtlasForge answers

> *I changed this N-ATLaS model. Did I actually make it better on my task, and where did it get worse?*

N-ATLaS is published as gated open weights on Hugging Face, so developers get the models but no tooling to measure them on their own data. AtlasForge is that tooling.

## Install

Python 3.10 to 3.13.

```bash
pip install --pre brainers-atlasforge
atlasforge --version
```

The distribution is `brainers-atlasforge`, not `atlasforge` — that name on PyPI belongs to an unrelated bioinformatics project. The import name and the command are both `atlasforge`. `--pre` is written above but is not required while the alpha is the only version published: pip falls back to pre-releases when a project has no stable release to prefer, so the plain command installs it too. It becomes load-bearing as soon as a stable version exists, because pip will then choose that one.

To work on AtlasForge itself, or to run the tip of `main` rather than a release:

```bash
git clone https://github.com/Brainers-Labs/atlasforge
cd atlasforge
python -m venv .venv
. .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

That base install pulls in no PyTorch and no other machine-learning framework. Extras add one job at a time — `pip install "brainers-atlasforge[local]"` to run the weights in-process, `[asr]` for speech, `[finetune]` for QLoRA. [Installation](https://github.com/Brainers-Labs/atlasforge/blob/main/docs/get-started/installation.md) lists them all, and the [Quickstart](https://github.com/Brainers-Labs/atlasforge/blob/main/docs/get-started/quickstart.md) reaches a real comparison in about two minutes with no model at all.

## Works today

```bash
atlasforge doctor                                   # Python, GPU/VRAM, disk, ffmpeg, HF token, extras
atlasforge run "Ina kwana?" --base-url http://127.0.0.1:8000/v1
atlasforge dataset validate data.jsonl --task generation [--against test.jsonl]
atlasforge eval data.jsonl --out runs/base --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
atlasforge eval data.jsonl --out runs/tuned --base-url http://127.0.0.1:8001/v1 --model my-finetune
atlasforge compare data.jsonl --base runs/base --candidate runs/tuned --out cmp --slice domain
atlasforge report runs/base --dataset data.jsonl    # re-score with no model
atlasforge transcribe note.ogg --lang ha --base-url ...
```

- **`eval`**: JSONL in, resumable run out, `report.md` + `report.json` + `report.html` (one self-contained page: no JavaScript, no network, no server). Every text metric is reported under both a **tone-aware** and a **tone-insensitive** view (Yoruba/Igbo underdots and Hausa hooked letters are never stripped). Failed calls count as wrong.

- **`compare`**: paired bootstrap confidence intervals, an exact McNemar test for right/wrong metrics, per-slice results (language, length, has-number, or any `meta` field), and the same three report shapes as `eval`. A change is only called *improved* or *regressed* when the whole interval is on one side of zero, and slices under 30 examples are reported as *insufficient data*.
- **`dataset validate`**: malformed lines (all of them, with line numbers), duplicates, conflicting labels, train/test leakage, broken Unicode, stripped diacritics, class imbalance.
- **Backends**: `openai` works with any OpenAI-compatible server (vLLM, llama.cpp, Ollama, HF Endpoints, community gateways). It retries 429/5xx and connection errors, and never puts response bodies in error messages.
- **Safety nets**: `eval` stops after 20 consecutive failures instead of hammering a dead server, and keeps everything finished so it can resume.
- **`finetune --dry-run`** (no GPU): validates the training data, refuses train/test leakage and tiny datasets, and prints the plan. **`card`** writes a model card with the licence obligations (attribution, "Powered by Awarri", the 1,000-user cap), the evaluation numbers and any regressions.
- **ASR helpers**: any audio format via ffmpeg (including WhatsApp `.ogg`/opus), automatic splitting of audio over the models' 30-second limit, transcript merging.

## Verified on real weights, and what is not

Run for real on a MacBook (Apple M1, 16 GB) and recorded in the [release verification report](https://github.com/Brainers-Labs/atlasforge/blob/main/evidence/release-verification-2026-10-09/REPORT.md):

- All four official speech models, through `atlasforge transcribe` and `atlasforge eval --task asr` on real FLEURS audio (10 clips per language, indicative only): Hausa WER 0.29, Yoruba 0.61 (0.46 when tone marks are ignored), Igbo 0.41, English 0.085 on US-accent audio. Long recordings are split and merged correctly.
- The official LLM as an int4 import served by Ollama, through `run`, `eval`, `compare` and a classification run. This is a quantised copy, so it says nothing about full-precision quality.
- Loading and evaluating a LoRA adapter through `--backend local --adapter`, with a tiny stand-in model (not N-ATLaS).

**Not run**, and why:

- `atlasforge finetune` training: it needs an NVIDIA GPU; data checks and `--dry-run` are tested.
- The LLM at full precision through `--backend local` (16 GB of weights; does not fit a 16 GB machine).
- vLLM and llama.cpp serving, and the AfroBench-LITE wrapper against the real harness.
- Nigerian-accented speech accuracy, and any real-user (beta tester) evidence.

`scripts/live_smoke.py`, which writes a dated evidence file on a machine that has the weights, has not been run.

## Metrics

exact match, accuracy, macro-F1, chrF / chrF++, WER, CER, latency percentiles. Datasets are JSONL, one example per line: `id`, `input` (or `messages`), `reference`, `lang`, `meta`; ASR uses `audio` instead of `input`. Unknown keys are rejected, so typos fail loudly.

## Models and licence

AtlasForge **never redistributes** N-ATLaS weights. You download them from Hugging Face after accepting the Awarri licence yourself. The licence includes a 1,000 active-end-user cap, attribution requirements and a "Powered by Awarri" suffix for derivatives. Read it before you build on the models. AtlasForge's own code is Apache-2.0.

## Development

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy
```

The default test suite needs no GPU, credentials or gated weights. Extras: `[local]` (transformers backend), `[asr]`, `[finetune]`. ffmpeg is needed for audio.
