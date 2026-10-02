# AtlasForge

Run, evaluate, compare and fine-tune the official **N-ATLaS** models (Hausa, Yoruba, Igbo, Nigerian-accented English).

> **Status: pre-alpha (`0.1.0.dev0`).** Built for NAIC 2026, Problem 01 (Developer Infrastructure). Everything under "Works today" is tested. Several parts have **never been run against real N-ATLaS weights** because they need a GPU or a large-memory machine: the `local` backend, the ASR models and fine-tuning. They are marked below. Nothing is claimed to work until it is listed here.

## The question AtlasForge answers

> *I changed this N-ATLaS model. Did I actually make it better on my task, and where did it get worse?*

N-ATLaS is published as gated open weights on Hugging Face, so developers get the models but no tooling to measure them on their own data. AtlasForge is that tooling.

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

- **`eval`**: JSONL in, resumable run out, `report.md` + `report.json`. Every text metric is reported under both a **tone-aware** and a **tone-insensitive** view (Yoruba/Igbo underdots and Hausa hooked letters are never stripped). Failed calls count as wrong.
- **`compare`**: paired bootstrap confidence intervals, an exact McNemar test for right/wrong metrics, per-slice results (language, length, has-number, or any `meta` field). A change is only called *improved* or *regressed* when the whole interval is on one side of zero, and slices under 30 examples are reported as *insufficient data*.
- **`dataset validate`**: malformed lines (all of them, with line numbers), duplicates, conflicting labels, train/test leakage, broken Unicode, stripped diacritics, class imbalance.
- **Backends**: `openai` works with any OpenAI-compatible server (vLLM, llama.cpp, Ollama, HF Endpoints, community gateways). It retries 429/5xx and connection errors, and never puts response bodies in error messages.
- **Safety nets**: `eval` stops after 20 consecutive failures instead of hammering a dead server, and keeps everything finished so it can resume.
- **`finetune --dry-run`** (no GPU): validates the training data, refuses train/test leakage and tiny datasets, and prints the plan. **`card`** writes a model card with the licence obligations (attribution, "Powered by Awarri", the 1,000-user cap), the evaluation numbers and any regressions.
- **ASR helpers**: any audio format via ffmpeg (including WhatsApp `.ogg`/opus), automatic splitting of audio over the models' 30-second limit, transcript merging.

## Not yet run against real weights

- `--backend local` (transformers): wiring is tested with stand-in modules only.
- Official ASR models: the chunking and merging are tested; the models themselves have not been loaded.
- `atlasforge finetune` (QLoRA via PEFT/TRL): data checks and `--dry-run` are tested; the training run itself needs an NVIDIA GPU and has never been executed.
- `--backend local --adapter DIR`: evaluating a fine-tuned adapter; wiring tested with stand-ins only.

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
