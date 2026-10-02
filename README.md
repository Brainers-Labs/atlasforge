# AtlasForge

Run, evaluate, compare and fine-tune the official **N-ATLaS** models (Hausa, Yoruba, Igbo, Nigerian-accented English).

> **Status: pre-alpha (`0.1.0.dev0`).** Built for NAIC 2026, Problem 01 (Developer Infrastructure). Everything under "Works today" is tested (559 tests). The four ASR models and an int4-quantised copy of the LLM (via Ollama) have been run for real on a Mac M1; the LLM at fp16, vLLM and fine-tuning have **not** (they need a GPU). Nothing is claimed to work until it is listed here.

**Full documentation:** build and browse it with `pip install -e ".[docs]" && mkdocs serve` (sources in [`docs/`](docs/index.md)).

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
- **ASR helpers**: any audio format via ffmpeg (including WhatsApp `.ogg`/opus), automatic splitting of audio over the models' 30-second limit, transcript merging.

## Verified against real weights, and what is not

- **Run for real (Mac M1, 16 GB):** all four ASR models through `--backend local` / `transformers`; the LLM as an int4 Ollama import through `atlasforge run`. Evidence is in `planning/21_NATLAS_DISCOVERY.md`.
- **Not yet run:** the `local` backend for the **LLM** (fp16 needs a GPU of 24 GB or more), vLLM serving, formal WER or any benchmark, and fine-tuning recipes and `card` (not written).

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
