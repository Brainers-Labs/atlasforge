# AtlasForge

Run, evaluate, compare and fine-tune the official **N-ATLaS** models (Hausa, Yoruba, Igbo, Nigerian-accented English).

> **Status: pre-alpha (`0.1.0.dev0`).** Built for NAIC 2026, Problem 01 (Developer Infrastructure). Only the foundation exists so far: `atlasforge doctor`, the error/type model, the backend contract and Nigerian-language text normalisation. Evaluation, comparison, ASR and fine-tuning are being built now. Nothing here is claimed to work until it is listed under "Works today".

## The question AtlasForge answers

> *I changed this N-ATLaS model. Did I actually make it better on my task, and where did it get worse?*

N-ATLaS is published as gated open weights on Hugging Face, so developers get the models but no tooling to measure them on their own data. AtlasForge is that tooling.

## Works today

```bash
pip install atlasforge          # once released; for now: pip install -e ".[dev]"
atlasforge doctor               # Python, GPU/VRAM, disk, ffmpeg, HF token, optional extras
atlasforge doctor --json
```

Library: `atlasforge.eval.normalize` (tone-aware and tone-insensitive normalisation that keeps Yoruba/Igbo underdots and Hausa hooked letters).

## Planned for v0.1

`run`, `transcribe`, `eval`, `compare` (base vs fine-tuned with confidence intervals and per-slice regressions), `dataset validate`, one QLoRA fine-tuning recipe, licence-aware model cards.

## Models and licence

AtlasForge **never redistributes** N-ATLaS weights. You download them from Hugging Face after accepting the Awarri licence yourself. The licence includes a 1,000 active-end-user cap, attribution requirements and a "Powered by Awarri" suffix for derivatives. Read it before you build on the models. AtlasForge's own code is Apache-2.0.

## Development

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check . && mypy
```

The default test suite needs no GPU, credentials or gated weights.
