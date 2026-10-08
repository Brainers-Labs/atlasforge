# Product Requirements — AtlasForge v0.1

## Goal

Ship a small, credible, open-source toolkit that lets a developer **evaluate and improve N-ATLaS on their own task, and produce evidence they can trust**.

## Primary user journeys

**J1 — "Is N-ATLaS good enough for my task?"** (any developer)
```text
pip install atlasforge -> atlasforge doctor -> atlasforge run "..." -> atlasforge eval my_task.jsonl -> report.md
```
Target: **first eval report in under 30 minutes** using an existing endpoint, or under 60 minutes with local weights (the download dominates).

**J2 — "Prove my fine-tune beats base."** (Problem 03 teams)
```text
atlasforge finetune llm --data train.jsonl -> atlasforge compare --base N-ATLaS --candidate ./adapter --data test.jsonl -> comparison report with confidence intervals + model card
```

**J3 — "Transcribe and score ASR."** (Problem 02 teams)
```text
atlasforge transcribe note.ogg --lang ha -> atlasforge eval asr_test.jsonl --task asr -> WER/CER (tone-aware and tone-insensitive)
```

## P0 requirements

### Backends
- `local`: `transformers` with fp16/bf16, optional 4-bit (`bitsandbytes`), uses the model's chat template.
- `openai`: any OpenAI-compatible `/v1/chat/completions` and `/v1/audio/transcriptions` endpoint (vLLM, HF Inference Endpoints, community gateways).
- One `Backend` protocol so the official NAIC API (shortlisted teams) can be added later without changing callers.
- Default generation settings come from the model card (temperature 0.1, repetition_penalty 1.12) and can be overridden.

### ASR
- Models: `NCAIR1/Hausa-ASR`, `NCAIR1/Yoruba-ASR`, `NCAIR1/Igbo-ASR`, `NCAIR1/NigerianAccentedEnglish`, routed by `--lang ha|yo|ig|en`.
- Convert any ffmpeg-readable input (`.ogg/.opus/.mp3/.m4a/.wav`) to 16 kHz mono.
- Chunk audio longer than 30 s (fixed windows with overlap by default; silence-aware chunking added in v0.1 as an opt-in `--silence-aware`, off by default until a run measures which splitter is better).
- Output: `text`, `chunks[]` with *computed* chunk offsets. No confidence scores and no word timestamps unless verified.
- ASR evaluation reports WER/CER plus **alignment-based error analysis**: top substitutions, deletions and insertions, and WER by audio-length bucket. Named-entity, code-switching and dialect analysis are out of scope for v0.1.

### Evaluation engine
- Dataset: JSONL, one example per line: `{"id", "input" | "audio", "reference", "lang", "meta"}`.
- Task types: `generation`, `classification` (label extraction), `asr`.
- Metrics: exact match, accuracy, chrF / chrF++ (sacrebleu), WER / CER (jiwer), latency p50/p95, error counts.
- **Nigerian-language normalization**: Unicode NFC, punctuation/case rules, and every text metric reported twice: *tone-aware* (diacritics kept) and *tone-insensitive* (diacritics stripped). This is our core differentiator. Yoruba especially needs it.
- Custom metric plug-in: a Python callable `(prediction, reference, example) -> float`.
- Deterministic runs: seed, config, model revision (HF commit SHA) and package version recorded in every report.
- Outputs: `results.jsonl` (per example), `report.json`, `report.md`, rich terminal table.
- Resumable: re-running skips completed examples.

### Dataset validation (`atlasforge dataset validate`)
- Schema and required fields per task type, with line-numbered errors.
- Exact and normalized duplicates, **train/test leakage** (exact + normalized overlap between two files), label imbalance.
- Encoding and Unicode health: non-NFC text, mixed combining/precomposed forms, replacement characters, control characters.
- Diacritic statistics per declared language (tone-mark rate, underdot rate, Hausa hooked-letter rate) to catch stripped-diacritic data.
- Reports the *declared* `lang` field. We do **not** claim language identification (no reliable off-the-shelf LID for ha/yo/ig).

### Compare
- Runs two models on the same dataset (or reads two existing `results.jsonl` files, so reports need no GPU).
- Candidate may be a LoRA adapter loaded on the same base weights; models run sequentially and results are cached.
- Per-metric delta with a **paired bootstrap 95% CI**. Binary outcomes also get an exact McNemar test. Win/tie/loss counts.
- **Slice analysis**: results split by any `meta` field (language, domain, length bucket, contains-number, ...). Every slice gets its own CI. A slice with fewer than 30 examples is reported as "insufficient data", never as an improvement or regression. Regressions are listed prominently.
- **Failure-mode flags** (deterministic, per example, clearly not a "hallucination rate"): format non-compliance, missing required terms/entities from the reference, number mismatch, empty/truncated/repeated output, output language different from expected (script-statistics heuristic, labelled as such).
- Writes `report.md` and `report.json`, fit to paste into a NAIC Problem 03 submission. A self-contained `report.html` is P1.

### CLI
```bash
atlasforge doctor                      # python, GPU/VRAM, disk, ffmpeg, HF token, gated-access status per model
atlasforge run "Kí ni orúkọ rẹ?"       # one-off prompt
atlasforge transcribe FILE --lang ha
atlasforge eval DATASET [--task] [--backend] [--out]
atlasforge compare --base ... --candidate ... --data DATASET
atlasforge report RESULTS_DIR          # re-render report from results
atlasforge dataset validate FILE [--against TEST_FILE]
```

Task presets choose default metrics so users don't pick from a long list: `qa` (exact match, chrF, failure flags), `classification` (accuracy, macro-F1), `translation` (chrF++), `asr` (WER, CER, error analysis).
Every command supports `--backend local|openai`, `--json` output, and exits non-zero on failure with an actionable message.

## P1 requirements

- `atlasforge finetune llm`: QLoRA via `peft` + `trl`, one YAML config, targeting a single 24 GB GPU or Colab. Outputs a LoRA adapter, never full weights.
- `atlasforge finetune asr`: Whisper-Small fine-tune recipe for one ASR model.
- `atlasforge card`: generates a model card for an adapter, including required attribution, "Powered by Awarri" suffix guidance, the license user cap, and the linked eval/compare report.
- `atlasforge bench afrobench`: documented wrapper around `lm-evaluation-harness` to reproduce published N-ATLaS AfroBench-LITE numbers. We do not reimplement the tasks.
- Two Colab notebooks: J1 and J2.

## Explicitly out of scope for v0.1

Playground UI, JS/TS SDK, hosted gateway, streaming UI, multi-GPU training, full-weight fine-tuning, weight redistribution, quantized-weight publishing, LLM-as-judge or any "hallucination rate" metric (would need a judge model; a non-N-ATLaS judge in the core path risks NAIC disqualification), language identification, named-entity/dialect ASR analysis.

## Non-functional

- Python 3.10–3.13. Core install is light (`httpx`, `pydantic`, `typer`, `rich`, `jiwer`, `sacrebleu`). Heavy dependencies go in extras: `[local]`, `[asr]`, `[finetune]`.
- `pip install atlasforge` must not pull in torch.
- The default test suite runs on CPU with no credentials and no gated weights.
- No secrets or prompts in logs by default.
- Type hints throughout. `ruff` + `mypy` (or `pyright`) clean.
- Apache-2.0 for our code.
