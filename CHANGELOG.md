# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: src layout, ruff, mypy (strict), pytest, CI matrix, pre-commit with gitleaks.
- `atlasforge doctor` (and `--json`).
- Error hierarchy with actionable hints, shared types, `mask_secret`.
- `Backend` protocol.
- Tone-aware and tone-insensitive text normalisation for ha/yo/ig/en.
- Strict JSONL dataset loader (line-numbered errors, content fingerprint, unique ids).
- Metrics (exact match, accuracy, macro-F1, chrF/chrF++, WER, CER, latency percentiles) and `score_run`: every text metric under both tone-aware and tone-insensitive views; failed or missing predictions count as wrong; per-example values kept for paired statistics.
- `compare`: paired bootstrap, exact McNemar, slice analysis (n>=30 rule), Markdown/JSON reports.
- OpenAI-compatible backend (retries, error mapping, no response bodies in messages, HTTPS-or-loopback guard).
- `local` transformers backend (stand-in tested only; not yet run against real weights).
- ASR: ffmpeg decoding to 16 kHz mono, windowing past the 30 s limit, seam-aware transcript merging, `LongAudioBackend`.
- `dataset validate`: duplicates, conflicting labels, leakage, Unicode health, diacritic statistics, class balance.
- CLI: `run`, `transcribe`, `eval`, `report`, `compare`, `dataset validate`; `python -m atlasforge`.
- `finetune`: QLoRA recipe (config, data checks, `--dry-run`, leakage/truncation guards, TRL argument-name detection) and `training_run.json`; not yet run on a GPU.
- `card`: licence-aware model cards from `comparison.json` and `training_run.json`.
- `--adapter` on the local backend (and `run`/`eval`) to evaluate LoRA adapters; the adapter is part of the model identity.
- Circuit breaker: `eval` aborts after N consecutive failures (default 20) and stays resumable; bounded in-flight window for threaded runs.
- Resumable evaluation runner: manifest guard against mixing runs, per-example error isolation, crash-safe results file, optional threaded concurrency.
