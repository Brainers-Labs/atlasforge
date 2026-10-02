# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Documentation site (MkDocs Material): getting started, concepts, guides, CLI/dataset/metrics/output references, generated Python API reference, model facts, licence, troubleshooting, security, FAQ. GitHub Pages workflow builds with `--strict`. `docs` extra.
- Verification of the four ASR models and an int4 LLM serve on real weights (see `planning/21`).
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
- Circuit breaker: `eval` aborts after N consecutive failures (default 20) and stays resumable; bounded in-flight window for threaded runs.
- Resumable evaluation runner: manifest guard against mixing runs, per-example error isolation, crash-safe results file, optional threaded concurrency.

### Fixed
- `dataset validate` printed the line number twice for line errors (`line 2: line 2: ...`); `Issue.message` no longer repeats it.
- `asr` and `local` extras pin `scipy<1.15` on macOS, where the default wheel fails to load on recent releases.
- End-to-end concurrency test no longer flakes on macOS (larger listen backlog on the fake server).
