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
- Resumable evaluation runner: manifest guard against mixing runs, per-example error isolation, crash-safe results file, optional threaded concurrency.
