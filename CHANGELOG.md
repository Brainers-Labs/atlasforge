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
- Resumable evaluation runner: manifest guard against mixing runs, per-example error isolation, crash-safe results file, optional threaded concurrency.
