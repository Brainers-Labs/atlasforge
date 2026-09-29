# Contributing

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
ruff check . && ruff format --check . && mypy && pytest
```

## Ground rules

- Only official `NCAIR1/*` models in core paths. Never another foundation model.
- Never invent model behaviour. If a fact isn't verified, verify it and record it (see the project's discovery notes) or don't expose it.
- Core install must stay light: no torch or transformers imports at package import time. Heavy dependencies go in extras and are imported lazily.
- Every public function has type hints and a test. Default tests run on CPU with no credentials.
- No secrets, prompts or audio in logs.

## Good first issues

Normalisation rules, benchmark packs (with provenance and licence), and documentation, especially from native speakers of Hausa, Yoruba and Igbo.
