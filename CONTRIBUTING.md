# Contributing

Please read the
[Code of Conduct](https://github.com/Brainers-Labs/atlasforge/blob/main/CODE_OF_CONDUCT.md);
by taking part you agree to uphold it.

The rest of this page is also part of the documentation site, so every command below is
tested and every link is checked when the site builds.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
ruff check . && ruff format --check . && mypy && pytest
```

Run those with nothing overridden, and on Windows without `PYTHONIOENCODING` set. That variable
changes what a child process's output decodes as, so it can make the suite pass locally while the
Windows runner — which does not set it — fails on an encoding mismatch that has nothing to do with
the code under test.

## Ground rules

- Only official `NCAIR1/*` models in core paths. Never another foundation model.
- Never invent model behaviour. If a fact isn't verified, verify it and record it (see the project's discovery notes) or don't expose it.
- Core install must stay light: no torch or transformers imports at package import time. Heavy dependencies go in extras and are imported lazily.
- Every public function has type hints and a test. Default tests run on CPU with no credentials.
- No secrets, prompts or audio in logs.

## Good first issues

Normalisation rules, benchmark packs (with provenance and licence), and documentation, especially from native speakers of Hausa, Yoruba and Igbo.
