# Contributing

Thank you for helping. AtlasForge is small and its rules are few but firm, because its value is that its numbers can be trusted.

## Ground rules

- **Only official `NCAIR1/*` models in core paths.** Never another foundation model, and no LLM-as-judge.
- **Never invent model behaviour.** If a fact about a model is not verified, verify it and record it (see [Verified model facts](../natlas/model-facts.md)) or do not expose it. Label claims `VERIFIED`, `INFERRED` or `UNKNOWN`.
- **Never fabricate validation or data.** No made-up benchmarks, testers or results.
- **Keep the core install light.** No `torch` or `transformers` imports at package import time. Heavy dependencies go in extras and are imported lazily.
- **Every public function has type hints and a test.** The default test suite runs on a CPU with no credentials and no gated weights.
- **No secrets, prompts or audio in logs, errors or reports.**
- **Never redistribute model weights**, quantised or otherwise.

## Good first contributions

- **Normalisation rules** for Hausa, Yoruba and Igbo, especially from native speakers who can say what is and is not the same word.
- **Benchmark packs**, always with provenance and licence recorded alongside.
- **Documentation**, including translations. A translation must be reviewed by a named human speaker; unreviewed machine translation is not accepted.
- **Bug reports** with the output of `atlasforge doctor --json`, the exact command, and the full error including its `->` hint line.

## Workflow

1. Fork and clone, then set up the environment:

    ```bash
    python3 -m venv .venv && . .venv/bin/activate
    pip install -e ".[dev]"
    pre-commit install
    ```

2. Make your change with tests.
3. Run all the checks:

    ```bash
    ruff check . && ruff format --check . && mypy && pytest
    ```

4. If you changed behaviour, update the docs in `docs/` and add a line to `CHANGELOG.md`.
5. Open a pull request describing what changed and why.

See [Development](development.md) for the project layout and how to extend it.

## Code conventions

- Python 3.10+; `from __future__ import annotations` in library modules (not in `cli.py`, where Typer reads annotations at runtime).
- `mypy --strict` must pass. Prefer small frozen dataclasses with `slots=True, kw_only=True` for data.
- Raise an `AtlasForgeError` subclass with a **hint** for any error a user can act on.
- Comments explain *why*, never *what*. No multi-paragraph docstrings.
- Public docstrings use Google style so they render in the [API reference](../reference/python-api.md).

## Reporting a security issue

Do not open a public issue. See [Security and privacy](../operations/security-privacy.md#reporting-a-vulnerability).
