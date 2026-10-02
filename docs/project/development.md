# Development

For people changing AtlasForge itself.

## Set up

```bash
git clone https://github.com/im-aderm/atlasforge && cd atlasforge
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Add `".[asr]"` or `".[local]"` if you work on the model backends, and `".[docs]"` to build this site.

## The checks

```bash
pytest                       # the whole suite
pytest --cov                 # with coverage
ruff check .                 # lint
ruff format --check .        # formatting
mypy                         # strict type checking
```

All of these must pass before merging, and CI runs them on Linux, macOS and Windows for Python 3.10 and 3.13.

## Project layout

```text
src/atlasforge/
  cli.py            Typer command line, thin over the library
  types.py          shared dataclasses and the Lang type
  errors.py         the error hierarchy
  doctor.py         environment checks
  security.py       secret masking
  render.py         Rich terminal rendering
  backends/         base protocol, openai, local, factory
  eval/             dataset, normalize, metrics, runner, score, report, validate, format
  compare/          stats, slices, compare, report
  asr/              audio (ffmpeg), chunking
tests/
  conftest.py       skips `live` tests unless ATLASFORGE_LIVE=1
  unit/             one test module per source module, plus end-to-end CLI tests
docs/               this site (MkDocs)
planning/           design documents, decision log, discovery notes (not shipped)
```

## Testing

### Layers

| Layer | Needs | What it covers |
|---|---|---|
| Unit | CPU only | normalisation, metrics, bootstrap and McNemar, dataset parsing, chunking maths, error mapping, CLI parsing |
| Contract | CPU only | the `openai` backend against a real local HTTP server speaking the OpenAI protocol |
| End to end | CPU only | the real CLI run against that server: `run`, `eval`, `compare`, `report`, resume, circuit breaker, concurrency |
| Live | GPU, gated weights, `HF_TOKEN` | marked `@pytest.mark.live`, skipped unless `ATLASFORGE_LIVE=1` |

The default `pytest` run needs **no GPU, credentials or gated weights**. The `local` backend's logic is tested with stand-in `torch` and `transformers` modules; real-weight behaviour is verified by hand and recorded in [Verified model facts](../natlas/model-facts.md).

### What the tests guarantee

- Yoruba precomposed and combining Unicode score identically; tone-insensitive scoring keeps underdots and hooked letters.
- WER and CER match `jiwer` on fixtures; the bootstrap is deterministic for a seed and zero-width when runs are identical.
- A 95-second input produces correct windows and a merge without duplicated words.
- An interrupted `eval` resumes without re-running finished examples; a dead server trips the circuit breaker and stays resumable.
- Concurrent and sequential runs produce identical scores.

!!! tip "Test servers need a real listen backlog"
    The runner deliberately bursts up to `2 x concurrency` connections at once. A test HTTP server with the default `socketserver` backlog of 5 refuses some of them on macOS, which looks like a scoring bug but is not. The shared `FakeModelServer` in `tests/unit/test_cli_e2e.py` sets a larger `request_queue_size` for this reason.

## Extending AtlasForge

### Add a backend

Implement the four `Backend` methods (`generate`, `transcribe`, `info`, `close`); no inheritance is needed (see [Write your own backend](../guides/python-api.md#write-your-own-backend)). To expose it on the command line, add a branch to `build_backend` in `backends/factory.py`, importing the module lazily. Add contract tests that run without credentials.

### Add a metric

1. Write the per-example (and, if meaningful, pooled) function in `eval/metrics.py`, expecting already-normalised input.
2. Register it in `KNOWN_METRICS`, `_PER_EXAMPLE` and `_CORPUS` in `eval/score.py`, and in `LOWER_IS_BETTER` if lower is better.
3. Add tests with fixed fixtures. Document it in [Metrics](../reference/metrics.md).

### Change normalisation

Normalisation changes **every score**. Add a test for the exact characters involved, keep the function idempotent, and document the rule in [Normalisation](../reference/normalization.md). Prefer native-speaker review for any language-specific rule.

### Add a task

Tasks live in `eval/dataset.py` (`TASKS`, parsing rules) and `eval/score.py` (`TASK_DEFAULTS`). Update `dataset validate`, the CLI help and the docs together.

## Building the documentation

```bash
pip install -e ".[docs]"
mkdocs serve              # live preview at http://127.0.0.1:8000
mkdocs build --strict     # build into site/, failing on any warning
```

`--strict` turns broken links and missing anchors into errors; CI builds with it. The API reference is generated from docstrings by `mkdocstrings`, so keep public docstrings accurate. Examples in the CLI reference come from the real `--help` output; regenerate them when options change.

### Publishing

A GitHub Actions workflow (`.github/workflows/docs.yml`) builds the site on every pull request and deploys it to GitHub Pages from `main`. The built `site/` directory is plain static files, so any static host also works.

!!! note "Private repositories"
    GitHub Pages for a private repository requires a paid GitHub plan. Until the repository is public, serve the built `site/` directory from any static host, or share it as an archive.

## Releases (planned)

Before tagging `v0.1.0`: all CI jobs green on all platforms; a fresh-venv `pip install` followed by `atlasforge --help` works without `torch`; a live smoke run on a GPU box with its evidence committed; docs build with `--strict` and every quickstart command copy-paste runs; secret scan and `pip-audit` clean; placeholders in `SECURITY.md` and `pyproject.toml` replaced.
