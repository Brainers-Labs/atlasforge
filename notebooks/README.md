# Notebooks

**[`atlasforge-quickstart.ipynb`](atlasforge-quickstart.ipynb)** — the whole workflow on synthetic
demo data: score a run, compare two, read the HTML report. No model, no token, no GPU, so it runs on
a free Colab runtime.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brainers-Labs/atlasforge/blob/main/notebooks/atlasforge-quickstart.ipynb)

## Keeping a notebook honest

Nothing imports a notebook, so nothing notices when it goes stale. Two conventions make these
checkable, and `tests/unit/test_notebook.py` enforces both:

- **A cell whose first line is `# notebook: colab-only` is skipped by the tests.** It is there for
  the reader, not for the suite — installing packages, or calling a model that needs a GPU.
  Every other cell must run on a plain CPython with the package installed, because that is what the
  notebook tells a reader it will do.
- **IPython magics (`%pip install`) are not Python.** The test strips lines beginning with `%`
  before compiling, so a magic in a skipped cell is fine; stray Python that no longer parses is not.

The tests then run the portable cells in order in a temporary directory and assert the files the
notebook claims to write are really there.

To run the notebook yourself:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
jupyter lab notebooks/atlasforge-quickstart.ipynb
```
