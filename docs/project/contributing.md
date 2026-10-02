--8<-- "CONTRIBUTING.md"

## Building these docs

```bash
pip install -e ".[docs]"
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what CI runs: any broken link or missing page fails the build
```

The CLI reference, metrics table, fine-tune settings, file-format examples and every terminal transcript are **generated from the code** while the site builds (see `docs_macros.py`), so do not edit those by hand. If you add a metric or a fine-tune setting without documenting it, the build fails on purpose.

Commands shown in the docs are also checked by a test that every flag they use really exists.
