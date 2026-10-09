--8<-- "CONTRIBUTING.md"

## Building these docs

```bash
pip install -e ".[docs]"
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what CI runs: any broken link or missing page fails the build
```

The CLI reference, metrics table, fine-tune settings, file-format examples and every terminal transcript are **generated from the code** while the site builds (see `docs_macros.py`), so do not edit those by hand. If you add a metric or a fine-tune setting without documenting it, the build fails on purpose.

Commands shown in the docs are also checked by a test that every flag they use really exists.

A new feature has three more places to land, each with a test that will tell you if you forget:

| If you add... | Also update | Checked by |
|---|---|---|
| an optional dependency (an extra in `pyproject.toml`) | the extras table on the installation page | `test_every_extra_is_offered_on_the_installation_page` |
| a command that writes a file | [File formats](../reference/file-formats.md) | `test_every_file_the_tool_writes_is_documented_in_the_formats_page` |
| a public module or entry point | [Python API](../reference/python-api.md) | `test_the_api_reference_names_every_entry_point` |
| a whole new guide | `mkdocs.yml` nav **and** the table on the guides index | `test_every_page_is_in_the_navigation_and_every_nav_entry_exists` |
| a credential or a new environment variable | [Environment variables](../reference/environment.md) | `test_the_credentials_the_tool_reads_are_documented` |

An entry on [Status](../help/status.md) is the fifth: say whether the feature has been exercised
against a real model, and if it has not, say so plainly.
