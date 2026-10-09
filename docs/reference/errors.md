# Errors and exit codes

Every error AtlasForge raises on purpose extends `AtlasForgeError` and carries a **hint**: the concrete next step. The CLI prints both:

```text
error: Cannot access NCAIR1/Yoruba-ASR.
  -> Accept the licence on the model's Hugging Face page, then set HF_TOKEN ...
```

In Python, catch `AtlasForgeError` to handle anything AtlasForge raises deliberately; read `.hint` for the suggestion.

```python
from atlasforge.errors import AtlasForgeError

try:
    ...
except AtlasForgeError as exc:
    print(exc.message, "|", exc.hint)
```

## Exceptions

{{ errors_table() }}

## Privacy of error messages

Messages are written so they cannot leak your data:

- A server's **response body** is never put in a message (servers sometimes echo the prompt). `BackendHTTPError.body_excerpt` holds a short excerpt for debugging, separately.
- For unexpected exceptions only the exception **type** is recorded, never its text.
- Tokens are never printed unmasked.

## Exit codes

| Code | Meaning | Examples |
|---|---|---|
| `0` | Success | Everything ran. Warnings never change this. |
| `1` | The command ran and found a problem | `doctor` has a failed check; `dataset validate` found errors; every `eval` example failed; a `transcribe` file failed |
| `2` | AtlasForge could not do what you asked | Bad config, unreachable server, missing file, refused training data, aborted run |

Exit code 2 always comes with a message and a hint. See [Troubleshooting](../help/troubleshooting.md) for the common ones.
