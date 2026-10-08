## What this changes

<!-- One or two sentences. Link the issue it closes, if any. -->

## Checklist

- [ ] `ruff check . && ruff format --check . && mypy && pytest` passes locally.
- [ ] Every new public function has type hints and a test.
- [ ] No new heavy dependency in the core install (torch/transformers stay in extras, imported lazily).
- [ ] If behaviour or a command changed, the docs under `docs/` are updated (the docs build is strict).
- [ ] Nothing claims a result that was not actually measured. Unverified behaviour is labelled as such.
- [ ] No tokens, prompts, audio or personal data in code, tests, logs or fixtures.

## Verification

<!-- How did you check it? Paste the command and its output. If it needs a GPU or real
     N-ATLaS weights and you could not run it, say so plainly here and in the PR body. -->
