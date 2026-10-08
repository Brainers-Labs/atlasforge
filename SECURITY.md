# Security Policy

## Reporting a vulnerability

Report privately through GitHub's [security advisory form](https://github.com/im-aderm/atlasforge/security/advisories/new) with details and reproduction steps. Please do not open a public issue. We aim to acknowledge within 72 hours.

## Scope and practices

- AtlasForge never logs tokens, prompts or audio by default.
- Tokens are shown only masked (`hf_****abcd`).
- Model weights are never bundled or redistributed.
- CI runs gitleaks and `pip-audit`.
