# Releasing

For maintainers. A release is a **tag**; pushing it does the rest.

## The short version

1. Make sure everything in [Project status](../help/status.md) that changed is still true.
2. Move the entries under `## [Unreleased]` in `CHANGELOG.md` under a new `## [VERSION] - YYYY-MM-DD` heading.
3. Set `__version__` in `src/atlasforge/__init__.py` to that same version.
4. Commit, then tag and push:

   ```bash
   git tag -a v0.1.0a1 -m "AtlasForge 0.1.0a1"
   git push origin v0.1.0a1
   ```

The [Release workflow](https://github.com/Brainers-Labs/atlasforge/blob/main/.github/workflows/release.yml)
then checks that the tag, the package version and the changelog agree, builds the sdist and the
wheel, installs the wheel into a fresh virtual environment and runs it, writes `SHA256SUMS.txt`,
and attaches all of it to a GitHub release.

If the three disagree, the run fails before anything is published and says which one to fix.

## Installing it

The distribution is **`brainers-atlasforge`**. The import name and the command are both
`atlasforge`:

```bash
pip install --pre brainers-atlasforge
atlasforge --version
```

The distribution name is org-scoped for one reason: the plain name is taken. PyPI's
[`atlasforge`](https://pypi.org/project/atlasforge/) is an unrelated bioinformatics project
(gene-family atlases) first published in August 2026, so `pip install atlasforge` would install
someone else's package. Splitting the distribution name from the import name is ordinary —
Pillow and PIL do the same thing.

Pre-release versions (`0.1.0a1`) need `--pre`; without it pip will not see them.

## Publishing to PyPI

Separate, and manual, on purpose: **no tag push uploads to the index.** Run the
[Publish workflow](https://github.com/Brainers-Labs/atlasforge/blob/main/.github/workflows/publish.yml)
from the Actions tab with the tag as its input.

It uses PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is
stored in the repository or in CI. That needs a one-time setting on PyPI before the first run:

| Field | Value |
|---|---|
| PyPI project | `brainers-atlasforge` (as a *pending* publisher) |
| Owner | `Brainers-Labs` |
| Repository | `atlasforge` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Until that exists the workflow fails at the last step with PyPI's own message. That is on
purpose: publishing cannot be undone, and a silently-skipped publish is worse than a loud one.

## What a release must not say

The [status page](../help/status.md) lists what has and has not been run against the real
N-ATLaS models. A release does not change any of that. If the release notes are generated from
commits, check them before the release goes wider: nothing here may claim a measurement that
was never taken.
