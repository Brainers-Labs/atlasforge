# Open Source Plan

## Repository

Proposed: `github.com/brainerslabs/atlasforge`. Check availability on D1. Private until `v0.1.0a1` (D7), then public.

## Licensing

| Thing | Licence |
|---|---|
| AtlasForge code | **Apache-2.0** (patent grant, standard for ML tooling, compatible with the core dependencies, all to be verified with a licence scan) |
| Docs | CC-BY-4.0 |
| Benchmark packs | Inherit the source licence (e.g. FLORES-200 CC-BY-SA-4.0). Each pack has its own `PROVENANCE.md`. |
| N-ATLaS weights | **Not distributed.** Awarri Open-Source Research and Innovation License, accepted by each user on Hugging Face. |
| User fine-tuned adapters | Governed by the Awarri licence. `atlasforge card` writes the required attribution and naming. |

## Relationship to other community projects

Other N-ATLAS SDKs and gateways exist (see [22](22_COMPETITIVE_LANDSCAPE.md)). Our stance is to **interoperate and credit**: document how to point AtlasForge's `openai` backend at their gateway, and link to them in the README. The ecosystem is small, and being the evaluation layer for everyone is stronger than being a second SDK.

## Governance

- Now: Brainers Labs maintains. Public issues, PRs, tagged releases, CHANGELOG.
- `good first issue` labels on metrics, normalization rules and benchmark packs. Language experts can contribute there without deep ML knowledge.
- Later: named maintainers outside Brainers Labs, a benchmark-pack contribution guide, and a community leaderboard of N-ATLaS derivatives evaluated with AtlasForge.

## Success metrics (post-NAIC)

PyPI downloads, number of external projects producing AtlasForge reports (especially NAIC Problem 02/03 teams), community benchmark packs contributed, external contributors, issues resolved. Stars are not a metric we optimise for.
