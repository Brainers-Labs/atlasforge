# Developer Experience

## README first screen

```bash
pip install "atlasforge[local,asr]"
export HF_TOKEN=...            # after accepting the NCAIR1 licences on Hugging Face
atlasforge doctor
atlasforge run "Bawo ni o se wa?"
atlasforge eval benchmarks/flores_yo_en_100.jsonl
```

Followed by one screenshot of a real report (tone-aware and tone-insensitive columns) and a 3-line explanation of what makes AtlasForge different.

## `atlasforge doctor` is the onboarding

The largest early friction is environmental, not code. `doctor` checks and explains:
- Python version and installed extras
- GPU presence, VRAM, and whether fp16 or only 4-bit will fit (8B needs ~16 GB fp16, ~6 GB 4-bit, INFERRED, to verify)
- free disk for weights
- ffmpeg presence (ASR)
- `HF_TOKEN` present and valid
- **gated-access status for each of the 5 NCAIR1 repos**, with a direct link to accept the licence if it hasn't been accepted
- endpoint reachability if `ATLASFORGE_BASE_URL` is set

## Documentation structure (mkdocs, published on GitHub Pages)

```text
Home (what / why / 60-second demo)
Quickstart
  ├─ Path A: use an endpoint (no GPU)
  └─ Path B: local weights
Licences & access (accepting the gated licences, the 1,000-user cap, attribution)
Guides
  ├─ Evaluate your task
  ├─ Compare base vs fine-tuned (Problem 03 guide)
  ├─ Transcribe & score ASR (Problem 02 guide)
  ├─ Fine-tune N-ATLaS with QLoRA (P1)
  └─ Serve N-ATLaS with vLLM
Reference: CLI, Python API, dataset format, metrics & normalization
Troubleshooting (generated from real beta errors)
Contributing
```

## Bilingual documentation

NAIC lists bilingual docs as a Problem 01 example. v0.1: translate the **Quickstart** into Hausa (and Yoruba if a reviewer is available), with a named human reviewer. Machine translation without review is not acceptable. Any N-ATLaS-assisted draft is labelled as such, which is itself a nice integration story.

## Language examples

Verified, working prompts and datasets for Hausa, Yoruba, Igbo and Nigerian English. Each example's output is captured from a real run and dated.

## Error message standard

```text
✗ Model access denied: NCAIR1/Yoruba-ASR is gated.
  → Accept the licence at https://huggingface.co/NCAIR1/Yoruba-ASR then re-run `atlasforge doctor`.
```
What happened, why, and the exact next step.
