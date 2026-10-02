# Licence and compliance

!!! danger "This page is a summary, not legal advice"
    The terms below come from the model cards and a third-party summary. **Read the full licence text on each model page before you build on, deploy or redistribute anything.** Where this page says "to confirm", the point has not been verified against the licence itself.

## Two different licences

| Thing | Licence |
|---|---|
| **AtlasForge** (this tool's code) | Apache-2.0 |
| **The N-ATLaS models** | Awarri's custom "Open-Source Research and Innovation License" |

AtlasForge's Apache-2.0 licence does not extend to the models, and the models' licence does not extend to AtlasForge.

## What the model licence says (as summarised on the cards)

| Term | Summary | Status |
|---|---|---|
| **User cap** | limited to organisations with at most 1,000 active end users (per 30 days, per a third-party summary) | to confirm |
| **Commercial use** | needs a separate agreement with Awarri and the relevant ministry | VERIFIED as summarised |
| **Attribution** | credit Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy | VERIFIED as summarised |
| **Derivatives** | derivative works must carry the suffix **"Powered by Awarri"** | VERIFIED as summarised |
| **Llama 3 terms** | whether the Llama 3 Community License also applies (the base model is Llama-3 8B) | UNKNOWN |

## What AtlasForge does to keep you compliant

- **It never redistributes weights.** No model files are bundled, mirrored or uploaded. You download from Hugging Face after accepting the terms yourself, and `doctor` and the docs point you to that step.
- **It does not publish quantised or merged weights.** The Ollama recipe in [Serve the models](../guides/serve-models.md) produces a local copy for your own testing. Do not upload or share it.
- **It names only official models** in its core paths.

A planned `atlasforge card` command will generate a model card for a fine-tuned adapter with the required attribution and naming guidance already filled in. It is not written yet.

## What you must do

1. Accept the licence **yourself** on each model page. Do not share your token or your downloaded weights.
2. If you build a **derivative** (a fine-tune, an adapter that is merged or distributed, a quantised copy that leaves your machine), name it with the "Powered by Awarri" suffix and include the attribution.
3. Check the **1,000 active-user** limit against your deployment. Beyond it, you need an agreement with Awarri.
4. For **commercial** use, obtain the separate agreement first.
5. Keep a copy of the licence text that applied when you downloaded the model.

## Evaluation data licences

AtlasForge evaluates **your** data. Make sure you have the right to use any text or audio you put in a dataset, and respect the licence of public benchmarks you use (for example, FLEURS is CC BY 4.0 according to its dataset card; other benchmarks have their own terms, so check each one). Record provenance and licence next to every benchmark file you share.

## Voice data

Recordings of people's voices are personal data. In Nigeria the Data Protection Act 2023 applies to data you collect. Get explicit consent, keep transcripts rather than raw audio where you can, delete raw audio after use, and never commit raw recordings. See [Security and privacy](../operations/security-privacy.md).
