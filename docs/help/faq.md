# FAQ

## The basics

### Does N-ATLaS have an API?
Not a public one. It is distributed as gated open weights on Hugging Face, and NAIC issues API credentials only to shortlisted teams. AtlasForge works with the weights (loaded locally or served by an OpenAI-compatible server). See [Choose your setup](../get-started/choose-your-setup.md).

### Do I need a GPU?
Not to evaluate, compare, validate data, or try the demo: those work with any model server and need no GPU on your side. You need a GPU (or a server with one) to run the 8B model at full quality, and an NVIDIA GPU to fine-tune. The small speech models run on a CPU.

### Is AtlasForge a wrapper around ChatGPT, Claude or Gemini?
No. It never sends your data to any company's model. It only talks to the endpoint **you** configure, and the model behind it should be an official `NCAIR1` model. There is no AI-as-judge.

### How is this different from the community N-ATLAS SDK and playground?
Those focus on calling the models. AtlasForge focuses on **measuring them**: evaluation, comparison, dataset checks, and fine-tuning. They work together, because AtlasForge can use any OpenAI-compatible gateway, including theirs.

### Does it send any data anywhere?
No telemetry, no analytics. The only network traffic is the requests to the endpoint you configure, and downloads you trigger from Hugging Face.

## Using it

### Can I try it without a model?
Yes: [the quickstart](../get-started/quickstart.md) uses `atlasforge demo` and needs no model, token or GPU.

### How many examples do I need?
For a meaningful comparison, a few hundred is a good start. Slices need at least 30 each to be judged. Fine-tuning needs at least 20 (the enforced floor), but more is much better.

### Why are failed calls counted as wrong?
Because dropping them flatters the model. A model that crashes on hard questions would look better than one that attempts them. See [Statistics](../concepts/statistics.md#failures-count-as-wrong).

### Why two scores for everything?
Tone marks matter in Yoruba, Igbo and Hausa, and models often get them wrong or leave them out. Showing both views separates "wrong word" from "missing tone marks". See [Tone-aware scoring](../concepts/tone-aware-scoring.md).

### My improvement says "no clear change". Did it fail?
Not necessarily. It means the data cannot distinguish the models with confidence, perhaps because the dataset is small or the difference is. More examples narrow the interval.

### A slice says "insufficient data". Can I force a verdict?
You can lower `--min-slice-n`, but a verdict from a handful of examples is mostly noise. Better to collect more examples for that slice.

### Can I add my own metric?
Yes. Pass a `(prediction, reference, example) -> float` callable to `atlasforge.evaluate(..., metrics=[...])`, or name a function on the command line with `--metric module:function`. A custom metric gets per-example values, so it flows into `compare` and the slices; it gets no pooled figure. See [Your own metrics](../reference/metrics.md#your-own-metrics).

### Can I use other models with it?
You can point it at any OpenAI-compatible server, so technically yes, for example to compare against another model as a baseline. The tool and its docs are built around the official N-ATLaS models.

### Does it support streaming, or a chat interface?
No. It is an evaluation and fine-tuning toolkit, not a chat application.

## Licences

### Can I publish my fine-tuned model?
You can publish a **LoRA adapter**, subject to the base licence: attribution, the "Powered by Awarri" suffix, the 1,000 active-user cap, and a separate agreement for commercial use. `atlasforge card` writes these into the card. Read the licence yourself first. See [Publish a model card](../guides/publish-a-model-card.md).

### Can I publish a quantised or merged copy of N-ATLaS?
AtlasForge does not do that and does not help with it. Check the licence before redistributing any weights.

### What licence is AtlasForge under?
Apache-2.0 for the code. The N-ATLaS models have their own licence.

## Reliability

### What happens if my server crashes mid-run?
After 20 failures in a row the run stops, keeps everything finished, and tells you. Fix the server and run the same command to resume. See [Run directories](../concepts/run-directories.md).

### Are results reproducible?
The statistics are deterministic for a given seed. Model outputs are only reproducible if the model is: use `--temperature 0` and the same server settings. Every run records the dataset fingerprint, the model, the revision and the settings.

### What is verified and what is not?
See [Project status](status.md). In short: everything that does not need real model weights is tested; the local backend, the official speech models and fine-tuning have not yet been run on real weights.
