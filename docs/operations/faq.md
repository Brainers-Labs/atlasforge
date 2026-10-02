# FAQ

## General

### What is AtlasForge, in one sentence?

A Python library and command-line tool to run, evaluate, compare and (eventually) fine-tune the official N-ATLaS models, with scoring that understands Hausa, Yoruba and Igbo tone marks and statistics that tell you whether a change really helped.

### Is it an SDK, a playground or a gateway for N-ATLaS?

No. Those exist elsewhere, and AtlasForge works with them through the OpenAI-compatible backend. AtlasForge's job is the part that was missing: measuring models on your own data.

### Does it work with models other than N-ATLaS?

The core paths are intentionally limited to the official `NCAIR1/*` models, and there is no LLM-as-judge. Technically the `openai` backend will talk to any OpenAI-compatible server, but AtlasForge's defaults, documentation and claims are about N-ATLaS, and it makes no promises elsewhere.

### Is it free? What licence?

AtlasForge is Apache-2.0. The N-ATLaS models have their own, separate licence; see [Licence and compliance](../natlas/licence.md).

### What is its status?

Pre-alpha, `0.1.0.dev0`. See [Status and roadmap](../project/status.md) for exactly what has and has not been verified against real models.

## Using it

### Do I need a GPU?

Not to install, validate datasets, re-score runs or use the speech models (they run on a laptop). For the **LLM** you need either a GPU box, or a quantised model served by Ollama or llama.cpp, or someone else's endpoint. See [Serve the models](../guides/serve-models.md).

### Can I run the LLM on my laptop?

On a 16 GB machine, the fp16 weights (16.08 GB) do not fit. An int4 copy served through Ollama does (6.6 GB) and was verified on an Apple M1 16 GB. It is a degraded copy, so treat scores from it accordingly. See [Run on a Mac](../guides/mac.md).

### How do I get access to the models?

Accept the licence on each of the five Hugging Face repositories and create a token. Step by step in [Models and access](../getting-started/models-and-access.md).

### How big should my evaluation dataset be?

There is no magic number. Hundreds of examples from your real task is a sensible start. The honest answer comes from `compare`: if the confidence intervals are wide and verdicts say "no clear change", you need more data. For per-slice verdicts, each slice needs at least 30 examples.

### Why are there two scores for every metric?

Hausa, Yoruba and Igbo text is written inconsistently with respect to tone marks. Reporting both a tone-aware and a tone-insensitive view means you never have to guess which one a number used. See [Tone-aware scoring](../concepts/tone-aware-scoring.md).

### Why does a failed request lower my score?

Because a model behind an unreliable server is worse in practice than one behind a reliable server. Dropping failures would hide that. Fix the failures and resume the run to retry them. See [Evaluate a model](../guides/evaluate.md#failures-and-the-circuit-breaker).

### Can I compare two prompts?

Not directly in v0.1: prompts live inside the dataset, and `compare` pairs runs by dataset hash. You can compare models, quantisations, checkpoints and decoding settings on one dataset. See the note in [Compare two models](../guides/compare.md).

### Can I use my own metric?

Not through the CLI yet. From Python you can score with your own functions using the per-example outputs in `results.jsonl`, and the statistics (`paired_bootstrap`, `mcnemar_exact`) are public and work on any aligned lists of numbers.

### Can I fine-tune with AtlasForge?

Not yet; the recipes are not written. AtlasForge helps before and after a fine-tune (validate data, baseline, compare). See [Fine-tuning](../guides/fine-tuning.md).

### Does it support streaming, tool calling or multi-turn chat?

Multi-turn prompts are supported in datasets via `messages`. Streaming and tool calling are not part of the evaluation flow.

### Which languages are supported?

Hausa (`ha`), Yoruba (`yo`), Igbo (`ig`) and Nigerian-accented English (`en`), matching the models.

### Can I use it in CI?

Yes. The commands have meaningful exit codes, `--json` outputs and deterministic statistics (fixed seed). See [Compare two models](../guides/compare.md#using-the-json-in-scripts-and-ci).

## Trust and verification

### How do I know the numbers are right?

The test suite has 559 tests with 98% line coverage, including end-to-end tests that run the real CLI against a real local HTTP server. The metrics use established libraries (`sacrebleu` for chrF, `jiwer` for WER and CER). The statistics are implemented directly and tested. See [Development](../project/development.md).

### What has been run against the real models?

The four speech models (loaded and transcribing real clips) and the LLM as an int4 Ollama import. fp16 loading, vLLM, formal WER and a benchmark run have **not** been done. The complete list, with evidence, is in [Verified model facts](../natlas/model-facts.md#our-verification-log).

### Is any of this tested on a GPU?

Not yet. No NVIDIA GPU has been available so far.

### Does AtlasForge send my data anywhere?

Only to the model server you point it at (and to Hugging Face if the `local` backend downloads a model). See [Security and privacy](security-privacy.md).

## Contributing

### How can I help?

Normalisation rules, benchmark packs (with provenance and licence), and documentation, especially from native speakers of Hausa, Yoruba and Igbo. See [Contributing](../project/contributing.md).
