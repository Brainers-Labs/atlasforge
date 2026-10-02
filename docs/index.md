# AtlasForge

**Run, evaluate, compare and fine-tune the official N-ATLaS models, and know whether you actually made them better.**

N-ATLaS is Nigeria's open language and speech model for Hausa, Yoruba, Igbo and Nigerian-accented English. It is published as downloadable weights. AtlasForge is the tooling around them, built to answer one question:

> *I changed this model. Did it get better on my task, and where did it get worse?*

<div class="grid cards" markdown>

- :material-rocket-launch:{ .lg .middle } **Try it in two minutes**

    ---

    No model, no GPU, no token, no API. Generate demo data and run a real comparison offline.

    [:octicons-arrow-right-24: Quickstart](get-started/quickstart.md)

- :material-server-network:{ .lg .middle } **Connect a model**

    ---

    Point AtlasForge at any OpenAI-compatible server, or load the weights directly.

    [:octicons-arrow-right-24: Choose your setup](get-started/choose-your-setup.md)

- :material-chart-line:{ .lg .middle } **Prove an improvement**

    ---

    Paired statistics, per-slice results and honest "insufficient data" instead of over-claiming.

    [:octicons-arrow-right-24: Compare two models](guides/compare-two-models.md)

- :material-tune:{ .lg .middle } **Fine-tune and publish**

    ---

    A QLoRA recipe with safety checks, and a model card that carries the licence obligations.

    [:octicons-arrow-right-24: Fine-tune with QLoRA](guides/fine-tune-with-qlora.md)

</div>

## What you get

| | |
|---|---|
| **Evaluate** | Run any dataset through a model, resume after a crash, score it with chrF, WER, accuracy and more. [Guide](guides/evaluate-a-model.md) |
| **Compare** | Base vs fine-tuned with confidence intervals, an exact significance test, and a list of regressions. [Guide](guides/compare-two-models.md) |
| **Respect the languages** | Every text metric is reported with tone marks kept *and* ignored, so Yoruba and Igbo scores mean something. [Concept](concepts/tone-aware-scoring.md) |
| **Check your data** | Duplicates, conflicting labels, train/test leakage, broken Unicode and stripped diacritics. [Guide](guides/validate-your-data.md) |
| **Speech** | Transcribe any audio format with the official ASR models, including clips longer than their 30-second limit. [Guide](guides/transcribe-speech.md) |
| **Fine-tune** | QLoRA recipe that refuses leaky data and tiny datasets before it touches a GPU. [Guide](guides/fine-tune-with-qlora.md) |
| **Publish** | Model cards with attribution, "Powered by Awarri", the user cap, evaluation numbers and regressions. [Guide](guides/publish-a-model-card.md) |

## Honest by default

AtlasForge is built to avoid the usual ways evaluation tools flatter their users:

- **Failed calls count as wrong.** They are never dropped to make the score look better.
- **Small slices are not judged.** Fewer than 30 examples is reported as *insufficient data*.
- **Both tone views are always shown.** Nothing is silently normalised away.
- **Nothing leaves your machine** except the requests you send to the endpoint you configure. There is no telemetry.

!!! warning "Project status: pre-alpha ({{ version }})"
    The evaluation, comparison, dataset-checking and demo features are fully tested. The parts that need real N-ATLaS weights, namely the **local backend, the official speech models and fine-tuning**, are written and tested against stand-ins but **have not yet been run on real models**. See the [project status](help/status.md) page for exactly what is verified.

## Who it is for

- **Developers and teams** deciding whether N-ATLaS is good enough for their application.
- **NAIC teams** that must show a fine-tuned model beats the base model, or that need reliable speech evaluation.
- **Researchers** comparing Nigerian-language models on their own data.

## What it is not

AtlasForge is not a new model, a chatbot, a hosted service, or a wrapper around another company's model. It never redistributes N-ATLaS weights. See [Access and licences](get-started/access-and-licences.md).
