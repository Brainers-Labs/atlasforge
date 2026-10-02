# Quickstart: no model needed

**Time: about two minutes.** You will generate demo data, check it, score a run, and compare two models, all offline. No GPU, no Hugging Face token, no API.

!!! info "The demo data is synthetic"
    The answers in this tutorial come from fixed rules, **not from N-ATLaS or any model**. They are built to show every AtlasForge feature in a small space. The runs are labelled `synthetic-demo/...` everywhere so they cannot be mistaken for real results.

## 1. Install

You need Python 3.10 to 3.13. From a clone of the repository:

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv && . .venv/bin/activate
    pip install -e .
    atlasforge --version
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    atlasforge --version
    ```

The base install is small: it does not pull in PyTorch or any other machine-learning framework. See [Installation](installation.md) for the optional extras.

## 2. Create the demo data

```bash
atlasforge demo
```

{{ transcript("demo") }}

This writes a 92-question dataset and two finished runs, a **base** model and a **tuned** model:

```text
atlasforge-demo/
├── toy_qa.jsonl        the dataset
└── runs/
    ├── base/           results of the "base" model
    └── tuned/          results of the "tuned" model
```

## 3. Check the data

```bash
atlasforge dataset validate atlasforge-demo/toy_qa.jsonl
```

{{ transcript("validate") }}

`OK` means no errors. The validator also looks for duplicates, conflicting answers, train/test leakage, broken Unicode and stripped diacritics. See [Validate your data](../guides/validate-your-data.md).

## 4. Score one run

```bash
atlasforge report atlasforge-demo/runs/base --dataset atlasforge-demo/toy_qa.jsonl
```

{{ transcript("report") }}

Three things to notice:

- Every metric appears **twice**: *aware* keeps Yoruba tone marks, *insens.* ignores them. Why that matters is explained in [Tone-aware scoring](../concepts/tone-aware-scoring.md).
- The last line says some examples **failed**. Failed calls count as wrong; they are never quietly dropped.
- `report` needed no model. It re-scores results that already exist.

## 5. Compare the two models

```bash
atlasforge compare atlasforge-demo/toy_qa.jsonl \
  --base atlasforge-demo/runs/base \
  --candidate atlasforge-demo/runs/tuned \
  --slice domain
```

{{ transcript("compare") }}

## 6. Read the result

This is the heart of AtlasForge, so it is worth reading slowly.

**The table.** *Delta* is the candidate minus the base. The *95% CI* is the range the true difference plausibly lies in. The *Verdict* is only `improved` or `regressed` when that whole range is on one side of zero.

**The two views disagree.** Under *aware* scoring the tuned model is clearly better. Under *insens.* it is not. The tuned model keeps Yoruba tone marks that the base model drops, so the gain is partly a tone-mark gain. A tool that showed only one view would have hidden that.

**The regressions.** Overall the tuned model improved, but it got **20 points worse at arithmetic** (the `numeracy` domain and the `has_number` slice). A single headline number would have hidden this.

**Too few to judge.** The 12 Yoruba examples are under the 30-example minimum, so they are reported as *insufficient data* instead of as a result.

Open the full Markdown report that was just written to `comparison/comparison.md`. This is exactly what you would paste into a submission or a README:

??? example "The generated comparison report"
    {{ sample_report("comparison") }}

## What you just learned

| You saw | Why it matters |
|---|---|
| Two tone views | Yoruba and Igbo scores are meaningless if tone marks are silently stripped |
| Failures counted as wrong | A model that crashes on hard questions cannot look good |
| A hidden regression | Averages hide the places a fine-tune hurts |
| "Insufficient data" | Tiny slices produce noise, not findings |

## Next steps

<div class="grid cards" markdown>

- **Use your own model**

    [Choose your setup](choose-your-setup.md), then [Evaluate a model](../guides/evaluate-a-model.md).

- **Use your own data**

    Format your data as JSONL: see [Datasets](../concepts/datasets.md).

- **Understand the statistics**

    [Statistics](../concepts/statistics.md) explains the intervals and the verdict rule.

- **Fine-tune**

    [Fine-tune with QLoRA](../guides/fine-tune-with-qlora.md).

</div>
