# Fine-tune with QLoRA

Teach N-ATLaS your domain, then prove it worked.

!!! danger "The training run has never been executed"
    The recipe is written against the PEFT and TRL libraries, and its data checks, planning and wiring are fully tested. **The training itself has not been run on a GPU**, so memory use, speed and results are unknown. Run the `--dry-run` first, expect to adjust settings on first contact, and see [Project status](../help/status.md).

## What this produces

A small **LoRA adapter**: a few hundred megabytes of extra weights that sit on top of N-ATLaS. It is **not** a copy of the model. AtlasForge never saves, copies or uploads the base weights; they are always loaded from Hugging Face under your own accepted licence.

## Requirements

- An **NVIDIA GPU**. QLoRA uses 4-bit quantisation through bitsandbytes, which needs CUDA. A 24 GB card is a comfortable target for the defaults. It will not work on a Mac or a CPU.
- `pip install -e ".[finetune]"` and [access to the model](../get-started/access-and-licences.md).
- Training data: at least 20 examples (more is better), in the [JSONL format](../concepts/datasets.md) with a `reference` on every example.
- A **held-out test set** that is not in the training data.

## 1. Evaluate the base model first

You cannot show an improvement without a baseline:

```bash
atlasforge eval test.jsonl --out runs/base --base-url http://127.0.0.1:8000/v1 --model NCAIR1/N-ATLaS
```

## 2. Write a config

Save this as `finetune.yaml` (or `.json`):

```yaml
train_file: train.jsonl
eval_file: test.jsonl        # used only to refuse leakage; never trained on
output_dir: adapters/hausa-agri

lora_r: 16
lora_alpha: 32
learning_rate: 0.0002
num_epochs: 1
max_seq_len: 2048
```

Relative paths are resolved against the config file's folder. Unknown keys are rejected, so a typo cannot slip through. Every setting and its default is in the [settings reference](../reference/fine-tune-settings.md).

## 3. Dry run (no GPU needed)

```bash
atlasforge finetune finetune.yaml --dry-run
```

This validates the data, **refuses train/test leakage**, and prints the plan. The output below is a real dry run on a train/test split of the [demo data](../get-started/quickstart.md) (yours will show your own file and numbers):

{{ transcript("dry_run") }}

## 4. Train

```bash
atlasforge finetune finetune.yaml
```

When it finishes, `adapters/hausa-agri/` holds the adapter and a `training_run.json` recording what was trained and how.

## 5. Evaluate the adapter

```bash
atlasforge eval test.jsonl --out runs/tuned --backend local --adapter adapters/hausa-agri
```

The adapter is part of the model's identity in the run, so it can never be confused with the base run.

## 6. Prove it

```bash
atlasforge compare test.jsonl --base runs/base --candidate runs/tuned --slice domain
```

Read the [verdicts and the regressions](compare-two-models.md#3-read-it). A fine-tune that improves one slice and damages another is common; this is how you find out.

## 7. Publish responsibly

```bash
atlasforge card --name "Hausa Agri" --description "..." --training-data "..." \
  --comparison comparison/comparison.json \
  --training-run adapters/hausa-agri/training_run.json
```

See [Publish a model card](publish-a-model-card.md).

## Safety checks built in

`finetune` refuses, with an explanation, before touching a GPU, when:

| Problem | Why it is refused |
|---|---|
| Any validation **error** in the training data | Bad lines would silently corrupt training |
| Training data **overlaps the held-out file** | A model trained on its test set proves nothing |
| An example has **no `reference`** | There is no answer to learn |
| **Fewer than 20 examples** (`min_examples`) | Mostly teaches the model to repeat them |
| More than **10% of examples exceed `max_seq_len`** | They would be cut off mid-answer, teaching the model to stop early |
| The tokenizer has **no chat template** | Training text would not match how the model is prompted |
| **No NVIDIA GPU** for 4-bit | It cannot work |

## If it runs out of memory

Lower `max_seq_len` or `lora_r`, keep `batch_size` at 1 and raise `grad_accum` to keep the effective batch the same, or use a GPU with more memory. The error message says the same.

## What is not included

Full-weight fine-tuning, multi-GPU training, and fine-tuning the speech models are out of scope for this release.
