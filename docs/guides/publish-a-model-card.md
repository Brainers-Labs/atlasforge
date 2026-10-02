# Publish a model card

If you share a fine-tuned adapter, the N-ATLaS licence attaches obligations to it. `atlasforge card` writes them into a model card for you, together with your evaluation results, so nothing is forgotten.

!!! danger "A card helps; it does not replace reading the licence"
    The terms in the generated card were read from the base model's Hugging Face page and summarised. **The licence text governs.** Read it before you publish, and recheck it for each release.

## Write the card

```bash
atlasforge card \
  --name "Hausa Agri" \
  --description "A LoRA adapter that answers Hausa agriculture questions." \
  --training-data "1,200 questions written by our team and reviewed by two agronomists. Released under CC-BY-4.0." \
  --lang ha \
  --domain agriculture \
  --author "Brainers Labs" \
  --comparison comparison/comparison.json \
  --training-run adapters/hausa-agri/training_run.json \
  --out MODEL_CARD.md
```

Only three options are required: `--name`, `--description` and `--training-data`. For a long training-data description, pass `--training-data @notes.txt` to read it from a file.

## What goes into the card

| Section | Where it comes from |
|---|---|
| **Licence and attribution** | Fixed text: attribution to Awarri Technologies and the Federal Ministry of Communications, the **"Powered by Awarri"** suffix, the **1,000 active-user cap**, and "commercial use needs a separate agreement" |
| **Name** | Yours, with ` - Powered by Awarri` added if missing |
| **Training data** | Exactly what *you* wrote. State where it came from **and its licence**. |
| **Training details** | `training_run.json`, if you pass it: method, example count, data fingerprint, hyperparameters |
| **Evaluation** | `comparison.json`, if you pass it: the metrics table, intervals, verdicts and **regressions** |
| **Limitations** | The base model's stated limits (bias, dialect and accent bias, code-switching), and a note that nothing beyond your evaluation is known |
| **Usage** | A code sample for loading the adapter on top of the base model |
| **Front matter** | Hugging Face metadata: `license: other`, `base_model`, `library_name: peft`, languages |

The card **never invents** anything. If you give it no comparison, it says "No evaluation results were recorded" instead of making something up.

## Warnings it prints

`card` always writes the file, then lists anything worth fixing:

- The name lacks "Powered by Awarri".
- No languages given.
- No evaluation results.
- **The comparison shows regressions.** They are published in the card. That is intentional: a model card that hides a regression is worse than useless.

## Before you publish: a checklist

- [ ] I read the base model's licence on its Hugging Face page.
- [ ] The card says where the training data came from and under what licence.
- [ ] I have the right to share the training data, or I am not sharing it.
- [ ] I uploaded **only the adapter**, never base or merged weights.
- [ ] The evaluation was done on a held-out set that was not trained on.
- [ ] I read the regressions section and am comfortable with it.
- [ ] If my application will have more than 1,000 active users, or is commercial, I have asked about a separate agreement.

AtlasForge does not upload anything. Publishing to Hugging Face is a step you take yourself.
