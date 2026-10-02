# Fine-tune settings

Settings for [`atlasforge finetune`](cli.md#atlasforge-finetune), in a `.yaml`, `.yml` or `.json` file. See the [fine-tuning guide](../guides/fine-tune-with-qlora.md) for the full workflow.

- Relative paths are resolved against the config file's own folder.
- **Unknown keys are rejected**, so a typo cannot be silently ignored.
- Every invalid value is reported together, not one at a time.
- YAML needs the `finetune` extra (PyYAML); `.json` always works.

{{ fine_tune_settings() }}

!!! warning "The defaults are conventional, not tuned"
    They are common starting points for a 7-8B model on a single 24 GB GPU. They have **not** been measured on N-ATLaS. Treat them as a baseline and expect to adjust.

## Minimal config

```yaml
train_file: train.jsonl
output_dir: adapters/my-adapter
```

## Validated ranges

| Setting | Allowed |
|---|---|
| `lora_r`, `lora_alpha`, `num_epochs`, `batch_size`, `grad_accum`, `logging_steps`, `min_examples` | 1 or more |
| `lora_dropout`, `warmup_ratio` | at least 0 and below 1 |
| `learning_rate` | greater than 0 |
| `max_seq_len` | 64 to 8192 (the model's stated context window) |
| `quantize` | `4bit` or `none` |
| `target_modules` | a non-empty list of strings |

## Output

In `output_dir`: the LoRA adapter files, the tokenizer files, and `training_run.json` ([format](file-formats.md#training_runjson)). Intermediate checkpoints are not kept.
