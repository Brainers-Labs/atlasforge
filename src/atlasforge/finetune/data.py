"""Training-data preparation: validate, convert to chat conversations, and plan the run.

Everything here is model-free, so it is fully testable and powers ``finetune --dry-run``.
Training examples use the same JSONL format as evaluation (``input`` or ``messages`` plus a
``reference``, the answer to learn), so one file can be validated, trained on, and scored.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from atlasforge.errors import ConfigError
from atlasforge.eval.dataset import parse_file
from atlasforge.eval.validate import validate_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from atlasforge.finetune.config import QLoRAConfig
    from atlasforge.types import Message

MAX_ERRORS_SHOWN: Final = 3
MAX_TRUNCATED_SHARE: Final = 0.10


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingData:
    """Validated conversations ready to be rendered with the model's chat template."""

    conversations: tuple[tuple[Message, ...], ...]
    sha256: str

    def __len__(self) -> int:
        return len(self.conversations)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingPlan:
    n_examples: int
    steps_per_epoch: int
    total_steps: int
    effective_batch_size: int
    median_words: float


def prepare_data(config: QLoRAConfig) -> TrainingData:
    """Validate ``config.train_file`` and turn it into conversations.

    Refuses to continue on any validation error, including train/test leakage against
    ``config.eval_file``: a model trained on its own test set proves nothing.
    """
    report = validate_dataset(config.train_file, "generation", against=config.eval_file)
    if report.errors:
        shown = "; ".join(i.message for i in report.errors[:MAX_ERRORS_SHOWN])
        more = len(report.errors) - MAX_ERRORS_SHOWN
        raise ConfigError(
            f"The training data has {len(report.errors)} error(s): {shown}"
            + (f" (+{more} more)" if more > 0 else ""),
            hint=f"Run `atlasforge dataset validate {config.train_file} --task generation"
            + (f" --against {config.eval_file}" if config.eval_file else "")
            + "` for the full list.",
        )

    parsed = parse_file(config.train_file, "generation")
    examples = [example for _, example in parsed.examples]
    unanswered = [e.id for e in examples if e.reference is None]
    if unanswered:
        sample = ", ".join(unanswered[:MAX_ERRORS_SHOWN])
        raise ConfigError(
            f"{len(unanswered)} training example(s) have no 'reference' (the answer to learn): "
            f"{sample}.",
            hint="Every training example needs a 'reference'.",
        )
    if len(examples) < config.min_examples:
        raise ConfigError(
            f"Only {len(examples)} training example(s); at least {config.min_examples} are "
            "required.",
            hint="Fine-tuning on a handful of examples mostly teaches the model to repeat them. "
            "Add data, or lower min_examples if you really mean it.",
        )

    conversations = tuple(
        (*e.messages, {"role": "assistant", "content": e.reference or ""}) for e in examples
    )
    return TrainingData(conversations=conversations, sha256=parsed.sha256)  # type: ignore[arg-type]


def plan_run(config: QLoRAConfig, data: TrainingData) -> TrainingPlan:
    steps_per_epoch = math.ceil(len(data) / config.effective_batch_size)
    words = [sum(len(m["content"].split()) for m in conv) for conv in data.conversations]
    return TrainingPlan(
        n_examples=len(data),
        steps_per_epoch=steps_per_epoch,
        total_steps=steps_per_epoch * config.num_epochs,
        effective_batch_size=config.effective_batch_size,
        median_words=statistics.median(words),
    )


def render_texts(data: TrainingData, tokenizer: Any) -> list[str]:
    """Each conversation as one training string, via the model's own chat template."""
    if not getattr(tokenizer, "chat_template", None):
        raise ConfigError(
            "The tokenizer has no chat template, so training text cannot be built the same way "
            "the model is prompted at inference.",
            hint="Record how N-ATLaS expects prompts in planning/21, then add support.",
        )
    return [
        tokenizer.apply_chat_template(list(conv), tokenize=False) for conv in data.conversations
    ]


def count_truncated(texts: Sequence[str], tokenizer: Any, max_len: int) -> int:
    """How many rendered examples exceed ``max_len`` tokens (they would be cut mid-answer)."""
    encoded = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return sum(len(ids) > max_len for ids in encoded)


def check_truncation(n_truncated: int, n_total: int, max_len: int) -> None:
    """Refuse when many examples would be cut off: that teaches the model to stop mid-answer."""
    if n_total and n_truncated / n_total > MAX_TRUNCATED_SHARE:
        raise ConfigError(
            f"{n_truncated} of {n_total} training examples exceed max_seq_len ({max_len} tokens) "
            "and would be cut off mid-answer.",
            hint="Raise max_seq_len (more memory) or shorten or remove the long examples.",
        )


def describe(config: QLoRAConfig, data: TrainingData) -> str:
    """Human-readable summary used by ``finetune --dry-run``."""
    plan = plan_run(config, data)
    revision = f" @ {config.revision}" if config.revision else ""
    method = "QLoRA (4-bit)" if config.quantize == "4bit" else "LoRA (no quantisation)"
    held_out = config.eval_file or "(none given: leakage was NOT checked)"
    lines = [
        f"Base model:        {config.base_model}{revision}",
        (
            f"Training file:     {config.train_file} ({plan.n_examples} examples, "
            f"fingerprint {data.sha256[:16]})"
        ),
        f"Held-out file:     {held_out}",
        f"Median length:     {plan.median_words:.0f} words per example",
        f"Method:            {method}, r={config.lora_r}, alpha={config.lora_alpha}",
        (
            f"Schedule:          {config.num_epochs} epoch(s), effective batch "
            f"{plan.effective_batch_size}, {plan.steps_per_epoch} step(s)/epoch, "
            f"{plan.total_steps} total"
        ),
        f"Output:            {config.output_dir}",
    ]
    return "\n".join(lines)
