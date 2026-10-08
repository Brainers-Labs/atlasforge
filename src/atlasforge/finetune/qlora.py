"""QLoRA fine-tuning of N-ATLaS with PEFT and TRL.

**Status: written against the libraries' documented APIs, NOT yet run on a GPU.** The wiring
is tested with stand-in modules; real behaviour (memory, speed, loss) needs an NVIDIA GPU
with enough memory, and the first real run should be recorded in
``planning/21_NATLAS_DISCOVERY.md``.

Output is a LoRA *adapter* only. N-ATLaS weights are never saved, copied or redistributed;
the base model is always loaded from Hugging Face under its own licence.

TRL has renamed arguments across releases (``max_seq_length`` -> ``max_length``,
``tokenizer`` -> ``processing_class``), so those two are chosen by inspecting the installed
signatures rather than by guessing a version.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

from atlasforge import __version__
from atlasforge.errors import BackendError, ConfigError, ResourceError
from atlasforge.finetune.data import check_truncation, count_truncated, prepare_data, render_texts

if TYPE_CHECKING:
    from atlasforge.finetune.config import QLoRAConfig

RUN_FILE: Final = "training_run.json"
_EXTRAS_HINT: Final = 'pip install "brainers-atlasforge[finetune]"'


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingRun:
    """What was trained, on what, with which settings. Feeds ``atlasforge card``."""

    method: str
    base_model: str
    base_revision: str | None
    n_train_examples: int
    n_truncated: int
    dataset_sha256: str
    atlasforge_version: str
    started_at: str
    finished_at: str
    output_dir: str
    hyperparameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train(config: QLoRAConfig) -> TrainingRun:
    """Fine-tune ``config.base_model`` on ``config.train_file`` and save the adapter."""
    data = prepare_data(config)  # validates first: fail before touching a GPU
    torch = _import("torch")
    transformers = _import("transformers")
    peft = _import("peft")
    trl = _import("trl")
    datasets = _import("datasets")

    if config.quantize == "4bit" and not torch.cuda.is_available():
        raise ResourceError(
            "QLoRA (4-bit) training needs an NVIDIA GPU, and none was found.",
            hint="Use a GPU machine (Colab, a cloud GPU, a university cluster). Training an 8B "
            "model on a Mac or CPU is not supported.",
        )

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.base_model, revision=config.revision
        )
        texts = render_texts(data, tokenizer)
        n_truncated = count_truncated(texts, tokenizer, config.max_seq_len)
        check_truncation(n_truncated, len(texts), config.max_seq_len)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.base_model,
            revision=config.revision,
            **_model_kwargs(config, torch, transformers),
        )
        if config.quantize == "4bit":
            model = peft.prepare_model_for_kbit_training(model)

        lora = peft.LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        trainer = trl.SFTTrainer(
            **_trainer_kwargs(
                trl.SFTTrainer,
                model=model,
                args=trl.SFTConfig(**_sft_kwargs(config, trl.SFTConfig, _use_bf16(torch))),
                dataset=datasets.Dataset.from_dict({"text": texts}),
                peft_config=lora,
                tokenizer=tokenizer,
            )
        )
        trainer.train()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))
    except (ConfigError, ResourceError):
        raise
    except Exception as exc:
        raise _map_error(exc) from exc

    run = TrainingRun(
        method="qlora" if config.quantize == "4bit" else "lora",
        base_model=config.base_model,
        base_revision=config.revision,
        n_train_examples=len(data),
        n_truncated=n_truncated,
        dataset_sha256=data.sha256,
        atlasforge_version=__version__,
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        output_dir=str(config.output_dir),
        hyperparameters=config.hyperparameters(),
    )
    (config.output_dir / RUN_FILE).write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return run


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(
            f"Fine-tuning needs '{name}', which is not installed.", hint=_EXTRAS_HINT
        ) from exc


def _use_bf16(torch: Any) -> bool:
    check = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(check()) if callable(check) else False


def _model_kwargs(config: QLoRAConfig, torch: Any, transformers: Any) -> dict[str, Any]:
    dtype = torch.bfloat16 if _use_bf16(torch) else torch.float16
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if config.quantize == "4bit":
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype
    return kwargs


def _sft_kwargs(config: QLoRAConfig, sft_config_cls: Any, bf16: bool) -> dict[str, Any]:
    """Arguments for ``SFTConfig``, with the sequence-length name taken from its signature."""
    accepted = inspect.signature(sft_config_cls).parameters
    length_key = "max_length" if "max_length" in accepted else "max_seq_length"
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir / "checkpoints"),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "seed": config.seed,
        "bf16": bf16,
        "fp16": not bf16,
        "report_to": "none",
        "save_strategy": "no",  # the final adapter is saved explicitly
        "dataset_text_field": "text",
        length_key: config.max_seq_len,
    }
    return {k: v for k, v in kwargs.items() if k in accepted or "kwargs" in accepted}


def _trainer_kwargs(
    trainer_cls: Any,
    *,
    model: Any,
    args: Any,
    dataset: Any,
    peft_config: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Arguments for ``SFTTrainer``, with the tokenizer's name taken from its signature."""
    accepted = inspect.signature(trainer_cls).parameters
    tokenizer_key = "processing_class" if "processing_class" in accepted else "tokenizer"
    return {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "peft_config": peft_config,
        tokenizer_key: tokenizer,
    }


def _map_error(exc: Exception) -> Exception:
    if "OutOfMemory" in type(exc).__name__ or isinstance(exc, MemoryError):
        return ResourceError(
            "Ran out of memory during training.",
            hint="Lower max_seq_len, batch_size or lora_r (raise grad_accum to compensate), or "
            "use a GPU with more memory.",
        )
    name = type(exc).__name__
    if "Gated" in name or "RepositoryNotFound" in name:
        return BackendError(
            "Cannot access the base model on Hugging Face.",
            hint="Accept the model's licence and set HF_TOKEN. `atlasforge doctor` checks the token.",
        )
    return BackendError(
        f"Training failed ({name}).",
        hint="The message is withheld because it may contain training text; re-run with the "
        "failing step isolated to see it.",
    )
