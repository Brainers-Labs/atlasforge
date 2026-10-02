"""Fine-tuning configuration: one small, validated file describing a QLoRA run.

Defaults are conventional starting points for a 7-8B model on a single 24 GB GPU, not
tuned values for N-ATLaS: they have not been measured on it. Treat them as a baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal

from atlasforge.backends.factory import DEFAULT_MODEL
from atlasforge.errors import ConfigError

# Llama-style attention and MLP projections. N-ATLaS is described as a Llama-3 8B fine-tune,
# so these names are expected to exist; confirm with the model's own config on first use.
DEFAULT_TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
MAX_SEQ_LEN_LIMIT: Final = 8192  # the model card's stated context window
MIN_SEQ_LEN: Final = 64
Quantize = Literal["4bit", "none"]


@dataclass(frozen=True, slots=True, kw_only=True)
class QLoRAConfig:
    """A QLoRA run. ``eval_file`` is only used to check for train/test leakage."""

    train_file: Path
    output_dir: Path
    eval_file: Path | None = None
    base_model: str = DEFAULT_MODEL
    revision: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    seed: int = 42
    quantize: Quantize = "4bit"
    min_examples: int = 20

    def __post_init__(self) -> None:
        checks = (
            (self.lora_r >= 1, "lora_r must be >= 1"),
            (self.lora_alpha >= 1, "lora_alpha must be >= 1"),
            (0 <= self.lora_dropout < 1, "lora_dropout must be in [0, 1)"),
            (bool(self.target_modules), "target_modules must not be empty"),
            (self.learning_rate > 0, "learning_rate must be > 0"),
            (self.num_epochs >= 1, "num_epochs must be >= 1"),
            (
                self.batch_size >= 1 and self.grad_accum >= 1,
                "batch_size and grad_accum must be >= 1",
            ),
            (
                MIN_SEQ_LEN <= self.max_seq_len <= MAX_SEQ_LEN_LIMIT,
                f"max_seq_len must be in [{MIN_SEQ_LEN}, {MAX_SEQ_LEN_LIMIT}]",
            ),
            (0 <= self.warmup_ratio < 1, "warmup_ratio must be in [0, 1)"),
            (self.logging_steps >= 1, "logging_steps must be >= 1"),
            (self.quantize in ("4bit", "none"), "quantize must be '4bit' or 'none'"),
            (self.min_examples >= 1, "min_examples must be >= 1"),
        )
        problems = [message for ok, message in checks if not ok]
        if problems:
            raise ConfigError("Invalid fine-tuning config: " + "; ".join(problems) + ".")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    def hyperparameters(self) -> dict[str, Any]:
        """The settings that shape the result, for ``training_run.json`` and the model card."""
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": ",".join(self.target_modules),
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "effective_batch_size": self.effective_batch_size,
            "max_seq_len": self.max_seq_len,
            "warmup_ratio": self.warmup_ratio,
            "seed": self.seed,
            "quantize": self.quantize,
        }


_REQUIRED: Final = ("train_file", "output_dir")
_PATH_FIELDS: Final = ("train_file", "output_dir", "eval_file")


def load_config(path: str | Path) -> QLoRAConfig:
    """Read a ``.yaml``/``.yml`` (needs PyYAML) or ``.json`` config.

    Relative paths inside it are resolved against the config file's own directory, so a
    config can be moved together with its data. Unknown keys are rejected: a typo in a
    hyper-parameter name must not be silently ignored.
    """
    file = Path(path)
    raw = _read_raw(file)
    if not isinstance(raw, dict):
        raise ConfigError(f"{file} must contain a mapping of settings.")

    known = {f.name for f in fields(QLoRAConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) in {file.name}: {', '.join(unknown)}.",
            hint=f"Known settings: {', '.join(sorted(known))}.",
        )
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ConfigError(f"{file.name} is missing: {', '.join(missing)}.")

    values = dict(raw)
    for key in _PATH_FIELDS:
        if values.get(key) is not None:
            candidate = Path(str(values[key]))
            values[key] = candidate if candidate.is_absolute() else file.parent / candidate
    if "target_modules" in values:
        modules = values["target_modules"]
        if not isinstance(modules, (list, tuple)) or not all(isinstance(m, str) for m in modules):
            raise ConfigError("target_modules must be a list of strings.")
        values["target_modules"] = tuple(modules)
    try:
        return QLoRAConfig(**values)
    except TypeError as exc:
        raise ConfigError(f"Invalid setting in {file.name}: {exc}") from exc


def _read_raw(file: Path) -> Any:
    """Parse the config file by extension."""
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Cannot read {file}: {exc.strerror or exc}", hint="Check the path."
        ) from exc
    suffix = file.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(text, file)
    if suffix != ".json":
        raise ConfigError(
            f"Unsupported config format {file.suffix!r}.", hint="Use .yaml, .yml or .json."
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{file} is not valid JSON: {exc.msg} (line {exc.lineno}).") from exc


def _load_yaml(text: str, file: Path) -> Any:
    try:
        import yaml  # noqa: PLC0415 - optional dependency, imported only for YAML configs
    except ImportError as exc:
        raise ConfigError(
            "Reading a YAML config needs PyYAML.",
            hint='pip install "atlasforge[finetune]", or use a .json config.',
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file} is not valid YAML.") from exc
