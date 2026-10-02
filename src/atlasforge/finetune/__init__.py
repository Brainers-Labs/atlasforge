"""Fine-tuning recipes. Heavy dependencies (torch, peft, trl) are imported lazily."""

from atlasforge.finetune.config import QLoRAConfig, load_config
from atlasforge.finetune.data import TrainingData, describe, plan_run, prepare_data
from atlasforge.finetune.qlora import RUN_FILE, TrainingRun, train

__all__ = [
    "RUN_FILE",
    "QLoRAConfig",
    "TrainingData",
    "TrainingRun",
    "describe",
    "load_config",
    "plan_run",
    "prepare_data",
    "train",
]
