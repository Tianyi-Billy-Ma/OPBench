"""Sweep configuration."""

from dataclasses import dataclass, field
from typing import Any
from .base_config import BaseConfig


@dataclass
class SweepConfig(BaseConfig):
    """Configuration for hyperparameter sweeping.

    Attributes:
        parameters: Dictionary mapping parameter names (dot-notation) to list of values.
                    Example: {"train.lr": [0.01, 0.001], "model.hidden_dim": [64, 128]}
        sweep_metric: Target metric for sweep comparison (e.g., val_acc_mean, test_f1_macro_mean).
        higher_is_better: Whether higher metric values are better. Auto-set based on metric name.
    """

    parameters: dict[str, list[Any]] = field(
        default_factory=dict,
        metadata={"help": "Hyperparameter search space"},
    )
    sweep_metric: str = field(
        default="val_acc_mean",
        metadata={"help": "Target metric for sweep comparison"},
    )
    higher_is_better: bool | None = field(
        default=None,
        metadata={
            "help": "Whether higher metric values are better. Auto-set to False for 'loss' metrics."
        },
    )
