"""Training-related configuration."""

from dataclasses import dataclass, field
from typing import Literal

from .base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration.

    Controls training hyperparameters, optimization, and scheduling.

    Attributes:
        mode: Training mode (sft, pretrain, finetune).
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        optimizer: Optimizer type.
        scheduler: Learning rate scheduler.
        patience: Early stopping patience (0 to disable).
        min_delta: Minimum improvement for early stopping.
        batch_size: Batch size for training.
        seed: Random seed for reproducibility.
        device: Device to use for training.
        gradient_clip_val: Gradient clipping value (0 to disable).
        eval_step: Evaluate every N epochs.

        # Pretrain specific
        pretrain_epochs: Number of pretraining epochs.
        pretrain_lr: Learning rate for pretraining.
        pretrain_weight_decay: Weight decay for pretraining.
        pretrain_patience: Early stopping patience for pretraining.

        # Finetune specific
        finetune_epochs: Number of finetuning epochs.
        freeze_backbone: Whether to freeze the backbone model.
    """

    # Training mode
    mode: Literal["gen", "ft"] = field(
        default="ft",
        metadata={"help": "Training mode: gen (pretrain+finetune), ft (finetune only)"},
    )

    # Basic training hyperparameters
    epochs: int = field(
        default=200,
        metadata={"help": "Maximum number of training epochs"},
    )
    lr: float = field(
        default=0.01,
        metadata={"help": "Learning rate"},
    )
    weight_decay: float = field(
        default=5e-4,
        metadata={"help": "Weight decay (L2 regularization)"},
    )

    # Optimization
    optimizer: Literal["adam", "adamw", "sgd"] = field(
        default="adam",
        metadata={"help": "Optimizer type: adam, adamw, sgd"},
    )
    scheduler: Literal["none", "cosine", "step", "plateau"] = field(
        default="none",
        metadata={"help": "Learning rate scheduler: none, cosine, step, plateau"},
    )

    # Early stopping
    patience: int = field(
        default=50,
        metadata={"help": "Early stopping patience (0 to disable)"},
    )
    min_delta: float = field(
        default=0.0,
        metadata={"help": "Minimum improvement for early stopping"},
    )

    # Batch and device
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for training"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )
    device: str = field(
        default="auto",
        metadata={"help": "Device to use: cuda, cpu, auto"},
    )
    accelerator: str = field(
        default="auto",
        metadata={"help": "PyTorch Lightning accelerator (computed from device)"},
    )
    gradient_clip_val: float = field(
        default=0.0,
        metadata={"help": "Gradient clipping value (0 to disable)"},
    )
    eval_step: int = field(
        default=1,
        metadata={"help": "Evaluate every N epochs"},
    )

    # Pretrain specific
    pretrain_epochs: int | None = field(
        default=None,
        metadata={"help": "Number of pretraining epochs (defaults to epochs)"},
    )
    pretrain_lr: float | None = field(
        default=None,
        metadata={"help": "Learning rate for pretraining (defaults to lr)"},
    )
    pretrain_weight_decay: float | None = field(
        default=None,
        metadata={"help": "Weight decay for pretraining (defaults to weight_decay)"},
    )
    pretrain_patience: int = field(
        default=20,
        metadata={"help": "Early stopping patience for pretraining"},
    )
    pretrain_monitor: str = field(
        default="train_loss",
        metadata={"help": "Metric to monitor for pretraining callbacks"},
    )

    # Finetune specific
    finetune_epochs: int | None = field(
        default=None,
        metadata={"help": "Number of finetuning epochs (defaults to epochs)"},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone model during finetuning"},
    )
    finetune_monitor: str = field(
        default="val_acc",
        metadata={"help": "Metric to monitor for finetuning callbacks"},
    )

    # Experiment settings
    num_runs: int = field(
        default=5,
        metadata={"help": "Number of runs for 'exp' mode"},
    )
    sweep_type: Literal["grid", "coordinate"] = field(
        default="grid",
        metadata={"help": "Sweep type: grid, coordinate"},
    )

    def __post_init__(self):
        super().__post_init__()
