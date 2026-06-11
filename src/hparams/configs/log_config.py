"""Logging-related configuration."""

from dataclasses import dataclass, field
from pathlib import Path

from .base_config import BaseConfig


@dataclass
class LogConfig(BaseConfig):
    """Logging configuration.

    Controls experiment tracking and logging behavior.

    Attributes:
        project: W&B project name.
        entity: W&B entity (username or team).
        use_wandb: Whether to use Weights & Biases logging.
        log_every_n_steps: Logging frequency.
        save_checkpoints: Whether to save model checkpoints.
        output_dir: Base output directory (default: "outputs").
        save_dir: Specific run directory. If None, constructed automatically.
    """

    # W&B configuration
    project: str = field(
        default="opbench",
        metadata={"help": "W&B project name"},
    )
    entity: str | None = field(
        default=None,
        metadata={"help": "W&B entity (username or team)"},
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to use Weights & Biases logging"},
    )

    report_to: str | list[str] = field(
        default="lit",
        metadata={
            "help": "Logger(s) to use. Options: lit, wandb. "
            "Can be comma-separated string or list."
        },
    )

    # Logging behavior
    log_every_n_steps: int = field(
        default=1,
        metadata={"help": "Logging frequency (every N steps)"},
    )
    save_checkpoints: bool = field(
        default=True,
        metadata={"help": "Whether to save model checkpoints"},
    )

    # Output configuration
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Output directory for experiment results"},
    )
    run_name: str | None = field(
        default=None,
        metadata={
            "help": "Name of the run. Supports placeholders: {mode}, {timestamp}, "
            "{model.model_name}, {data.dataset}, {section.attribute}"
        },
    )

    save_dir: str | None = field(
        default=None,
        metadata={
            "help": "Specific run directory. If specified, overrides default construction."
        },
    )

    # Computed fields (set by patcher)
    finetune_dir: str | None = field(
        default=None,
        metadata={"help": "Finetune checkpoints directory (computed)"},
    )
    eval_dir: str | None = field(
        default=None,
        metadata={"help": "Evaluation results directory (computed)"},
    )
    cfg_dir: str | None = field(
        default=None,
        metadata={"help": "Configuration directory (computed)"},
    )
    pretrain_dir: str | None = field(
        default=None,
        metadata={"help": "Pretrain checkpoints directory (computed)"},
    )
    log_dir: str | None = field(
        default=None,
        metadata={"help": "Logs directory (computed)"},
    )

    # Verbosity
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print verbose output"},
    )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def save_path(self) -> Path | None:
        return Path(self.save_dir) if self.save_dir else None

    @property
    def finetune_path(self) -> Path | None:
        return Path(self.finetune_dir) if self.finetune_dir else None

    @property
    def eval_path(self) -> Path | None:
        return Path(self.eval_dir) if self.eval_dir else None

    @property
    def cfg_path(self) -> Path | None:
        return Path(self.cfg_dir) if self.cfg_dir else None

    @property
    def pretrain_path(self) -> Path | None:
        return Path(self.pretrain_dir) if self.pretrain_dir else None

    @property
    def log_path(self) -> Path | None:
        return Path(self.log_dir) if self.log_dir else None

    def __post_init__(self):
        """Validate logging configuration."""
        super().__post_init__()

        if self.log_every_n_steps < 1:
            raise ValueError(
                f"log_every_n_steps must be >= 1, got {self.log_every_n_steps}"
            )
