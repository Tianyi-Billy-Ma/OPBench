import logging
import re
from datetime import datetime
from pathlib import Path

from src.hparams.configs import (
    DataConfig,
    LogConfig,
    ModelConfig,
    SweepConfig,
    TrainConfig,
)
from src.hparams.parser import FullArguments
from src.utils import parse_report_to

logger = logging.getLogger(__name__)


def _expand_run_name(run_name: str, config: FullArguments) -> str:
    """Expand placeholders in run_name.

    Supports placeholders like:
    - {mode} -> config.mode
    - {timestamp} -> current timestamp (YYYYMMDD_HHMMSS)
    - {model.model_name} -> config.model.model_name
    - {data.dataset} -> config.data.dataset
    - Any nested config attribute: {section.attribute}

    After expansion, dots in the result are replaced with 'p' (e.g., 0.1 -> 0p1).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def replace_placeholder(match: re.Match) -> str:
        key = match.group(1)

        if key == "timestamp":
            return timestamp

        if key == "mode":
            return str(config.mode)

        if "." in key:
            section, attr = key.split(".", 1)
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, attr):
                    value = str(getattr(section_obj, attr))
                    return value.replace(".", "p")

        logger.warning(f"Unknown placeholder in run_name: {{{key}}}")
        return f"{{{key}}}"

    return re.sub(r"\{([^}]+)\}", replace_placeholder, run_name)


def patch_train_config(config: TrainConfig) -> TrainConfig:
    if config.pretrain_epochs is None:
        config.pretrain_epochs = config.epochs
    if config.pretrain_lr is None:
        config.pretrain_lr = config.lr
    if config.pretrain_weight_decay is None:
        config.pretrain_weight_decay = config.weight_decay

    if config.finetune_epochs is None:
        config.finetune_epochs = config.epochs

    if config.device in ["cpu", "gpu", "tpu"]:
        config.accelerator = config.device
    else:
        config.accelerator = "auto"

    return config


def patch_data_config(config: DataConfig) -> DataConfig:
    config.test_prop = 1.0 - config.train_prop - config.val_prop
    return config


def patch_sweep_config(config: SweepConfig) -> SweepConfig:
    if config.higher_is_better is None:
        config.higher_is_better = "loss" not in config.sweep_metric.lower()
    return config


def patch_model_config(config: ModelConfig) -> ModelConfig:
    # Patch output_dim
    if config.output_dim is None:
        config.output_dim = config.hidden_dim

    # Patch classifier dropout
    if config.classifier_dropout is None:
        config.classifier_dropout = config.dropout
    return config


def patch_log_config(config: FullArguments) -> LogConfig:
    log_config = config.log

    # Base output dir
    output_dir = Path(log_config.output_dir)

    # Determine save_dir
    if log_config.save_dir:
        # If manually specified
        save_path = Path(log_config.save_dir)
    elif log_config.run_name:
        # If run_name specified, expand placeholders
        expanded_name = _expand_run_name(log_config.run_name, config)
        log_config.run_name = expanded_name  # Update run_name to expanded version
        save_path = output_dir / expanded_name
    else:
        # Default construction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = config.mode
        dataset = config.data.dataset
        model_name = config.model.model_name
        dir_name = f"{mode}_{dataset}_{model_name}_{timestamp}"
        save_path = output_dir / dir_name
        log_config.run_name = dir_name

    # Set paths
    log_config.save_dir = str(save_path)
    log_config.finetune_dir = str(save_path / "finetune")
    log_config.eval_dir = str(save_path / "eval")
    log_config.cfg_dir = str(save_path / "config")
    log_config.pretrain_dir = str(save_path / "pretrain")
    log_config.log_dir = str(save_path / "logs")

    # Normalize report_to to list
    log_config.report_to = parse_report_to(log_config.report_to)

    return log_config


def patch_config(config: FullArguments, overrides: dict | None = None) -> FullArguments:
    # 1. Apply overrides
    if overrides:
        for key, value in overrides.items():
            # Support nested keys like "data.num_features"
            if "." in key:
                section, attr = key.split(".", 1)
                if hasattr(config, section):
                    setattr(getattr(config, section), attr, value)
            else:
                # Direct attribute on config? Unlikely given structure, but handling it.
                if hasattr(config, key):
                    setattr(config, key, value)

    config.data = patch_data_config(config.data)
    config.model = patch_model_config(config.model)
    config.train = patch_train_config(config.train)
    config.log = patch_log_config(config)
    config.sweep = patch_sweep_config(config.sweep)

    return config


def update_log_paths(config: FullArguments, new_save_dir: Path) -> None:
    """Update log paths without creating directories."""
    config.log.save_dir = str(new_save_dir)
    config.log.finetune_dir = str(new_save_dir / "finetune")
    config.log.eval_dir = str(new_save_dir / "eval")
    config.log.cfg_dir = str(new_save_dir / "config")
    config.log.pretrain_dir = str(new_save_dir / "pretrain")
    config.log.log_dir = str(new_save_dir / "logs")


def create_output_dirs(config: FullArguments) -> None:
    config.log.finetune_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    config.log.eval_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    config.log.cfg_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    config.log.pretrain_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    config.log.log_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
