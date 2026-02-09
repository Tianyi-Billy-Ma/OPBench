from src.hparams import FullArguments
from src.utils import VALID_LOGGERS


def verify_config(config: FullArguments) -> None:
    _verify_data_config(config)
    _verify_model_config(config)
    _verify_train_config(config)
    _verify_log_config(config)
    _verify_sweep_config(config)


def _verify_log_config(config: FullArguments) -> None:
    log = config.log
    if not log.save_dir:
        raise ValueError("log.save_dir is not set. Did you run patch_config?")
    if not log.finetune_dir:
        raise ValueError("log.finetune_dir is not set. Did you run patch_config?")
    if not log.eval_dir:
        raise ValueError("log.eval_dir is not set. Did you run patch_config?")
    if not log.cfg_dir:
        raise ValueError("log.cfg_dir is not set. Did you run patch_config?")
    if not log.pretrain_dir:
        raise ValueError("log.pretrain_dir is not set. Did you run patch_config?")
    if not log.log_dir:
        raise ValueError("log.log_dir is not set. Did you run patch_config?")

    for logger_name in log.report_to:
        if logger_name not in VALID_LOGGERS:
            raise ValueError(
                f"Invalid logger '{logger_name}' in report_to. "
                f"Valid options: {VALID_LOGGERS}"
            )


def _verify_data_config(config: FullArguments) -> None:
    data = config.data

    total_prop = data.train_prop + data.val_prop + data.test_prop
    if abs(total_prop - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, got {total_prop:.4f}")

    if data.train_prop <= 0:
        raise ValueError(f"train_prop must be > 0, got {data.train_prop}")
    if data.val_prop < 0:
        raise ValueError(f"val_prop must be >= 0, got {data.val_prop}")
    if data.test_prop < 0:
        raise ValueError(f"test_prop must be >= 0, got {data.test_prop}")

    if data.num_features <= 0:
        raise ValueError(f"num_features must be > 0, got {data.num_features}")
    if data.num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {data.num_classes}")


def _verify_model_config(config: FullArguments) -> None:
    model = config.model

    if model.num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {model.num_layers}")
    if model.hidden_dim < 1:
        raise ValueError(f"hidden_dim must be >= 1, got {model.hidden_dim}")
    if not 0 <= model.dropout <= 1:
        raise ValueError(f"dropout must be in [0, 1], got {model.dropout}")
    if not 0 <= model.hgmae_mask_rate <= 1:
        raise ValueError(
            f"hgmae_mask_rate must be in [0, 1], got {model.hgmae_mask_rate}"
        )


def _verify_train_config(config: FullArguments) -> None:
    train = config.train

    if train.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {train.epochs}")
    if train.lr <= 0:
        raise ValueError(f"lr must be > 0, got {train.lr}")
    if train.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {train.weight_decay}")

    # Validate monitor values
    valid_pretrain_monitors = {"train_loss"}
    valid_finetune_monitors = {"val_acc", "val_loss", "val_auc"}

    if train.pretrain_monitor not in valid_pretrain_monitors:
        raise ValueError(
            f"pretrain_monitor must be one of {valid_pretrain_monitors}, "
            f"got '{train.pretrain_monitor}'"
        )

    if train.finetune_monitor not in valid_finetune_monitors:
        raise ValueError(
            f"finetune_monitor must be one of {valid_finetune_monitors}, "
            f"got '{train.finetune_monitor}'"
        )


def _verify_sweep_config(config: FullArguments) -> None:
    sweep = config.sweep

    if sweep.higher_is_better is None:
        raise ValueError(
            "sweep.higher_is_better must be set after patching. Did you run patch_config?"
        )

    if not sweep.sweep_metric:
        raise ValueError("sweep.sweep_metric must be a non-empty string")
