from pathlib import Path
from typing import Literal

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.hparams import FullArguments


def setup_callbacks(
    config: FullArguments,
    phase: Literal["pretrain", "finetune"] = "finetune",
) -> list:
    callbacks = []

    if phase == "pretrain":
        dirpath = config.log.pretrain_path
        assert dirpath is not None, "Checkpoint directory must be set for pretrain"

        monitor = config.train.pretrain_monitor
        mode = "min" if "loss" in monitor else "max"

        if config.train.pretrain_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    patience=config.train.pretrain_patience,
                    verbose=True,
                )
            )

        # Pretrain always saves checkpoints (required for loading into finetune phase)
        callbacks.append(
            ModelCheckpoint(
                dirpath=dirpath,
                filename="best-pretrain-{epoch:02d}",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_last=True,
            )
        )

    elif phase == "finetune":
        assert config.log.finetune_path is not None, (
            "Finetune path must be set before setting up callbacks"
        )
        finetune_dir = config.log.finetune_path

        monitor = config.train.finetune_monitor
        mode = "max" if "acc" in monitor or "auc" in monitor else "min"

        if config.train.patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode=mode,
                    patience=config.train.patience,
                    min_delta=config.train.min_delta,
                    verbose=True,
                )
            )

        if config.log.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=finetune_dir,
                    filename="{epoch}-{step}",
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                )
            )

    return callbacks


def get_checkpoint_callback(callbacks: list) -> ModelCheckpoint | None:
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            return cb
    return None


def load_checkpoint(
    callbacks: list,
    module: "torch.nn.Module",
    phase: str = "pretrain",
) -> None:
    """Load the best checkpoint from callbacks into the module.

    Args:
        callbacks: List of callbacks containing ModelCheckpoint.
        module: PyTorch Lightning module to load state into.
        phase: Phase name for logging (pretrain or finetune).
    """
    checkpoint_cb = get_checkpoint_callback(callbacks)
    if checkpoint_cb is None:
        return

    best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    if best_path and Path(best_path).exists():
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded best {phase} checkpoint: {best_path}")
