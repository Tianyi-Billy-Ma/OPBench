from typing import List, Literal

from litlogger import LightningLogger

from src.hparams import FullArguments

VALID_LOGGERS = ["lit", "wandb", "none", None]


def parse_report_to(report_to: str | list[str]) -> list[str]:
    if isinstance(report_to, list):
        return [r.strip().lower() for r in report_to]
    return [r.strip().lower() for r in report_to.split(",")]


def cleanup_loggers() -> None:
    """Clean up any existing Weights & Biases run before creating new loggers."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


def setup_loggers(
    config: FullArguments,
    phase: Literal["pretrain", "finetune"] = "finetune",
) -> List:
    cleanup_loggers()

    logger_names = config.log.report_to
    loggers = []

    run_name = config.log.run_name
    if run_name:
        run_name = f"{run_name}_{phase}"
    else:
        run_name = f"{config.data.dataset}_{config.model.model_name}_{phase}"

    for name in logger_names:
        if name == "lit":
            logger = LightningLogger(
                name=run_name,
                root_dir=str(config.log.log_path),
                metadata={
                    "dataset": config.data.dataset,
                    "model": config.model.model_name,
                    "seed": str(config.train.seed),
                    "phase": phase,
                },
            )
            logger._version = ""
            loggers.append(logger)
        elif name == "wandb":
            try:
                from pytorch_lightning.loggers import WandbLogger
            except ImportError as e:
                raise ImportError(
                    "Weights & Biases logging requires the 'wandb' extra. "
                    "Install it with: pip install -e '.[wandb]'"
                ) from e
            logger = WandbLogger(
                project=config.log.project,
                entity=config.log.entity,
                name=run_name,
                save_dir=str(config.log.log_path),
                group=config.log.run_name,
                tags=[phase, config.data.dataset, config.model.model_name],
                config=config.to_dict(),
            )
            loggers.append(logger)

    return loggers
