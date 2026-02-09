from dataclasses import dataclass, field
from pathlib import Path
import sys
import yaml

from simple_parsing import ArgumentParser

from .configs import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    LogConfig,
    SweepConfig,
)


@dataclass
class FullArguments:
    mode: str = field(default="run")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "FullArguments":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        mode = config_dict.get("mode", "run")
        data_dict = config_dict.get("data", {})
        model_dict = config_dict.get("model", {})
        train_dict = config_dict.get("train", {})
        log_dict = config_dict.get("log", {})
        sweep_dict = config_dict.get("sweep", {})

        return cls(
            mode=mode,
            data=DataConfig(**data_dict),
            model=ModelConfig(**model_dict),
            train=TrainConfig(**train_dict),
            log=LogConfig(**log_dict),
            sweep=SweepConfig(**sweep_dict),
        )

    def to_yaml(self, yaml_path: str | Path) -> None:
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_configs(self, cfg_dir: str | Path) -> None:
        cfg_dir = Path(cfg_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        self.data.to_yaml(cfg_dir / "data_config.yaml")
        self.model.to_yaml(cfg_dir / "model_config.yaml")
        self.train.to_yaml(cfg_dir / "train_config.yaml")
        self.log.to_yaml(cfg_dir / "log_config.yaml")
        if self.sweep.parameters:
            self.sweep.to_yaml(cfg_dir / "sweep_config.yaml")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "log": self.log.to_dict(),
            "sweep": self.sweep.to_dict(),
        }


def parse_args() -> FullArguments:
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python -m src.main <mode> [config.yaml] [--overrides]")
        print("  mode: run, exp, or sweep")
        print("  config.yaml: optional config file path")
        print("  --overrides: optional command line overrides (e.g., --model_name gcn)")
        sys.exit(1)

    mode = args[0]
    if mode not in ("run", "exp", "sweep"):
        raise ValueError(f"Invalid mode: {mode}. Must be one of: run, exp, sweep")

    config_file = None
    override_start = 1

    if len(args) > 1 and not args[1].startswith("-"):
        config_file = args[1]
        override_start = 2

    if config_file:
        base_config = FullArguments.from_yaml(config_file)
    else:
        base_config = FullArguments()

    base_config.mode = mode

    parser = ArgumentParser(add_help=True)
    parser.add_arguments(DataConfig, dest="data", default=base_config.data)
    parser.add_arguments(ModelConfig, dest="model", default=base_config.model)
    parser.add_arguments(TrainConfig, dest="train", default=base_config.train)
    parser.add_arguments(LogConfig, dest="log", default=base_config.log)
    parser.add_arguments(SweepConfig, dest="sweep", default=base_config.sweep)

    parsed = parser.parse_args(args[override_start:])

    return FullArguments(
        mode=mode,
        data=parsed.data,
        model=parsed.model,
        train=parsed.train,
        log=parsed.log,
        sweep=parsed.sweep,
    )
