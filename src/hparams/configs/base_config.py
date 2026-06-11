"""Base configuration class with common methods."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TypeVar

import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class with serialization support.

    All configuration classes should inherit from this class to get
    common methods like from_yaml, to_yaml, and to_dict.
    """

    @classmethod
    def from_yaml(cls: type[T], yaml_path: str | Path) -> T:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Config object populated from YAML.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Filter only fields that exist in this dataclass
        valid_fields = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration.
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    def update(self, **kwargs) -> None:
        """Update configuration with new values.

        Args:
            **kwargs: Key-value pairs to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    def __post_init__(self):
        """Validate configuration after initialization."""
        pass
