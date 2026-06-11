from typing import Type

from src.hparams import FullArguments
from src.models.base import BaseBackbone


class ModelRegistry:
    _registry: dict[str, Type[BaseBackbone]] = {}
    _heterogeneous: dict[str, bool] = {}

    @classmethod
    def register(cls, name: str, heterogeneous: bool = False):
        def decorator(model_cls: Type[BaseBackbone]):
            cls._registry[name] = model_cls
            cls._heterogeneous[name] = heterogeneous
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseBackbone]:
        if name not in cls._registry:
            raise ValueError(
                f"Model '{name}' not found in registry. Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def is_heterogeneous(cls, name: str) -> bool:
        return cls._heterogeneous.get(name, False)

    @classmethod
    def build(cls, name: str, config: FullArguments) -> BaseBackbone:
        model_cls = cls.get(name)
        return model_cls(config)
