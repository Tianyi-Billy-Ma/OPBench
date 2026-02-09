from abc import ABC, abstractmethod
from typing import Any, Literal
import torch.nn as nn

from src.metrics import Evaluator


class BaseBackbone(ABC, nn.Module):
    """Abstract base class for all backbone models.

    Provides common utilities for normalization and activation that can be
    used by all GNN/HetGNN implementations.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...

    def compute_loss(self, batch, outputs: dict) -> dict:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support pretraining. "
            "Use PTModel only with backbones that implement compute_loss."
        )

    @abstractmethod
    def reset_parameters(self) -> None:
        raise NotImplementedError

    def _get_norm(
        self,
        channels: int,
        normalization: Literal["bn", "ln", "none"],
        apply_norm: bool,
    ) -> nn.Module:
        """Create a normalization layer.

        Args:
            channels: Number of input channels.
            normalization: Type of normalization ('bn', 'ln', or 'none').
            apply_norm: Whether to actually apply normalization.

        Returns:
            A normalization module or nn.Identity if disabled.
        """
        if not apply_norm:
            return nn.Identity()

        if normalization == "bn":
            return nn.BatchNorm1d(channels)
        elif normalization == "ln":
            return nn.LayerNorm(channels)
        else:
            return nn.Identity()

    def _get_activation(
        self, activation: Literal["relu", "elu", "prelu", "gelu", "id"]
    ) -> nn.Module:
        """Create an activation layer.

        Args:
            activation: Type of activation function.

        Returns:
            An activation module.
        """
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "elu":
            return nn.ELU(inplace=True)
        elif activation == "prelu":
            return nn.PReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            return nn.Identity()

    def _reset_norm_parameters(self, norms: nn.ModuleList) -> None:
        """Reset parameters for a list of normalization layers.

        Args:
            norms: ModuleList of normalization layers.
        """
        for norm in norms:
            if isinstance(norm, (nn.LayerNorm, nn.BatchNorm1d)):
                norm.reset_parameters()


class BaseModel(nn.Module, ABC):
    target_node_type: str | None = None
    backbone: "BaseBackbone | None"
    classifier: "BaseBackbone | None"
    evaluator: Evaluator

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = None
        self.classifier = None
        self.evaluator = Evaluator(num_classes=config.data.num_classes)

    def load_model(self):
        pass

    def get_train_mask(self, batch):
        return batch.train_mask

    def get_val_mask(self, batch):
        return batch.val_mask

    def get_test_mask(self, batch):
        return batch.test_mask

    def get_targets(self, batch):
        return batch.y

    @abstractmethod
    def forward(self, batch) -> dict:
        pass

    @abstractmethod
    def compute_loss(self, batch, outputs, mask_type="train") -> dict:
        raise NotImplementedError

    def _compute_metrics(
        self, batch, outputs, mask_type: Literal["train", "val", "test"] = "val"
    ) -> dict[str, float]:
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError(
                f"Cannot compute metrics: 'logits' not found in outputs. "
                f"Got keys: {list(outputs.keys())}"
            )

        targets = self.get_targets(batch)
        mask = getattr(self, f"get_{mask_type}_mask")(batch)

        if mask.sum() == 0:
            return self.evaluator.empty_metrics(f"{mask_type}_")

        return self.evaluator.evaluate(
            logits[mask], targets[mask], prefix=f"{mask_type}_"
        )

    def compute_metrics(
        self, batch, outputs, mask_type: str = "all"
    ) -> dict[str, float]:
        mask_types = mask_type.split(",")
        mask_set: set[str] = set()
        valid_types = {"train", "val", "test", "all"}
        for mt in mask_types:
            if mt in ["train", "val", "test"]:
                mask_set.add(mt)
            elif mt == "all":
                mask_set.update(["train", "val", "test"])
            else:
                raise ValueError(
                    f"Invalid mask_type: '{mt}'. Must be one of {valid_types} or comma-separated."
                )
        metrics_to_return: dict[str, float] = {}
        for mt in mask_set:
            metrics = self._compute_metrics(batch, outputs, mask_type=mt)  # type: ignore
            metrics_to_return.update(metrics)
        return metrics_to_return

    def freeze_backbone(self, freeze_backbone: bool = False) -> None:
        if freeze_backbone and isinstance(self.backbone, BaseBackbone):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def reset_parameters(self, reset_backbone: bool = True):
        if reset_backbone and isinstance(self.backbone, BaseBackbone):
            self.backbone.reset_parameters()
        if isinstance(self.classifier, BaseBackbone):
            self.classifier.reset_parameters()
