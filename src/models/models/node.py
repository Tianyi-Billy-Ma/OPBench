from typing import Literal

import torch.nn.functional as F
from torch_geometric.data import HeteroData

from ..base import BaseBackbone
from ..mlp import MLP
from ..registry import ModelRegistry
from .het import HetModel


class NodeModel(HetModel):
    def __init__(
        self,
        config,
        backbone: BaseBackbone | None = None,
        reset_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__(config)
        self.backbone = backbone
        self._backbone_is_heterogeneous = ModelRegistry.is_heterogeneous(
            config.model.model_name.lower()
        )

        input_dim = config.model.output_dim
        assert input_dim is not None, "config.model.output_dim must be set"

        num_classes = config.data.num_classes
        num_layers = config.model.classifier_num_layers
        hidden_dim = config.model.classifier_hidden
        dropout = config.model.classifier_dropout or config.model.dropout

        self.classifier = MLP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            normalization="none",
            input_norm=False,
        )
        self.reset_parameters(reset_backbone=reset_backbone)
        self.freeze_backbone(
            freeze_backbone=freeze_backbone and isinstance(backbone, BaseBackbone)
        )

    def forward(self, batch):
        assert self.backbone is not None, "Backbone must be set before forward pass"
        outputs = self.backbone(batch)
        embeddings = outputs["embeddings"]

        if isinstance(embeddings, dict):
            if self.target_node_type:
                z = embeddings[self.target_node_type]
            else:
                raise ValueError(
                    "Embeddings is a dict (heterogeneous) but target_node_type is not set."
                )
        else:
            z = embeddings

        logits = self.classifier(z)
        return {"logits": logits, "embeddings": embeddings}

    def get_train_mask(self, batch):
        if isinstance(batch, HeteroData) and self.target_node_type:
            return batch[self.target_node_type].train_mask
        return batch.train_mask

    def get_val_mask(self, batch):
        if isinstance(batch, HeteroData) and self.target_node_type:
            return batch[self.target_node_type].val_mask
        return batch.val_mask

    def get_test_mask(self, batch):
        if isinstance(batch, HeteroData) and self.target_node_type:
            return batch[self.target_node_type].test_mask
        return batch.test_mask

    def get_targets(self, batch):
        if isinstance(batch, HeteroData) and self.target_node_type:
            return batch[self.target_node_type].y
        return batch.y

    def compute_loss(
        self, batch, outputs, mask_type: Literal["train", "val", "test", "all"] = "all"
    ):
        mask_types = mask_type.split(",")
        mask_set = set()
        for mt in mask_types:
            if mt in ["train", "val", "test"]:
                mask_set.add(mt)
            elif mt == "all":
                mask_set.update(["train", "val", "test"])
            else:
                raise ValueError(
                    f"Invalid mask_type: {mask_type}. Must be 'train', 'val', 'test', or 'all'."
                )
        loss_to_return = {}
        for mt in mask_types:
            loss = self._compute_loss(batch, outputs, mask_type=mt)
            loss_to_return.update({f"{mt}_loss": loss["loss"]})
        return loss_to_return

    def _compute_loss(
        self, batch, outputs, mask_type: Literal["train", "val", "test"] = "train"
    ):
        logits = outputs["logits"]
        targets = self.get_targets(batch)
        mask = getattr(self, f"get_{mask_type}_mask")(batch)

        loss = F.cross_entropy(logits[mask], targets[mask])
        return {"loss": loss}
