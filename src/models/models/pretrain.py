from ..base import BaseModel, BaseBackbone


class PTModel(BaseModel):
    def __init__(self, config, backbone: BaseBackbone):
        super().__init__(config)
        self.backbone = backbone

    def forward(self, batch):
        return self.backbone(batch)

    def compute_loss(self, batch, outputs, mask_type="train") -> dict:
        return self.backbone.compute_loss(batch, outputs)
