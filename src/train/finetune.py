from .base import BaseTrainer


class FTTrainer(BaseTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.lr = config.train.lr
        self.weight_decay = config.train.weight_decay

    def training_step(self, batch, batch_idx, *args, **kwargs):
        outputs = self.model(batch)
        loss_dict = self.model.compute_loss(batch, outputs, mask_type="train")
        loss = loss_dict["train_loss"]
        batch_size = getattr(batch, "num_nodes", None)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        outputs = self.model(batch)
        loss_dict = self.model.compute_loss(batch, outputs, mask_type="val")
        metrics_dict = self.model.compute_metrics(batch, outputs, mask_type="all")
        batch_size = getattr(batch, "num_nodes", None)
        self.log(
            "val_loss", loss_dict["val_loss"], prog_bar=True, batch_size=batch_size
        )
        for k, v in metrics_dict.items():
            self.log(k, v, prog_bar=True, batch_size=batch_size)
        return loss_dict["val_loss"]

    def test_step(self, batch, batch_idx, *args, **kwargs):
        outputs = self.model(batch)
        metrics_dict = self.model.compute_metrics(batch, outputs, mask_type="all")
        batch_size = getattr(batch, "num_nodes", None)
        for k, v in metrics_dict.items():
            self.log(k, v, prog_bar=True, batch_size=batch_size)
