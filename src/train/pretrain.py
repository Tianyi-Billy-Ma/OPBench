from .base import BaseTrainer


class PTTrainer(BaseTrainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.lr = config.train.pretrain_lr
        self.weight_decay = config.train.pretrain_weight_decay
        self.reset_parameters()

    def training_step(self, batch, batch_idx):
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        outputs = self.model(data)
        loss_dict = self.model.compute_loss(data, outputs)
        loss = loss_dict["loss"]
        self.log("train_loss", loss, prog_bar=True, batch_size=1)
        return loss
