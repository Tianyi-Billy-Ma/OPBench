import pytorch_lightning as pl
import torch.optim as optim


class BaseTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.lr = config.train.lr
        self.weight_decay = config.train.weight_decay

    def reset_parameters(self):
        if hasattr(self.model, "reset_parameters"):
            self.model.reset_parameters()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
