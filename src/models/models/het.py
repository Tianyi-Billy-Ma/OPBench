from ..base import BaseModel


class HetModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.target_node_type = config.data.target_node_type

    def get_train_mask(self, batch):
        return batch[self.target_node_type].train_mask

    def get_val_mask(self, batch):
        return batch[self.target_node_type].val_mask

    def get_test_mask(self, batch):
        return batch[self.target_node_type].test_mask

    def get_targets(self, batch):
        return batch[self.target_node_type].y
