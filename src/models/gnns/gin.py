import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from ..base import BaseBackbone
from src.hparams import FullArguments
from src.models.registry import ModelRegistry


@ModelRegistry.register("gin")
class GIN(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        self.dropout = config.model.dropout
        normalization = config.model.normalization
        activation = config.model.activation
        input_norm = config.model.input_norm

        assert output_dim is not None, "output_dim must be set"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.input_norm_layer = self._get_norm(num_features, normalization, input_norm)
        self.activation = self._get_activation(activation)

        in_dim = num_features
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(self._get_norm(out_dim, normalization, True))
            in_dim = out_dim

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self._reset_norm_parameters(self.norms)
        if isinstance(self.input_norm_layer, (nn.LayerNorm, nn.BatchNorm1d)):
            self.input_norm_layer.reset_parameters()

    def get_embedding(self, batch):
        x, edge_index = batch.x, batch.edge_index

        x = self.input_norm_layer(x)

        for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms[:-1])):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)
        return x

    def forward(self, batch) -> dict:
        h = self.get_embedding(batch)
        return {"embeddings": h}

    def compute_loss(self, batch, outputs: dict) -> dict:
        return {"loss": torch.tensor(0.0, requires_grad=True)}
