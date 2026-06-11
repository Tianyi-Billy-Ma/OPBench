import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.hparams import FullArguments
from src.models.registry import ModelRegistry

from ..base import BaseBackbone


@ModelRegistry.register("gat")
class GAT(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        num_heads = config.model.num_heads
        self.dropout = config.model.dropout
        normalization = config.model.normalization
        activation = config.model.activation
        input_norm = config.model.input_norm

        assert output_dim is not None, "output_dim must be set"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.norms.append(self._get_norm(num_features, normalization, input_norm))
            self.convs.append(GATConv(num_features, output_dim, heads=1, concat=False))
        else:
            self.norms.append(self._get_norm(num_features, normalization, input_norm))
            self.convs.append(GATConv(num_features, hidden_dim, heads=num_heads))

            for _ in range(num_layers - 2):
                self.norms.append(
                    self._get_norm(hidden_dim * num_heads, normalization, True)
                )
                self.convs.append(
                    GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
                )

            self.norms.append(
                self._get_norm(hidden_dim * num_heads, normalization, True)
            )
            self.convs.append(
                GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
            )

        self.activation = self._get_activation(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self._reset_norm_parameters(self.norms)

    def get_embedding(self, batch):
        x, edge_index = batch.x, batch.edge_index

        x = self.norms[0](x)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.norms[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x
