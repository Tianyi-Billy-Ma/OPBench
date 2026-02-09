import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from ..base import BaseBackbone
from src.hparams import FullArguments
from src.models.registry import ModelRegistry


@ModelRegistry.register("rgcn")
class RGCN(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        num_relations = config.model.num_relations
        num_bases = config.model.rgcn_num_bases
        self.dropout = config.model.dropout
        normalization = config.model.normalization
        activation = config.model.activation
        input_norm = config.model.input_norm

        assert output_dim is not None, "output_dim must be set (should be patched)"
        assert num_relations is not None, "num_relations must be set for RGCN"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.norms.append(self._get_norm(num_features, normalization, input_norm))
            self.convs.append(
                RGCNConv(num_features, output_dim, num_relations, num_bases=num_bases)
            )
        else:
            self.norms.append(self._get_norm(num_features, normalization, input_norm))
            self.convs.append(
                RGCNConv(num_features, hidden_dim, num_relations, num_bases=num_bases)
            )
            for _ in range(num_layers - 2):
                self.norms.append(self._get_norm(hidden_dim, normalization, True))
                self.convs.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
                )
            self.norms.append(self._get_norm(hidden_dim, normalization, True))
            self.convs.append(
                RGCNConv(hidden_dim, output_dim, num_relations, num_bases=num_bases)
            )

        self.activation = self._get_activation(activation)

    def get_embedding(self, batch):
        x, edge_index, edge_type = batch.x, batch.edge_index, batch.edge_type

        x = self.norms[0](x)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            x = self.activation(x)
            x = self.norms[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, edge_index, edge_type)
        return h

    def forward(self, batch) -> dict:
        h = self.get_embedding(batch)
        return {"embeddings": h}

    def compute_loss(self, batch, outputs: dict) -> dict:
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        self._reset_norm_parameters(self.norms)
