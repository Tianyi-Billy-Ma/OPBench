import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

from src.hparams import FullArguments
from src.models.registry import ModelRegistry

from ..base import BaseBackbone


@ModelRegistry.register("hgt", heterogeneous=True)
class HGT(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        num_heads = config.model.num_heads
        dropout = config.model.dropout
        metadata = config.data.metadata
        normalization = config.model.normalization
        activation = config.model.activation
        input_norm = config.model.input_norm

        assert output_dim is not None, "output_dim must be set"

        self.node_types = metadata[0]
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.lin_dict = nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)

        self.input_norms = nn.ModuleDict()
        if input_norm:
            for node_type in self.node_types:
                if normalization == "bn":
                    self.input_norms[node_type] = nn.BatchNorm1d(hidden_dim)
                elif normalization == "ln":
                    self.input_norms[node_type] = nn.LayerNorm(hidden_dim)
                else:
                    self.input_norms[node_type] = nn.Identity()
        else:
            for node_type in self.node_types:
                self.input_norms[node_type] = nn.Identity()

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=out_dim,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)

        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            if normalization == "ln":
                norm_dict = nn.ModuleDict()
                for node_type in self.node_types:
                    norm_dict[node_type] = nn.LayerNorm(hidden_dim)
                self.norms.append(norm_dict)
            elif normalization == "bn":
                norm_dict = nn.ModuleDict()
                for node_type in self.node_types:
                    norm_dict[node_type] = nn.BatchNorm1d(hidden_dim)
                self.norms.append(norm_dict)
            else:
                norm_dict = nn.ModuleDict()
                for node_type in self.node_types:
                    norm_dict[node_type] = nn.Identity()
                self.norms.append(norm_dict)

        self.activation = self._get_activation(activation)

    def reset_parameters(self) -> None:
        from torch.nn.parameter import UninitializedParameter

        for lin in self.lin_dict.values():
            if not isinstance(lin.weight, UninitializedParameter):
                lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if norm is not None:
                if isinstance(norm, nn.ModuleDict):
                    for n in norm.values():
                        if isinstance(n, (nn.LayerNorm, nn.BatchNorm1d)):
                            n.reset_parameters()
                elif hasattr(norm, "reset_parameters"):
                    norm.reset_parameters()
        for norm in self.input_norms.values():
            if isinstance(norm, (nn.LayerNorm, nn.BatchNorm1d)):
                norm.reset_parameters()

    def get_embedding(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        x_dict = {
            node_type: self.input_norms[node_type](x) for node_type, x in x_dict.items()
        }

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.norms):
                norm = self.norms[i]
                if isinstance(norm, nn.ModuleDict):
                    x_dict = {
                        node_type: norm[node_type](x) for node_type, x in x_dict.items()
                    }
                else:
                    x_dict = norm(x_dict)
                x_dict = {key: self.activation(x) for key, x in x_dict.items()}
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }

        return x_dict
