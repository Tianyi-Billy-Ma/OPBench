import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv

from src.hparams import FullArguments
from src.models.registry import ModelRegistry

from ..base import BaseBackbone


class HANEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        in_channels,
        hidden_dim,
        out_dim,
        num_layers,
        heads,
        dropout,
        normalization="bn",
        activation="elu",
        input_norm=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.node_types = metadata[0]
        self.dropout = dropout

        self.input_norms = nn.ModuleDict()
        if input_norm:
            for node_type in self.node_types:
                if normalization == "bn":
                    self.input_norms[node_type] = nn.BatchNorm1d(in_channels)
                elif normalization == "ln":
                    self.input_norms[node_type] = nn.LayerNorm(in_channels)
                else:
                    self.input_norms[node_type] = nn.Identity()
        else:
            for node_type in self.node_types:
                self.input_norms[node_type] = nn.Identity()

        self.inter_norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(
                HANConv(
                    in_channels=in_channels,
                    out_channels=out_dim,
                    heads=heads,
                    metadata=metadata,
                    dropout=dropout,
                )
            )
        else:
            self.layers.append(
                HANConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=heads,
                    metadata=metadata,
                    dropout=dropout,
                )
            )
            self._add_inter_norm(hidden_dim, normalization)

            for _ in range(num_layers - 2):
                self.layers.append(
                    HANConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=heads,
                        metadata=metadata,
                        dropout=dropout,
                    )
                )
                self._add_inter_norm(hidden_dim, normalization)

            self.layers.append(
                HANConv(
                    in_channels=hidden_dim,
                    out_channels=out_dim,
                    heads=heads,
                    metadata=metadata,
                    dropout=dropout,
                )
            )

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def _add_inter_norm(self, channels, normalization):
        norm_dict = nn.ModuleDict()
        for node_type in self.node_types:
            if normalization == "bn":
                norm_dict[node_type] = nn.BatchNorm1d(channels)
            elif normalization == "ln":
                norm_dict[node_type] = nn.LayerNorm(channels)
            else:
                norm_dict[node_type] = nn.Identity()
        self.inter_norms.append(norm_dict)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm_dict in [self.input_norms] + list(self.inter_norms):
            for norm in norm_dict.values():
                if isinstance(norm, (nn.LayerNorm, nn.BatchNorm1d)):
                    norm.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.input_norms[node_type](x) for node_type, x in x_dict.items()
        }

        for i, conv in enumerate(self.layers):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.layers) - 1:
                norm_dict = self.inter_norms[i]
                x_dict = {
                    node_type: norm_dict[node_type](x)
                    for node_type, x in x_dict.items()
                }
                x_dict = {key: self.activation(x) for key, x in x_dict.items()}
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }
        return x_dict


@ModelRegistry.register("han", heterogeneous=True)
class HAN(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        metadata = config.data.metadata
        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        num_heads = config.model.num_heads
        dropout = config.model.dropout
        normalization = config.model.normalization
        activation = config.model.activation
        input_norm = config.model.input_norm

        assert output_dim is not None, "output_dim must be set"

        self.encoder = HANEncoder(
            metadata=metadata,
            in_channels=num_features,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_layers=num_layers,
            heads=num_heads,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
            input_norm=input_norm,
        )

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()

    def get_embedding(self, batch):
        return self.encoder(batch.x_dict, batch.edge_index_dict)
