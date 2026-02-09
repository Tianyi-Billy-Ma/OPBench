from typing import Literal, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from ..base import BaseBackbone
from src.hparams import FullArguments
from src.models.registry import ModelRegistry
from src.models.mlp import MLP


class EDHNNConv(nn.Module):
    """ED-HNN convolution layer (EquivSetConv)."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        mlp1_layers: int,
        mlp2_layers: int,
        mlp3_layers: int,
        dropout: float = 0.0,
        normalization: Literal["bn", "ln", "none"] = "ln",
        aggregate: str = "mean",
        alpha: float = 0.0,
    ):
        super().__init__()

        self.aggregate = aggregate
        self.alpha = alpha
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # MLP1: Transform node features before V->E aggregation
        if mlp1_layers > 0:
            self.mlp1: nn.Module = MLP(
                in_channels=in_dim,
                hidden_channels=hid_dim,
                out_channels=hid_dim,
                num_layers=mlp1_layers,
                dropout=dropout,
                normalization=normalization,
            )
        else:
            self.mlp1 = nn.Identity()

        # MLP2: Transform [original node features, aggregated edge features] -> node features
        self.mlp2: MLP | None = None
        if mlp2_layers > 0:
            self.mlp2 = MLP(
                in_channels=in_dim + hid_dim,
                hidden_channels=hid_dim,
                out_channels=hid_dim,
                num_layers=mlp2_layers,
                dropout=dropout,
                normalization=normalization,
            )

        # MLP3: Final transformation
        if mlp3_layers > 0:
            self.mlp3: nn.Module = MLP(
                in_channels=hid_dim,
                hidden_channels=hid_dim,
                out_channels=out_dim,
                num_layers=mlp3_layers,
                dropout=dropout,
                normalization=normalization,
            )
        else:
            self.mlp3 = nn.Identity()

    def reset_parameters(self) -> None:
        if isinstance(self.mlp1, MLP):
            self.mlp1.reset_parameters()
        if self.mlp2 is not None:
            self.mlp2.reset_parameters()
        if isinstance(self.mlp3, MLP):
            self.mlp3.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        N = x.size(0)

        # MLP1: Transform node features
        x_transformed = self.mlp1(x)

        # V -> E aggregation: aggregate node features to hyperedges
        xve = x_transformed[vertex]  # [nnz, hid_dim]
        xe = scatter(xve, edges, dim=0, reduce=self.aggregate)  # [E, hid_dim]

        # E -> V aggregation: aggregate edge features back to nodes
        xev = xe[edges]  # [nnz, hid_dim]

        # MLP2: Transform concatenation of [original node features, edge features]
        if self.mlp2 is not None:
            xev = self.mlp2(torch.cat([x[vertex], xev], dim=-1))  # [nnz, hid_dim]

        xv = scatter(
            xev, vertex, dim=0, reduce=self.aggregate, dim_size=N
        )  # [N, hid_dim]

        # Restart: combine with initial features
        if self.alpha > 0:
            if x0.size(-1) == xv.size(-1):
                x_out = (1 - self.alpha) * xv + self.alpha * x0
            else:
                x_out = xv
        else:
            x_out = xv

        # MLP3: Final transformation
        x_out = self.mlp3(x_out)

        return x_out


@ModelRegistry.register("edhnn")
class EDHNN(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout

        mlp_layers = config.model.edhnn_num_layers_1
        mlp2_layers = config.model.edhnn_num_layers_2
        mlp3_layers = config.model.edhnn_num_layers_3
        self.restart_alpha = config.model.restart_alpha
        aggregate = config.model.aggregate
        normalization = cast(Literal["bn", "ln", "none"], config.model.normalization)
        activation = config.model.activation

        assert output_dim is not None, "output_dim must be set"

        # Use default if not specified (following reference logic)
        if mlp2_layers < 0:
            mlp2_layers = mlp_layers
        if mlp3_layers < 0:
            mlp3_layers = mlp_layers

        # Input linear layer
        self.lin_in = nn.Linear(num_features, hidden_dim)

        # Convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # Final layer maps to output_dim
            curr_out_dim = output_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(
                EDHNNConv(
                    in_dim=hidden_dim,
                    hid_dim=hidden_dim,
                    out_dim=curr_out_dim,
                    mlp1_layers=mlp_layers,
                    mlp2_layers=mlp2_layers,
                    mlp3_layers=mlp3_layers,
                    dropout=dropout,
                    normalization=normalization,
                    aggregate=aggregate,
                    alpha=self.restart_alpha,
                )
            )

        self.dropout = dropout
        self.act = self._get_activation(activation)

    def reset_parameters(self) -> None:
        self.lin_in.reset_parameters()
        for conv in self.convs:
            if isinstance(conv, EDHNNConv):
                conv.reset_parameters()

    def forward(self, *args, **kwargs) -> Any:
        if args:
            batch = args[0]
        else:
            batch = kwargs.get("batch")

        assert batch is not None, "Forward requires batch"

        x = batch.x
        num_nodes = x.size(0)
        edge_index = batch.edge_index

        # Extract V->E edges only
        mask = edge_index[0] < num_nodes
        v2e_edges = edge_index[:, mask]

        # Get vertex and edge indices
        vertex = v2e_edges[0]
        edges = v2e_edges[1]

        # Normalize hyperedge IDs to start from 0
        if edges.numel() > 0:
            edges = edges - edges.min()

        # Input transformation
        x = F.relu(self.lin_in(x))
        x0 = x

        # Convolution layers
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if isinstance(conv, EDHNNConv):
                x = conv(x, vertex, edges, x0)
            x = self.act(x)

        return {"embeddings": x}
