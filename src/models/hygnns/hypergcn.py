import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.hparams import FullArguments
from src.models.base import BaseBackbone
from src.models.registry import ModelRegistry


class HyperGCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mediators: bool = True,
        cached: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_mediators = use_mediators
        self.cached = cached

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self._cached_structure = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self._cached_structure = None

    def forward(
        self,
        x: torch.Tensor,
        He_dict: dict[int, list[int]],
        num_nodes: int,
        recompute: bool = False,
    ) -> torch.Tensor:
        if self._cached_structure is None or recompute or not self.cached:
            edge_index, edge_weight = self._compute_laplacian(x, He_dict, num_nodes)
            if self.cached:
                self._cached_structure = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = self._cached_structure

        x = x @ self.weight

        if edge_index.size(1) > 0:
            row, col = edge_index
            out = scatter(
                x[col] * edge_weight.unsqueeze(-1),
                row,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
        else:
            out = x

        out = out + self.bias
        return out

    def _compute_laplacian(
        self,
        x: torch.Tensor,
        He_dict: dict[int, list[int]],
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        rows, cols, weights = [], [], []

        for nodes in He_dict.values():
            if len(nodes) < 2:
                continue

            nodes_tensor = torch.tensor(nodes, device=device, dtype=torch.long)
            x_he = x[nodes_tensor]

            with torch.no_grad():
                x_norm = F.normalize(x_he, p=2, dim=1)
                sim = x_norm @ x_norm.T
                sim.fill_diagonal_(float("inf"))
                min_idx = sim.argmin()
                i = min_idx // len(nodes)
                j = min_idx % len(nodes)

                node_i = nodes[i]
                node_j = nodes[j]

            if self.use_mediators and len(nodes) > 2:
                w = 1.0 / (len(nodes) - 1)
                for k, node_k in enumerate(nodes):
                    if k != i and k != j:
                        rows.extend([node_i, node_k, node_j, node_k])
                        cols.extend([node_k, node_i, node_k, node_j])
                        weights.extend([w, w, w, w])
            else:
                w = 1.0
                rows.extend([node_i, node_j])
                cols.extend([node_j, node_i])
                weights.extend([w, w])

        if len(rows) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, device=device),
            )

        edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        edge_weight = torch.tensor(weights, device=device)

        deg = scatter(
            edge_weight, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum"
        )
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        edge_weight = (
            deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]
        )
        return edge_index, edge_weight


@ModelRegistry.register("hypergcn")
class HyperGCN(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        self.use_mediators = config.model.hypergcn_mediators
        self.fast = config.model.hypergcn_fast

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(
            HyperGCNConv(num_features, hidden_dim, self.use_mediators, cached=False)
        )
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for i in range(num_layers - 1):
            cached = self.fast and (i > 0)
            self.convs.append(
                HyperGCNConv(hidden_dim, hidden_dim, self.use_mediators, cached)
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def _build_he_dict(self, batch) -> dict[int, list[int]]:
        edge_index = batch.edge_index
        num_nodes = batch.x.size(0)
        mask = edge_index[0] < num_nodes
        v2e_edges = edge_index[:, mask]
        node_ids = v2e_edges[0]
        he_ids = v2e_edges[1]
        he_id_min = he_ids.min().item() if he_ids.numel() > 0 else num_nodes

        He_dict: dict[int, list[int]] = {}
        for node, he in zip(node_ids.tolist(), he_ids.tolist()):
            normalized_id = he - he_id_min
            if normalized_id not in He_dict:
                He_dict[normalized_id] = []
            He_dict[normalized_id].append(node)
        return He_dict

    def get_embedding(self, batch):
        x = batch.x
        num_nodes = x.size(0)

        if hasattr(batch, "He_dict") and batch.He_dict is not None:
            He_dict = batch.He_dict
        else:
            He_dict = self._build_he_dict(batch)

        recompute = not self.fast
        for i, conv in enumerate(self.convs):
            x = conv(x, He_dict, num_nodes, recompute=(i == 0 or recompute))
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
