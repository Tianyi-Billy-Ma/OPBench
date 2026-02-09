import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseBackbone
from src.hparams import FullArguments
from src.models.registry import ModelRegistry


class HGNNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        if G.is_sparse:
            x = torch.sparse.mm(G, x)
        else:
            x = torch.matmul(G, x)
        return x


@ModelRegistry.register("hgnn")
class HGNN(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        self.dropout = config.model.dropout

        assert output_dim is not None, "output_dim must be set"

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(HGNNConv(num_features, output_dim))
        else:
            self.convs.append(HGNNConv(num_features, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(HGNNConv(hidden_dim, hidden_dim))
            self.convs.append(HGNNConv(hidden_dim, output_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def _compute_G(self, batch) -> torch.Tensor:
        edge_index = batch.edge_index
        num_nodes = batch.x.size(0)
        device = edge_index.device

        mask = edge_index[0] < num_nodes
        v2e_edges = edge_index[:, mask]
        node_ids = v2e_edges[0]
        he_ids = v2e_edges[1]

        he_id_min = he_ids.min().item() if he_ids.numel() > 0 else num_nodes
        he_ids_normalized = he_ids - he_id_min
        num_hyperedges = (
            he_ids_normalized.max().item() + 1 if he_ids_normalized.numel() > 0 else 0
        )

        H_indices = torch.stack([node_ids, he_ids_normalized], dim=0)
        H_values = torch.ones(H_indices.size(1), device=device)
        H = torch.sparse_coo_tensor(
            H_indices, H_values, (num_nodes, num_hyperedges)
        ).coalesce()

        H_dense = H.to_dense()

        d_v = H_dense.sum(dim=1).clamp(min=1e-12)
        d_v_inv_sqrt = d_v.pow(-0.5)

        d_e = H_dense.sum(dim=0).clamp(min=1e-12)
        d_e_inv = d_e.pow(-1)

        H_scaled = d_v_inv_sqrt.unsqueeze(1) * H_dense
        H_scaled = H_scaled * d_e_inv.unsqueeze(0)

        H_T_scaled = H_dense.T * d_v_inv_sqrt.unsqueeze(0)
        G = H_scaled @ H_T_scaled

        return G

    def get_embedding(self, batch):
        x = batch.x

        if hasattr(batch, "G") and batch.G is not None:
            G = batch.G
            if G.is_sparse:
                G = G.coalesce()
        else:
            G = self._compute_G(batch)

        for conv in self.convs[:-1]:
            x = conv(x, G)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, G)
        return x

    def forward(self, batch) -> dict:
        h = self.get_embedding(batch)
        return {"embeddings": h}
