import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax
from src.hparams import FullArguments
from src.models.registry import ModelRegistry
from src.models.base import BaseBackbone


class HCHAConv(nn.Module):
    """Hypergraph Convolution and Hypergraph Attention (HCHA) layer.

    Implemented with two-stage message passing (V->E and E->V) using scatter operations.
    Supports symmetric and asymmetric degree normalization and optional attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        symdegnorm: bool = False,
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.symdegnorm = symdegnorm
        self.use_attention = use_attention
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, out_channels)

        if self.use_attention:
            self.att = nn.Parameter(torch.Tensor(1, out_channels))
        else:
            self.register_parameter("att", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.att is not None:
            nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        # Handle bipartite edge_index: nodes [0, N), hyperedges [N, N+M)
        # Filter for node-to-hyperedge edges
        mask = edge_index[0] < num_nodes
        v_idx = edge_index[0, mask]
        e_idx_raw = edge_index[1, mask]

        if e_idx_raw.numel() > 0:
            he_id_min = int(e_idx_raw.min().item())
            e_idx = e_idx_raw - he_id_min
            num_hyperedges = int(e_idx.max().item() + 1)
        else:
            e_idx = e_idx_raw
            num_hyperedges = 0

        x = self.lin(x)

        # Node degrees D: number of hyperedges each node belongs to
        d_v = scatter(
            torch.ones(v_idx.size(0), device=x.device),
            v_idx,
            dim=0,
            dim_size=int(num_nodes),
            reduce="sum",
        )
        # Hyperedge degrees B: number of nodes in each hyperedge
        d_e = scatter(
            torch.ones(e_idx.size(0), device=x.device),
            e_idx,
            dim=0,
            dim_size=int(num_hyperedges),
            reduce="sum",
        )

        if self.use_attention:
            # Attention weight alpha_ve: softmax over nodes within each hyperedge
            alpha = (x[v_idx] * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = softmax(alpha, e_idx, num_nodes=int(num_hyperedges))
        else:
            alpha = None

        if self.symdegnorm:
            # Symmetric normalization: D^{-1/2} H W B^{-1} H^T D^{-1/2}
            d_v_inv_sqrt = d_v.pow(-0.5)
            d_v_inv_sqrt.masked_fill_(d_v_inv_sqrt == float("inf"), 0)

            # Stage 1: V -> E
            # h_e = B^{-1} H^T (D^{-1/2} X) or attentional sum
            x_scaled = x * d_v_inv_sqrt.view(-1, 1)
            if self.use_attention and alpha is not None:
                h_e = scatter(
                    x_scaled[v_idx] * alpha.view(-1, 1),
                    e_idx,
                    dim=0,
                    dim_size=int(num_hyperedges),
                    reduce="sum",
                )
            else:
                h_e = scatter(
                    x_scaled[v_idx],
                    e_idx,
                    dim=0,
                    dim_size=int(num_hyperedges),
                    reduce="sum",
                )
                d_e_inv = d_e.pow(-1)
                d_e_inv.masked_fill_(d_e_inv == float("inf"), 0)
                h_e = h_e * d_e_inv.view(-1, 1)

            # Stage 2: E -> V
            # out = D^{-1/2} H h_e
            out = scatter(
                h_e[e_idx], v_idx, dim=0, dim_size=int(num_nodes), reduce="sum"
            )
            out = out * d_v_inv_sqrt.view(-1, 1)
        else:
            # Asymmetric normalization: D^{-1} H W B^{-1} H^T
            d_v_inv = d_v.pow(-1)
            d_v_inv.masked_fill_(d_v_inv == float("inf"), 0)

            # Stage 1: V -> E
            if self.use_attention and alpha is not None:
                h_e = scatter(
                    x[v_idx] * alpha.view(-1, 1),
                    e_idx,
                    dim=0,
                    dim_size=int(num_hyperedges),
                    reduce="sum",
                )
            else:
                h_e = scatter(
                    x[v_idx], e_idx, dim=0, dim_size=int(num_hyperedges), reduce="sum"
                )
                d_e_inv = d_e.pow(-1)
                d_e_inv.masked_fill_(d_e_inv == float("inf"), 0)
                h_e = h_e * d_e_inv.view(-1, 1)

            # Stage 2: E -> V
            out = scatter(
                h_e[e_idx], v_idx, dim=0, dim_size=int(num_nodes), reduce="sum"
            )
            out = out * d_v_inv.view(-1, 1)

        return out


@ModelRegistry.register("hcha")
class HCHA(BaseBackbone):
    """HCHA Backbone implementation."""

    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        self.dropout = config.model.dropout

        assert output_dim is not None, "output_dim must be set"

        symdegnorm = config.model.hcha_symdegnorm
        use_attention = config.model.hcha_use_attention

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                HCHAConv(
                    num_features, output_dim, symdegnorm, use_attention, self.dropout
                )
            )
        else:
            self.convs.append(
                HCHAConv(
                    num_features, hidden_dim, symdegnorm, use_attention, self.dropout
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    HCHAConv(
                        hidden_dim, hidden_dim, symdegnorm, use_attention, self.dropout
                    )
                )
            self.convs.append(
                HCHAConv(
                    hidden_dim, output_dim, symdegnorm, use_attention, self.dropout
                )
            )

    def reset_parameters(self) -> None:
        for conv in self.convs:
            if isinstance(conv, HCHAConv):
                conv.reset_parameters()

    def get_embedding(self, batch) -> torch.Tensor:
        x = batch.x
        edge_index = batch.edge_index

        for i, conv in enumerate(self.convs):
            if isinstance(conv, HCHAConv):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, batch) -> dict:
        h = self.get_embedding(batch)
        return {"embeddings": h}
