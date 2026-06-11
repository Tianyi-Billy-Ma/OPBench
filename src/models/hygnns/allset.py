from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax

from src.hparams import FullArguments
from src.models.base import BaseBackbone
from src.models.mlp import MLP
from src.models.registry import ModelRegistry


class AllSetConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        normalization: Literal["bn", "ln", "none"] = "bn",
        input_norm: bool = False,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__(node_dim=0)
        self.use_attention = use_attention
        self.dropout = dropout
        self._aggr = "add"

        if use_attention:
            self.heads = num_heads
            self.head_dim = hid_dim // num_heads
            self.lin_k = nn.Linear(in_dim, hid_dim)
            self.lin_v = nn.Linear(in_dim, hid_dim)
            self.att_r = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
            self.ln0 = nn.LayerNorm(hid_dim)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.ff = MLP(
                hid_dim,
                hid_dim,
                out_dim,
                num_layers,
                dropout=dropout,
                normalization=normalization,
            )
            nn.init.xavier_uniform_(self.att_r)
        else:
            self.encoder = MLP(
                in_dim,
                hid_dim,
                hid_dim,
                num_layers,
                dropout=dropout,
                normalization=normalization,
                input_norm=input_norm,
            )
            self.decoder = MLP(
                hid_dim,
                hid_dim,
                out_dim,
                num_layers,
                dropout=dropout,
                normalization=normalization,
                input_norm=input_norm,
            )

    def reset_parameters(self) -> None:
        if self.use_attention:
            nn.init.xavier_uniform_(self.att_r)
            self.lin_k.reset_parameters()
            self.lin_v.reset_parameters()
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
            self.ff.reset_parameters()
        else:
            self.encoder.reset_parameters()
            self.decoder.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        norm: Optional[torch.Tensor] = None,
        aggr: str = "add",
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self._aggr = aggr

        if self.use_attention:
            H, C = self.heads, self.head_dim

            x_k = self.lin_k(x).view(-1, H, C)
            x_v = self.lin_v(x).view(-1, H, C)
            alpha = (x_k * self.att_r).sum(dim=-1)

            out = self.propagate(edge_index, x=x_v, alpha=alpha, norm=norm)

            out = out + self.att_r
            out = self.ln0(out.view(-1, H * C))
            out = self.ln1(out + F.relu(self.ff(out)))
        else:
            x = F.relu(self.encoder(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            out = self.propagate(edge_index, x=x, norm=norm)
            out = F.relu(self.decoder(out))

        return out

    def message(
        self,
        x_j: torch.Tensor,
        alpha_j: Optional[torch.Tensor] = None,
        norm: Optional[torch.Tensor] = None,
        index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        msg = x_j

        if alpha_j is not None and index is not None:
            alpha_j = F.leaky_relu(alpha_j, 0.2)
            alpha_j = softmax(alpha_j, index, num_nodes=int(index.max().item()) + 1)
            msg = msg * alpha_j.unsqueeze(-1)

        if norm is not None:
            if msg.dim() == 3:
                msg = msg * norm.view(-1, 1, 1)
            else:
                msg = msg * norm.view(-1, 1)

        return msg

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        return scatter(
            inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self._aggr
        )


@ModelRegistry.register("allset")
class AllSet(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        num_layers = config.model.num_layers
        mlp_layers = config.model.allset_num_layers
        dropout = config.model.dropout
        use_pma = config.model.allset_pma
        num_heads = config.model.allset_num_heads
        self.aggr = config.model.aggregate
        self.use_gpr = config.model.allset_gpr
        self.use_learn_mask = config.model.allset_learn_mask

        self.v2e_convs = nn.ModuleList()
        self.e2v_convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            self.v2e_convs.append(
                AllSetConv(
                    in_dim,
                    hidden_dim,
                    hidden_dim,
                    mlp_layers,
                    dropout,
                    "ln",
                    config.model.allset_input_norm,
                    use_pma,
                    num_heads,
                )
            )
            self.e2v_convs.append(
                AllSetConv(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    mlp_layers,
                    dropout,
                    "ln",
                    config.model.allset_input_norm,
                    use_pma,
                    num_heads,
                )
            )

        if self.use_gpr:
            self.gpr_mlp = MLP(
                num_features, hidden_dim, hidden_dim, mlp_layers, dropout, "ln"
            )
            self.gpr_weights = nn.Linear(num_layers + 1, 1, bias=False)
        else:
            self.gpr_mlp = None
            self.gpr_weights = None

        self.learn_mask_weight = None
        self.dropout = dropout
        self.input_dropout = config.model.input_dropout

    def reset_parameters(self) -> None:
        for i in range(len(self.v2e_convs)):
            self.v2e_convs[i].reset_parameters()
        for i in range(len(self.e2v_convs)):
            self.e2v_convs[i].reset_parameters()
        if self.use_gpr:
            if self.gpr_mlp is not None:
                self.gpr_mlp.reset_parameters()
            if self.gpr_weights is not None:
                self.gpr_weights.reset_parameters()
        if self.learn_mask_weight is not None:
            nn.init.ones_(self.learn_mask_weight)

    def get_embedding(self, batch):
        x = batch.x
        num_nodes = x.size(0)

        if hasattr(batch, "v2e_edge_index") and batch.v2e_edge_index is not None:
            edge_index = batch.v2e_edge_index.clone()
        else:
            full_edge_index = batch.edge_index
            mask = full_edge_index[0] < num_nodes
            edge_index = full_edge_index[:, mask].clone()

        if hasattr(batch, "norm") and batch.norm is not None:
            norm = batch.norm.clone()
        else:
            norm = torch.ones(edge_index.size(1), device=x.device)

        if self.use_learn_mask:
            if self.learn_mask_weight is None or self.learn_mask_weight.size(
                0
            ) != norm.size(0):
                self.learn_mask_weight = nn.Parameter(
                    torch.ones(norm.size(0), device=x.device)
                )
            norm = norm * self.learn_mask_weight

        cidx = edge_index[1].min() if edge_index.numel() > 0 else 0
        edge_index[1] = edge_index[1] - cidx
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        x = F.dropout(x, p=self.input_dropout, training=self.training)

        if self.use_gpr and self.gpr_mlp is not None and self.gpr_weights is not None:
            xs = [F.relu(self.gpr_mlp(x))]
            for i in range(len(self.v2e_convs)):
                x_e = self.v2e_convs[i](x, edge_index, norm=norm, aggr=self.aggr)
                x_e = F.relu(x_e)
                x_e = F.dropout(x_e, p=self.dropout, training=self.training)

                x = self.e2v_convs[i](
                    x_e, reversed_edge_index, norm=norm, aggr=self.aggr
                )
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = torch.stack(xs, dim=-1)
            x = self.gpr_weights(x).squeeze(-1)
        else:
            for i in range(len(self.v2e_convs)):
                x_e = self.v2e_convs[i](x, edge_index, norm=norm, aggr=self.aggr)
                x_e = F.relu(x_e)
                x_e = F.dropout(x_e, p=self.dropout, training=self.training)

                x = self.e2v_convs[i](
                    x_e, reversed_edge_index, norm=norm, aggr=self.aggr
                )
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
