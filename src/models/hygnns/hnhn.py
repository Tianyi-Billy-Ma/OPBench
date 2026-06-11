"""HNHN (Hypergraph Networks with Hyperedge Neurons) implementation.

Reference: Dong et al., "HNHN: Hypergraph Networks with Hyperedge Neurons",
ICML 2020 Graph Representation Learning Workshop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from src.hparams import FullArguments
from src.models.registry import ModelRegistry

from ..base import BaseBackbone


class HNHNConv(nn.Module):
    """HNHN convolution layer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        nonlinear_inbetween: bool = True,
    ):
        """Initialize HNHN convolution.

        Args:
            in_channels: Input dimension.
            hidden_channels: Hidden dimension for V2E transformation.
            out_channels: Output dimension.
            nonlinear_inbetween: Whether to apply nonlinearity between V2E and E2V.
        """
        super().__init__()
        self.weight_v2e = nn.Linear(in_channels, hidden_channels)
        self.weight_e2v = nn.Linear(hidden_channels, out_channels)
        self.nonlinear_inbetween = nonlinear_inbetween

    def reset_parameters(self) -> None:
        """Reset parameters."""
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        d_v_beta: torch.Tensor,
        d_e_beta_inv: torch.Tensor,
        d_e_alpha: torch.Tensor,
        d_v_alpha_inv: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features.
            edge_index: Bipartite edge index [2, E] where first row is source,
                       second row is target. Nodes in [0, num_nodes),
                       hyperedges in [num_nodes, num_nodes + num_edges).
            num_nodes: Number of nodes.
            num_edges: Number of hyperedges.
            d_v_beta: Node degree^beta normalization [num_nodes].
            d_e_beta_inv: Hyperedge degree^(-beta) normalization [num_edges].
            d_e_alpha: Hyperedge degree^alpha normalization [num_edges].
            d_v_alpha_inv: Node degree^(-alpha) normalization [num_nodes].

        Returns:
            Updated node features [num_nodes, out_channels].
        """
        # V -> E aggregation
        x = self.weight_v2e(x)
        x = d_v_beta.unsqueeze(-1) * x

        # Scatter from nodes to hyperedges
        node_idx = edge_index[0]
        he_idx = edge_index[1]
        mask = node_idx < num_nodes
        x_e = scatter(
            x[node_idx[mask]],
            he_idx[mask] - num_nodes,
            dim=0,
            dim_size=num_edges,
            reduce="mean",
        )
        x_e = d_e_beta_inv.unsqueeze(-1) * x_e

        if self.nonlinear_inbetween:
            x_e = F.relu(x_e)

        # E -> V aggregation
        x_e = self.weight_e2v(x_e)
        x_e = d_e_alpha.unsqueeze(-1) * x_e

        # Scatter from hyperedges to nodes
        mask = he_idx >= num_nodes
        x_v = scatter(
            x_e[he_idx[mask] - num_nodes],
            node_idx[mask],
            dim=0,
            dim_size=num_nodes,
            reduce="mean",
        )
        x_v = d_v_alpha_inv.unsqueeze(-1) * x_v

        return x_v


@ModelRegistry.register("hnhn")
class HNHN(BaseBackbone):
    """HNHN encoder for hypergraph.

    Uses degree-based normalizations with alpha/beta parameters
    to control the influence of node/hyperedge degrees.
    """

    def __init__(self, config: FullArguments):
        """Initialize HNHN encoder.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        self.dropout = config.model.dropout
        nonlinear = config.model.hnhn_nonlinear_inbetween
        self.alpha = config.model.hnhn_alpha
        self.beta = config.model.hnhn_beta

        assert output_dim is not None, "output_dim must be set"

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(HNHNConv(num_features, hidden_dim, output_dim, nonlinear))
        else:
            # First layer
            self.convs.append(HNHNConv(num_features, hidden_dim, hidden_dim, nonlinear))
            # Middle layers
            for _ in range(num_layers - 2):
                self.convs.append(
                    HNHNConv(hidden_dim, hidden_dim, hidden_dim, nonlinear)
                )
            # Last layer
            self.convs.append(HNHNConv(hidden_dim, hidden_dim, output_dim, nonlinear))

    def reset_parameters(self) -> None:
        """Reset all parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def _compute_normalizations(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Compute HNHN degree-based normalizations.

        Args:
            batch: PyG Data object.

        Returns:
            Tuple of (d_v_beta, d_e_beta_inv, d_e_alpha, d_v_alpha_inv, num_edges).
        """
        edge_index = batch.edge_index
        num_nodes = batch.x.size(0)
        device = edge_index.device

        node_idx = edge_index[0]
        he_idx = edge_index[1]

        # Get number of hyperedges
        num_edges = int((he_idx.max() - num_nodes + 1).item())

        # Node degree (count edges where node is source)
        mask = node_idx < num_nodes
        d_v = scatter(
            torch.ones(mask.sum(), device=device),
            node_idx[mask],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        ).clamp(min=1)

        # Hyperedge size (count edges where hyperedge is target)
        mask = he_idx >= num_nodes
        d_e = scatter(
            torch.ones(mask.sum(), device=device),
            he_idx[mask] - num_nodes,
            dim=0,
            dim_size=num_edges,
            reduce="sum",
        ).clamp(min=1)

        # Compute normalizations
        d_v_beta = d_v.pow(self.beta)
        d_e_beta_inv = d_e.pow(-self.beta)
        d_e_alpha = d_e.pow(self.alpha)
        d_v_alpha_inv = d_v.pow(-self.alpha)

        return d_v_beta, d_e_beta_inv, d_e_alpha, d_v_alpha_inv, num_edges

    def get_embedding(self, batch) -> torch.Tensor:
        """Get node embeddings.

        Args:
            batch: PyG Data object with hypergraph structure.

        Returns:
            Node embeddings tensor of shape [num_nodes, output_dim].
        """
        x = batch.x
        edge_index = batch.edge_index
        num_nodes = x.size(0)

        # Use precomputed normalizations if available, otherwise compute inline
        if (
            hasattr(batch, "D_v_beta")
            and hasattr(batch, "D_e_beta_inv")
            and hasattr(batch, "D_e_alpha")
            and hasattr(batch, "D_v_alpha_inv")
        ):
            d_v_beta = batch.D_v_beta
            d_e_beta_inv = batch.D_e_beta_inv
            d_e_alpha = batch.D_e_alpha
            d_v_alpha_inv = batch.D_v_alpha_inv
            num_edges = (
                batch.num_hyperedges_hnhn
                if hasattr(batch, "num_hyperedges_hnhn")
                else len(d_e_alpha)
            )
        else:
            d_v_beta, d_e_beta_inv, d_e_alpha, d_v_alpha_inv, num_edges = (
                self._compute_normalizations(batch)
            )

        for i, conv in enumerate(self.convs):
            x = conv(
                x,
                edge_index,
                num_nodes,
                num_edges,
                d_v_beta,
                d_e_beta_inv,
                d_e_alpha,
                d_v_alpha_inv,
            )
            # Apply activation and dropout except for last layer
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
