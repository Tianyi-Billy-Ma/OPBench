"""Self-loop transforms for graphs and hypergraphs."""

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops


class AddSelfLoop(BaseTransform):
    """Add self-loops to graph edge_index.

    For homogeneous graphs, adds self-loops to edge_index.
    For heterogeneous graphs, adds self-loops to edges where src_type == dst_type.
    """

    def __init__(self, fill_value: float = 1.0):
        """Initialize AddSelfLoop transform.

        Args:
            fill_value: Value for self-loop edge weights (default: 1.0).
        """
        super().__init__()
        self.fill_value = fill_value

    def forward(self, data: Data | HeteroData) -> Data | HeteroData:
        if isinstance(data, HeteroData):
            return self._add_hetero_self_loops(data)
        return self._add_homo_self_loops(data)

    def _add_homo_self_loops(self, data: Data) -> Data:
        """Add self-loops to homogeneous graph."""
        if data.edge_index is None or data.edge_index.numel() == 0:
            if hasattr(data, "num_nodes") and data.num_nodes is not None:
                num_nodes = data.num_nodes
            elif data.x is not None:
                num_nodes = data.x.size(0)
            else:
                return data

            self_loops = torch.arange(num_nodes, dtype=torch.long)
            data.edge_index = torch.stack([self_loops, self_loops], dim=0)
            return data

        num_nodes = data.num_nodes
        if num_nodes is None and data.x is not None:
            num_nodes = data.x.size(0)

        edge_index, edge_attr = add_self_loops(
            data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            fill_value=self.fill_value,
            num_nodes=num_nodes,
        )

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr

        return data

    def _add_hetero_self_loops(self, data: HeteroData) -> HeteroData:
        """Add self-loops to heterogeneous graph (same-type edges only)."""
        for edge_type in data.edge_types:
            src_type, rel_type, dst_type = edge_type
            if src_type != dst_type:
                continue

            edge_store = data[edge_type]
            edge_index = edge_store.edge_index

            num_nodes = data[src_type].num_nodes
            if num_nodes is None and hasattr(data[src_type], "x"):
                num_nodes = data[src_type].x.size(0)

            if num_nodes is None:
                continue

            edge_index, _ = add_self_loops(
                edge_index,
                num_nodes=num_nodes,
            )
            data[edge_type].edge_index = edge_index

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill_value={self.fill_value})"


class AddHypergraphSelfLoops(BaseTransform):
    """Add self-loops to hypergraph by creating individual hyperedges for each node.

    Each node gets its own hyperedge, creating a self-loop in the bipartite
    node-hyperedge representation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "num_nodes") or not isinstance(data.num_nodes, int):
            if hasattr(data, "edge_index") and data.edge_index is not None:
                num_nodes = int(data.edge_index[0].max().item()) + 1
            else:
                return data
        else:
            num_nodes = data.num_nodes

        if hasattr(data, "edge_index") and data.edge_index.numel() > 0:
            max_he_idx = int(data.edge_index[1].max().item())
        else:
            max_he_idx = num_nodes - 1

        self_loop_nodes = torch.arange(num_nodes, dtype=torch.long)
        self_loop_hes = torch.arange(
            max_he_idx + 1, max_he_idx + 1 + num_nodes, dtype=torch.long
        )

        # V->E only: give each node its own singleton hyperedge. This runs after
        # ExtractV2E, so edge_index must stay node->hyperedge (src < num_nodes);
        # adding the reverse direction here breaks that invariant for ConstructH
        # and the downstream hypergraph transforms/models.
        new_edges = torch.stack([self_loop_nodes, self_loop_hes], dim=0)

        if hasattr(data, "edge_index") and data.edge_index.numel() > 0:
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        else:
            data.edge_index = new_edges

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
