"""Transform to convert HeteroData to homogeneous Data using metapaths."""

import logging

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce

logger = logging.getLogger(__name__)


class HetToHomo(BaseTransform):
    """Convert HeteroData to homogeneous Data using metapath edges.

    This transform extracts only metapath edges (target-target relations),
    preferred for running homogeneous GNNs on heterogeneous graphs.

    Requirements:
    - Target node type must have labels (y attribute) for auto-inference
    - HeteroData must have precomputed metapath edges

    The target node type is automatically inferred from the node type
    that contains labels (y attribute).

    Example:
        >>> transform = HetToHomo()
        >>> homo_data = transform(hetero_data)
    """

    def __init__(self, metapath_prefix: str = "metapath_"):
        """Initialize HetToHomo transform.

        Args:
            metapath_prefix: Prefix for metapath edge type names.
        """
        super().__init__()
        self.metapath_prefix = metapath_prefix

    def forward(self, data: HeteroData) -> Data:
        if not isinstance(data, HeteroData):
            return data

        target = self._infer_target_node_type(data)
        return self._convert_metapath(data, target)

    def _convert_metapath(self, data: HeteroData, target: str) -> Data:
        """Convert using only metapath edges."""
        # Validate target node type exists
        if target not in data.node_types:
            raise ValueError(
                f"Target node type '{target}' not found in data. "
                f"Available: {data.node_types}"
            )

        # Extract features from target node type
        target_store = data[target]
        x = target_store.x
        y = target_store.y if hasattr(target_store, "y") else None

        num_nodes = x.size(0) if x is not None else target_store.num_nodes
        if num_nodes is None:
            raise ValueError(
                f"Cannot determine num_nodes for target '{target}'. "
                "Either 'x' or 'num_nodes' must be set on the target node store."
            )

        # Collect metapath edges
        all_edges = []
        all_edge_types = []
        edge_type_names = []

        for edge_type_key in data.edge_types:
            src_type, rel_type, dst_type = edge_type_key

            # Only include edges where:
            # 1. Both src and dst are target node type
            # 2. Relation is a metapath (starts with prefix)
            if src_type != target or dst_type != target:
                continue
            if not rel_type.startswith(self.metapath_prefix):
                continue

            edge_index = data[edge_type_key].edge_index
            if edge_index.numel() == 0:
                continue

            all_edges.append(edge_index)
            all_edge_types.append(
                torch.full(
                    (edge_index.size(1),),
                    len(edge_type_names),
                    dtype=torch.long,
                    device=edge_index.device,
                )
            )
            edge_type_names.append(rel_type)

        # Handle case where no metapath edges found
        if not all_edges:
            logger.warning(
                f"No metapath edges found for target '{target}'. "
                f"Edge types: {[et[1] for et in data.edge_types]}. "
                f"Looking for prefix: '{self.metapath_prefix}'"
            )
            device = x.device if x is not None else torch.device("cpu")
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)
        else:
            # Concatenate all metapath edges
            edge_index = torch.cat(all_edges, dim=1)
            edge_type = torch.cat(all_edge_types, dim=0)

            # Coalesce to remove duplicates
            edge_index, edge_type = coalesce(
                edge_index,
                edge_type,
                num_nodes=num_nodes,
            )

        # Create homogeneous Data object
        homo_data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
        )

        # Copy masks if present
        for attr in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(target_store, attr) and getattr(target_store, attr) is not None:
                setattr(homo_data, attr, getattr(target_store, attr))

        # Store metadata
        homo_data.target_node_type = target
        homo_data.metapath_edge_types = edge_type_names
        homo_data.num_relations = len(edge_type_names) if edge_type_names else 1

        if hasattr(data, "num_classes"):
            homo_data.num_classes = data.num_classes

        logger.info(
            f"HetToHomo: {num_nodes} nodes, "
            f"{edge_index.size(1)} edges from {len(edge_type_names)} metapaths"
        )

        return homo_data

    def _infer_target_node_type(self, data: HeteroData) -> str:
        """Infer target node type from labels."""
        for node_type in data.node_types:
            if hasattr(data[node_type], "y") and data[node_type].y is not None:
                return node_type

        raise ValueError(
            f"Could not infer target node type. No node type has labels. "
            f"Available node types: {data.node_types}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metapath_prefix={self.metapath_prefix!r})"
