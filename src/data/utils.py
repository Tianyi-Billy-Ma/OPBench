import logging
import warnings
import torch
from torch_geometric.data import Data, HeteroData
from .datamodule import HONADataModule

logger = logging.getLogger(__name__)


def extract_data_info(dm: HONADataModule, config) -> dict:
    """Extract runtime data information for config patching."""
    info = {}
    dataset_name = config.data.dataset.lower()
    model_name = config.model.model_name.lower()

    # Hardcoded or dynamic logic depending on dataset
    # This logic was previously in main.py
    if dataset_name == "cora":
        info["data.num_features"] = dm.data.x.shape[1]
        info["data.num_classes"] = len(torch.unique(dm.data.y))
        info["data.target_node_type"] = None
    elif dataset_name == "pdmp":
        info["data.num_features"] = 768
        info["data.num_classes"] = 2
        info["data.target_node_type"] = "patient"
    elif dataset_name == "nhance":
        info["data.num_features"] = dm.data["user"].x.shape[1]
        info["data.num_classes"] = 2
        info["data.target_node_type"] = "user"

    # Extract metadata for heterogeneous GNNs
    if model_name in ("hgmae", "hgt", "han"):
        info["data.metadata"] = dm.data.metadata()

    # Extract num_relations for RGCN
    if model_name == "rgcn":
        if isinstance(dm.data, HeteroData):
            target_node_type = info.get("data.target_node_type")
            if target_node_type:
                num_relations = sum(
                    1
                    for src, _, dst in dm.data.edge_types
                    if src == target_node_type and dst == target_node_type
                )
                info["model.num_relations"] = max(num_relations, 1)
            else:
                info["model.num_relations"] = len(dm.data.edge_types)
        elif hasattr(dm.data, "edge_type") and dm.data.edge_type is not None:
            info["model.num_relations"] = int(dm.data.edge_type.max().item()) + 1
        else:
            logger.warning(
                "RGCN requires edge_type attribute; num_relations must be set manually"
            )

    return info


def hetero_to_homo(
    hetero_data: HeteroData,
    target_node_type: str,
) -> Data:
    """DEPRECATED: Use ToMetapathHomogeneous transform instead.

    This function runs at training time which adds overhead.
    Use pre_transform with ToMetapathHomogeneous for preprocessing.
    """
    warnings.warn(
        "hetero_to_homo() is deprecated. Use ToMetapathHomogeneous transform "
        "via get_pre_transform() for preprocessing instead of runtime conversion.",
        DeprecationWarning,
        stacklevel=2,
    )
    target_store = hetero_data[target_node_type]
    device = target_store.x.device

    homo_data = Data(
        x=target_store.x,
        y=target_store.y if hasattr(target_store, "y") else None,
    )

    if hasattr(target_store, "train_mask"):
        homo_data.train_mask = target_store.train_mask
    if hasattr(target_store, "val_mask"):
        homo_data.val_mask = target_store.val_mask
    if hasattr(target_store, "test_mask"):
        homo_data.test_mask = target_store.test_mask

    all_edges = []
    all_edge_types = []
    edge_type_mapping = {}

    for edge_type_idx, edge_type_key in enumerate(hetero_data.edge_types):
        src_type, rel_type, dst_type = edge_type_key

        if src_type != target_node_type or dst_type != target_node_type:
            continue

        edge_index = hetero_data[edge_type_key].edge_index
        all_edges.append(edge_index)
        all_edge_types.append(
            torch.full(
                (edge_index.size(1),), edge_type_idx, dtype=torch.long, device=device
            )
        )
        edge_type_mapping[edge_type_idx] = edge_type_key

    if all_edges:
        homo_data.edge_index = torch.cat(all_edges, dim=1)
        homo_data.edge_type = torch.cat(all_edge_types, dim=0)
    else:
        homo_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        homo_data.edge_type = torch.empty((0,), dtype=torch.long, device=device)

    homo_data.num_relations = len(edge_type_mapping) if edge_type_mapping else 1

    return homo_data
