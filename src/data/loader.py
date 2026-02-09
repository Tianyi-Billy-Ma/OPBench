from collections.abc import Callable
from pathlib import Path
import logging
import pickle
import os

import numpy as np
import torch
from scipy import sparse as sp
from torch_geometric.data import Data, HeteroData

from scipy.sparse import load_npz

from src.data.metapaths import load_metapaths, add_precomputed_metapaths

_LOGGER = logging.getLogger(__name__)


DATASET_REGISTRY: dict[str, Callable[[Path], Data | HeteroData]] = {}


def register_loader(name: str):
    def decorator(fn: Callable[[Path], Data | HeteroData]):
        DATASET_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def build_dataset(name: str, data_dir: Path, **kwargs) -> Data | HeteroData:
    name_lower = name.lower().replace("-", "_")
    if name_lower not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    builder = DATASET_REGISTRY[name_lower]
    return builder(data_dir, **kwargs)


@register_loader("cora")
@register_loader("citeseer")
@register_loader("pubmed")
def citation_hypergraph_builder(data_dir: Path, **kwargs) -> Data:
    feature_path = data_dir / "features.pickle"
    label_path = data_dir / "labels.pickle"
    hypergraph_path = data_dir / "hypergraph.pickle"

    if not all(p.exists() for p in [feature_path, label_path, hypergraph_path]):
        raise FileNotFoundError(
            f"Missing HyperGCN assets in {data_dir}; expected "
            "'features.pickle', 'labels.pickle', 'hypergraph.pickle'."
        )

    with feature_path.open("rb") as f:
        features = pickle.load(f)
    if sp.issparse(features):
        features = features.todense()
    features = torch.as_tensor(np.asarray(features), dtype=torch.float32)

    with label_path.open("rb") as f:
        labels = pickle.load(f)
    labels = torch.as_tensor(labels, dtype=torch.long)

    with hypergraph_path.open("rb") as f:
        hypergraph = pickle.load(f)
    if not isinstance(hypergraph, dict):
        raise TypeError(f"Expected dict in hypergraph.pickle, got {type(hypergraph)}")

    num_nodes = int(features.size(0))
    num_classes = int(torch.unique(labels).numel())

    node_ids = []
    he_ids = []
    edge_idx = num_nodes
    for nodes in hypergraph.values():
        cur_nodes = list(map(int, nodes))
        node_ids.extend(cur_nodes)
        he_ids.extend([edge_idx] * len(cur_nodes))
        edge_idx += 1

    src = torch.as_tensor(node_ids + he_ids, dtype=torch.long)
    dst = torch.as_tensor(he_ids + node_ids, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_nodes = num_nodes
    data.num_hyperedges = torch.tensor([edge_idx - num_nodes], dtype=torch.long)
    data.num_classes = torch.tensor([num_classes], dtype=torch.long)
    return data


@register_loader("cora_graph")
def cora_graph_builder(data_dir: Path, **kwargs) -> Data:
    data = citation_hypergraph_builder(data_dir, **kwargs)

    _LOGGER.info("Converting hypergraph to graph via clique expansion...")

    num_nodes = data.x.shape[0]
    num_hyperedges = int(data.num_hyperedges.item())

    row, col = data.edge_index
    mask = (row < num_nodes) & (col >= num_nodes)
    node_indices = row[mask]
    he_indices = col[mask] - num_nodes

    values = torch.ones(node_indices.shape[0])
    H = torch.sparse_coo_tensor(
        torch.stack([node_indices, he_indices]), values, (num_nodes, num_hyperedges)
    )

    H_dense = H.to_dense()
    A = torch.matmul(H_dense, H_dense.t())
    A.fill_diagonal_(0)

    edge_index = A.nonzero().t()

    data.edge_index = edge_index
    if hasattr(data, "num_hyperedges"):
        del data.num_hyperedges
    data.edge_attr = None

    _LOGGER.info("Graph constructed with %d edges", edge_index.shape[1])
    return data


@register_loader("pdmp")
@register_loader("pdmp_opioid_detect")
def pdmp_builder(data_dir: Path, **kwargs) -> HeteroData:
    patient_parts = []
    for i in range(4):
        part_path = data_dir / f"patient_features_part{i}.npy"
        if part_path.exists():
            patient_parts.append(np.load(part_path))
    if patient_parts:
        patient_features = np.concatenate(patient_parts, axis=0)
    elif (data_dir / "patient_features.npy").exists():
        patient_features = np.load(data_dir / "patient_features.npy")
    else:
        raise FileNotFoundError(f"No patient features found in {data_dir}")

    prescriber_parts = []
    for i in range(2):
        part_path = data_dir / f"prescriber_features_part{i}.npy"
        if part_path.exists():
            prescriber_parts.append(np.load(part_path))
    if prescriber_parts:
        prescriber_features = np.concatenate(prescriber_parts, axis=0)
    else:
        prescriber_features = np.load(data_dir / "prescriber_features.npy")

    pharmacy_features = np.load(data_dir / "pharmacy_features.npy")
    drug_features = np.load(data_dir / "drug_features.npy")
    patient_labels = np.load(data_dir / "patient_labels.npy")

    edges_dir = data_dir / "edges"

    def load_edge(name: str) -> np.ndarray:
        path = edges_dir / f"{name}.npy"
        if path.exists():
            return np.load(path)
        raise FileNotFoundError(f"Edge file not found: {path}")

    p_d_edges = load_edge("patient_take_drug")
    p_ph_edges = load_edge("patient_pickup_at_pharmacy")
    p_pr_edges = load_edge("patient_visit_prescriber")
    pr_d_edges = load_edge("prescriber_prescribe_drug")
    ph_d_edges = load_edge("pharmacy_dispense_drug")

    data = HeteroData()

    data["patient"].x = torch.from_numpy(patient_features).float()
    data["patient"].y = torch.from_numpy(patient_labels).long()
    data["prescriber"].x = torch.from_numpy(prescriber_features).float()
    data["pharmacy"].x = torch.from_numpy(pharmacy_features).float()
    data["drug"].x = torch.from_numpy(drug_features).float()

    data["patient", "take", "drug"].edge_index = torch.from_numpy(p_d_edges).long()
    data["drug", "taken_by", "patient"].edge_index = torch.from_numpy(
        p_d_edges[::-1].copy()
    ).long()

    data["patient", "pickup_at", "pharmacy"].edge_index = torch.from_numpy(
        p_ph_edges
    ).long()
    data["pharmacy", "dispense_to", "patient"].edge_index = torch.from_numpy(
        p_ph_edges[::-1].copy()
    ).long()

    data["patient", "visit", "prescriber"].edge_index = torch.from_numpy(
        p_pr_edges
    ).long()
    data["prescriber", "visited_by", "patient"].edge_index = torch.from_numpy(
        p_pr_edges[::-1].copy()
    ).long()

    data["prescriber", "prescribe", "drug"].edge_index = torch.from_numpy(
        pr_d_edges
    ).long()
    data["drug", "prescribed_by", "prescriber"].edge_index = torch.from_numpy(
        pr_d_edges[::-1].copy()
    ).long()

    data["pharmacy", "dispense", "drug"].edge_index = torch.from_numpy(
        ph_d_edges
    ).long()
    data["drug", "dispensed_at", "pharmacy"].edge_index = torch.from_numpy(
        ph_d_edges[::-1].copy()
    ).long()

    _LOGGER.info(
        "Loaded PDMP: %d patients, %d prescribers, %d pharmacies, %d drugs, %d edge types",
        patient_features.shape[0],
        prescriber_features.shape[0],
        pharmacy_features.shape[0],
        drug_features.shape[0],
        len(data.edge_types),
    )

    metapaths = load_metapaths(data_dir)
    if metapaths:
        data = add_precomputed_metapaths(data, metapaths)
        _LOGGER.info("Loaded %d precomputed metapaths", len(metapaths))

    data.num_classes = 2

    return data


@register_loader("twitter_hydrug_role")
def twitter_hydrug_role_builder(data_dir: Path, **kwargs) -> Data:
    hyperedge_file_path = data_dir / "hyperedges.npz"
    emb_file_path = data_dir / "features.txt"
    label_file_path = data_dir / "labels.txt"

    H = load_npz(hyperedge_file_path).astype(np.int32).toarray().T
    X = np.loadtxt(emb_file_path)
    Y = np.loadtxt(label_file_path).astype(np.int32)
    Y = Y.argmax(1)

    X = torch.from_numpy(X).float()[:, 1:]
    Y = torch.from_numpy(Y).long()
    H = torch.from_numpy(H).long()
    num_nodes, num_hyperedges = H.shape

    node_idx, edge_idx = H.nonzero(as_tuple=True)
    hub_idx = edge_idx + num_nodes

    edge_index = torch.stack([node_idx, hub_idx], dim=0).long()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=X, edge_index=edge_index, y=Y)
    data.num_nodes = num_nodes
    data.num_hyperedges = torch.tensor([num_hyperedges], dtype=torch.long)
    data.num_classes = torch.tensor([int(Y.max().item()) + 1], dtype=torch.long)
    return data


@register_loader("twitter_hydrug_comm")
def twitter_hydrug_comm_builder(data_dir: Path, **kwargs) -> Data:
    """Load Twitter HyDrug Community dataset.

    Note: Unlike twitter_hydrug_role which uses single-label classification,
    this dataset uses multi-label classification where nodes can belong to
    multiple communities. Labels are kept as 2D tensors for BCE loss.
    """
    hyperedge_file_path = data_dir / "hyperedges.npz"
    emb_file_path = data_dir / "features.txt"
    label_file_path = data_dir / "labels.txt"

    H = load_npz(hyperedge_file_path).astype(np.int32).toarray().T
    X = np.loadtxt(emb_file_path)
    Y = np.loadtxt(label_file_path).astype(np.int32)

    X = torch.from_numpy(X).float()[:, 1:]
    Y = torch.from_numpy(Y)  # Keep as float for BCE loss if 2D
    H = torch.from_numpy(H).long()

    num_nodes, num_hyperedges = H.shape

    node_idx, edge_idx = H.nonzero(as_tuple=True)
    hub_idx = edge_idx + num_nodes

    edge_index = torch.stack([node_idx, hub_idx], dim=0).long()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Multi-label: Y is 2D [num_nodes, num_classes], keep as float for BCE
    # Single-label: Y is 1D [num_nodes], convert to long for CrossEntropy
    if Y.dim() == 2:
        Y = Y.float()  # BCE loss expects float targets
        num_classes = Y.size(1)
    else:
        Y = Y.long()
        num_classes = int(Y.max().item()) + 1

    data = Data(x=X, edge_index=edge_index, y=Y)
    data.num_nodes = num_nodes
    data.num_hyperedges = torch.tensor([num_hyperedges], dtype=torch.long)
    data.num_classes = torch.tensor([num_classes], dtype=torch.long)
    return data


@register_loader("nhance")
@register_loader("nhance_diet_role")
def nhance_builder(data_dir: Path, **kwargs) -> HeteroData:
    graph_path = data_dir / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"NHANCE graph file not found: {graph_path}")

    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(data)}")

    _LOGGER.info(
        "Loaded NHANCE: %d users, %d foods, %d ingredients, %d categories, %d habits, %d edge types",
        data["user"].x.size(0),
        data["food"].x.size(0),
        data["ingredient"].x.size(0),
        data["category"].x.size(0),
        data["habit"].x.size(0),
        len(data.edge_types),
    )

    data.num_classes = 2

    return data


@register_loader("twitter_mrdrug_role")
def twitter_mrdrug_role_builder(data_dir: Path, **kwargs) -> Data:
    """Build Twitter MRDrug Role dataset (multi-relation graph).

    This dataset contains Twitter users with 3 relation types:
    - net_uku: user-keyword-user (shared keywords)
    - net_ufu: user-follow-user (follow relationships)
    - net_utu: user-tweet-user (shared tweets/retweets)

    Args:
        data_dir: Path to raw data directory containing .npy files.

    Returns:
        PyG Data object with edge_type for multi-relation edges.
    """
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"
    edges_path = data_dir / "edges.npy"
    edge_types_path = data_dir / "edge_types.npy"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found at {features_path}. "
            "Run the .mat conversion script first."
        )

    X = np.load(features_path)
    Y = np.load(labels_path)

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    edges = np.load(edges_path)
    edge_index = torch.from_numpy(edges).long()

    edge_type = None
    if edge_types_path.exists():
        edge_types = np.load(edge_types_path)
        edge_type = torch.from_numpy(edge_types).long()

    data = Data(x=X, edge_index=edge_index, y=Y)
    data.num_nodes = X.size(0)
    data.num_classes = torch.tensor([int(Y.max().item()) + 1], dtype=torch.long)

    if edge_type is not None:
        data.edge_type = edge_type
        data.num_relations = torch.tensor(
            [int(edge_type.max().item()) + 1], dtype=torch.long
        )

    _LOGGER.info(
        "Loaded Twitter MRDrug Role: %d users, %d features, %d classes, %d edges, %d relations",
        data.num_nodes,
        X.size(1),
        int(data.num_classes.item()),
        edge_index.size(1),
        int(data.num_relations.item()) if edge_type is not None else 1,
    )

    return data
