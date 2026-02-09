"""Metapath definitions and utilities for heterogeneous graphs.

Provides metapath definitions compatible with PyG's AddMetaPaths transform
and IO utilities for saving/loading precomputed metapaths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

EdgeType = tuple[str, str, str]
Metapath = list[EdgeType]


# Metapath definitions for PDMP dataset.
# Format compatible with PyG's AddMetaPaths transform.
PDMP_METAPATHS: dict[str, Metapath] = {
    "p_d_p": [
        ("patient", "take", "drug"),
        ("drug", "taken_by", "patient"),
    ],
    "p_ph_p": [
        ("patient", "pickup_at", "pharmacy"),
        ("pharmacy", "dispense_to", "patient"),
    ],
    "p_pr_p": [
        ("patient", "visit", "prescriber"),
        ("prescriber", "visited_by", "patient"),
    ],
    "p_pr_d_pr_p": [
        ("patient", "visit", "prescriber"),
        ("prescriber", "prescribe", "drug"),
        ("drug", "prescribed_by", "prescriber"),
        ("prescriber", "visited_by", "patient"),
    ],
    "p_ph_d_ph_p": [
        ("patient", "pickup_at", "pharmacy"),
        ("pharmacy", "dispense", "drug"),
        ("drug", "dispensed_at", "pharmacy"),
        ("pharmacy", "dispense_to", "patient"),
    ],
}


def get_metapath_names() -> list[str]:
    return list(PDMP_METAPATHS.keys())


def save_metapaths(metapaths: dict[str, np.ndarray], output_dir: str | Path) -> None:
    """Save computed metapath edge indices to disk.

    Args:
        metapaths: Dict mapping metapath names to edge_index arrays (2, num_edges).
        output_dir: Directory to save metapath files.
    """
    output_dir = Path(output_dir)
    metapaths_dir = output_dir / "metapaths"
    metapaths_dir.mkdir(parents=True, exist_ok=True)

    for name, edge_index in metapaths.items():
        np.save(metapaths_dir / f"{name}.npy", edge_index)


def load_metapaths(data_dir: str | Path) -> dict[str, np.ndarray]:
    """Load precomputed metapath edge indices from disk.

    Args:
        data_dir: Directory containing the metapaths/ subdirectory.

    Returns:
        Dict mapping metapath names to edge_index arrays.
    """
    data_dir = Path(data_dir)
    metapaths_dir = data_dir / "metapaths"

    if not metapaths_dir.exists():
        return {}

    results = {}
    for path in metapaths_dir.glob("*.npy"):
        name = path.stem
        results[name] = np.load(path)

    return results


def add_precomputed_metapaths(
    data: HeteroData, metapaths: dict[str, np.ndarray]
) -> HeteroData:
    """Add precomputed metapath edges to a HeteroData object.

    Args:
        data: HeteroData object to augment.
        metapaths: Dict mapping metapath names to edge_index arrays.

    Returns:
        The augmented HeteroData object.
    """
    for name, edge_index in metapaths.items():
        edge_type_name = f"metapath_{name}"
        data["patient", edge_type_name, "patient"].edge_index = torch.from_numpy(
            edge_index
        ).long()
    return data
