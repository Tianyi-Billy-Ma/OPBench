"""Data-related configuration."""

from dataclasses import dataclass, field
from typing import Literal

from .base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Data configuration.

    Controls dataset loading, preprocessing, and splitting.

    Attributes:
        dataset: Name of the dataset (e.g., 'cora', 'pdmp').
        data_dir: Root directory for datasets.
        train_prop: Proportion of nodes for training.
        val_prop: Proportion of nodes for validation.
        test_prop: Proportion of nodes for testing.
        split_balance: Whether to balance class distribution in splits.
        add_self_loop: Whether to add self-loops to the graph.
        normalize_features: Whether to normalize node features.
        normalize_features_method: Method for feature normalization.
        heterogeneous: Whether the dataset is heterogeneous.

        # Runtime attributes (set during data loading)
        num_features: Number of input features (set at runtime).
        num_classes: Number of output classes (set at runtime).
        num_nodes: Number of nodes in the graph (set at runtime).
        num_hyperedges: Number of hyperedges (set at runtime).
        metadata: Metadata for heterogeneous graphs (set at runtime).
        target_node_type: Target node type for heterogeneous graphs.
    """

    # Dataset identification
    dataset: str = field(
        default="cora",
        metadata={"help": "Name of the dataset"},
    )
    data_dir: str = field(
        default="./datasets",
        metadata={"help": "Root directory for datasets"},
    )

    # Data splitting
    train_prop: float = field(
        default=0.6,
        metadata={"help": "Proportion of training data"},
    )
    val_prop: float = field(
        default=0.1,
        metadata={"help": "Proportion of validation data"},
    )
    test_prop: float = field(
        default=0.2,
        metadata={"help": "Proportion of test data"},
    )
    split_balance: bool = field(
        default=True,
        metadata={"help": "Whether to balance class distribution in splits"},
    )

    # Preprocessing
    add_self_loop: bool = field(
        default=True,
        metadata={"help": "Whether to add self-loops to the graph"},
    )
    normalize_features: bool = field(
        default=False,
        metadata={"help": "Whether to normalize node features"},
    )
    normalize_features_method: Literal["l2", "standard"] = field(
        default="l2",
        metadata={"help": "Method for feature normalization: 'l2' or 'standard'"},
    )
    reprocess: bool = field(
        default=False,
        metadata={"help": "Force reprocessing of dataset even if cached"},
    )
    hetero_to_homo: Literal["metapath", "pyg", "none"] = field(
        default="metapath",
        metadata={
            "help": "Method to convert HeteroData to homogeneous: "
            "'metapath' uses metapath edges, 'pyg' uses to_homogeneous(), 'none' keeps hetero"
        },
    )

    # Graph type
    heterogeneous: bool = field(
        default=False,
        metadata={"help": "Whether the dataset is heterogeneous"},
    )

    # Runtime attributes (set during data loading, not from CLI)
    num_features: int = field(
        default=0,
        metadata={"help": "Number of input features (set at runtime)"},
    )
    num_classes: int = field(
        default=0,
        metadata={"help": "Number of output classes (set at runtime)"},
    )
    num_nodes: int = field(
        default=0,
        metadata={"help": "Number of nodes in the graph (set at runtime)"},
    )
    num_hyperedges: int = field(
        default=0,
        metadata={"help": "Number of hyperedges (set at runtime)"},
    )
    metadata: tuple | None = field(
        default=None,
        metadata={"help": "Metadata for heterogeneous graphs (set at runtime)"},
    )
    target_node_type: str | None = field(
        default=None,
        metadata={"help": "Target node type for heterogeneous graphs"},
    )

    def __post_init__(self):
        super().__post_init__()
