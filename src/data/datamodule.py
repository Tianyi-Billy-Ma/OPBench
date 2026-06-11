import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from src.data.datasets import GraphDataset, HypergraphDataset
from src.transforms import get_pre_transform

# OPBench Dataset Configuration
# This benchmark includes 5 datasets for opioid crisis research:
# - pdmp: Prescription Drug Monitoring Program (heterogeneous graph)
# - nhance: NHANCE diet role classification (heterogeneous graph)
# - twitter_hydrug_role: Twitter drug role detection (hypergraph, single-label)
# - twitter_hydrug_comm: Twitter drug community detection (hypergraph, multi-label)
# - twitter_mrdrug_role: Twitter multi-relation drug role detection (graph)

DATASET_CONFIG = {
    # Heterogeneous graph datasets
    "pdmp": {
        "class": GraphDataset,
        "subdir": "hetgraphs/pdmp_opioid_detect",
        "target_node_type": "patient",
        "is_hetero": True,
    },
    "nhance": {
        "class": GraphDataset,
        "subdir": "hetgraphs/nhance_diet_role",
        "target_node_type": "user",
        "is_hetero": True,
    },
    # Hypergraph datasets (Twitter drug-related)
    "twitter_hydrug_role": {
        "class": HypergraphDataset,
        "subdir": "hypergraphs/twitter_hydrug_role",
        "target_node_type": None,
        "is_hetero": False,
    },
    "twitter_hydrug_comm": {
        "class": HypergraphDataset,
        "subdir": "hypergraphs/twitter_hydrug_comm",
        "target_node_type": None,
        "is_hetero": False,
    },
    # Multi-relation graph dataset
    "twitter_mrdrug_role": {
        "class": GraphDataset,
        "subdir": "graphs/twitter_mrdrug_role",
        "target_node_type": None,
        "is_hetero": False,
    },
}


class OPBenchDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data = None
        self.dataset = None
        self.repo_root = Path(os.getcwd())

    def setup(self, stage=None):
        dataset_name = self.config.data.dataset.lower()

        if dataset_name not in DATASET_CONFIG:
            raise NotImplementedError(f"Dataset {dataset_name} not supported.")

        ds_config = DATASET_CONFIG[dataset_name]
        root = self.repo_root / self.config.data.data_dir / ds_config["subdir"]

        method_name = self.config.model.model_name.lower()
        reprocess = self.config.data.reprocess
        is_hetero = ds_config.get("is_hetero", False)

        pre_transform = get_pre_transform(self.config, is_hetero_data=is_hetero)

        if ds_config["class"] == GraphDataset:
            self.dataset = GraphDataset(
                root=str(root),
                name=dataset_name,
                target_node_type=ds_config["target_node_type"],
                method_name=method_name,
                reprocess=reprocess,
                pre_transform=pre_transform,
            )
        else:
            self.dataset = ds_config["class"](
                root=str(root),
                name=dataset_name,
                method_name=method_name,
                reprocess=reprocess,
                pre_transform=pre_transform,
            )

        self.data = self.dataset[0]
        self.generate_splits(self.data)

    def generate_splits(self, data):
        target_node_type = self._get_target_node_type()

        if isinstance(data, HeteroData) and target_node_type:
            num_nodes = data[target_node_type].x.shape[0]
        else:
            target_node_type = None
            num_nodes = data.x.shape[0]

        g = torch.Generator()
        g.manual_seed(self.config.train.seed)
        indices = torch.randperm(num_nodes, generator=g)

        n_train = int(num_nodes * self.config.data.train_prop)
        n_val = int(num_nodes * self.config.data.val_prop)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[indices[:n_train]] = True

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[indices[n_train : n_train + n_val]] = True

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[indices[n_train + n_val :]] = True

        if target_node_type is not None:
            data[target_node_type].train_mask = train_mask
            data[target_node_type].val_mask = val_mask
            data[target_node_type].test_mask = test_mask
        else:
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask

    def _get_target_node_type(self) -> str | None:
        dataset_name = self.config.data.dataset.lower()
        if dataset_name in DATASET_CONFIG:
            return DATASET_CONFIG[dataset_name]["target_node_type"]
        return None

    def train_dataloader(self):
        assert self.data is not None
        return DataLoader([self.data], batch_size=1, shuffle=False)

    def val_dataloader(self):
        assert self.data is not None
        return DataLoader([self.data], batch_size=1, shuffle=False)

    def test_dataloader(self):
        assert self.data is not None
        return DataLoader([self.data], batch_size=1, shuffle=False)
