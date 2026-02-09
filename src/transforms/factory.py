"""Transform factory for model-specific preprocessing."""

from typing import Any

from torch_geometric.transforms import BaseTransform, Compose

from src.transforms.self_loops import AddSelfLoop, AddHypergraphSelfLoops
from src.transforms.het2homo import HetToHomo
from src.transforms.hypergraph import (
    ExtractV2E,
    ConstructH,
    GenerateG,
    GenerateNormHNHN,
    ConstructNorm,
    BuildHyperedgeDict,
)


HYPERGRAPH_MODELS = frozenset(
    {
        "hypergcn",
        "hnhn",
        "hgnn",
        "hcha",
        "allset",
        "alldeepsets",
        "unigcn",
        "unigcnii",
        "edhnn",
    }
)

HOMOGENEOUS_MODELS = frozenset(
    {
        "mlp",
        "gcn",
        "gat",
        "gin",
        "graphsage",
    }
)

HETEROGENEOUS_MODELS = frozenset(
    {
        "rgcn",
        "han",
        "heco",
        "hgmae",
        "hgt",
    }
)


def get_method_preprocess(
    config: Any,
    is_hetero_data: bool = False,
) -> list[BaseTransform]:
    """Get model-specific preprocessing transforms."""
    transforms: list[BaseTransform] = []
    model_key = config.model.model_name.lower()

    if model_key == "hgnn":
        transforms.append(ExtractV2E())
        if config.data.add_self_loop:
            transforms.append(AddHypergraphSelfLoops())
        transforms.append(ConstructH())
        transforms.append(GenerateG())
        return transforms

    if model_key == "hnhn":
        transforms.append(ExtractV2E())
        if config.data.add_self_loop:
            transforms.append(AddHypergraphSelfLoops())
        transforms.append(
            GenerateNormHNHN(
                alpha=config.model.hnhn_alpha,
                beta=config.model.hnhn_beta,
            )
        )
        return transforms

    if model_key in {"hcha", "allset", "alldeepsets", "edhnn", "unigcn", "unigcnii"}:
        transforms.append(ExtractV2E())
        if config.data.add_self_loop:
            transforms.append(AddHypergraphSelfLoops())
        transforms.append(ConstructNorm())
        return transforms

    if model_key == "hypergcn":
        transforms.append(ExtractV2E())
        transforms.append(BuildHyperedgeDict())
        return transforms

    if model_key == "rgcn" and is_hetero_data:
        transforms.append(HetToHomo())
        return transforms

    if model_key in HOMOGENEOUS_MODELS and is_hetero_data:
        transforms.append(HetToHomo())

        if config.data.add_self_loop:
            transforms.append(AddSelfLoop())

        return transforms

    if model_key in HOMOGENEOUS_MODELS:
        if config.data.add_self_loop:
            transforms.append(AddSelfLoop())
        return transforms

    return transforms


def get_pre_transform(config: Any, is_hetero_data: bool = False) -> Compose | None:
    """Get composed pre_transform for caching."""
    transforms = get_method_preprocess(config, is_hetero_data)
    if not transforms:
        return None
    return Compose(transforms)


def is_model_homogeneous(model_name: str) -> bool:
    """Check if a model expects homogeneous graph input."""
    return model_name.lower() in HOMOGENEOUS_MODELS


def is_model_heterogeneous(model_name: str) -> bool:
    """Check if a model handles heterogeneous graphs natively."""
    return model_name.lower() in HETEROGENEOUS_MODELS


def is_model_hypergraph(model_name: str) -> bool:
    """Check if a model works on hypergraphs."""
    return model_name.lower() in HYPERGRAPH_MODELS
