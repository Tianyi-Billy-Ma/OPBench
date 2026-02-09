import shutil
from collections.abc import Callable
from pathlib import Path

from torch_geometric.data import Data, HeteroData, InMemoryDataset

from src.data.loader import build_dataset


class GraphDataset(InMemoryDataset):
    """Unified dataset class for both homogeneous and heterogeneous graphs.

    Handles Data and HeteroData transparently, providing consistent properties
    that work for both types.
    """

    def __init__(
        self,
        root: str,
        name: str,
        target_node_type: str | None = None,
        method_name: str | None = None,
        reprocess: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        self.name = name.lower()
        self._target_node_type = target_node_type
        self._method_name = method_name.lower() if method_name else "default"
        self._reprocess = reprocess
        self.builder_kwargs = kwargs

        if reprocess:
            processed_path = Path(root) / "processed" / self._method_name
            if processed_path.exists():
                shutil.rmtree(processed_path)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed" / self._method_name)

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        data = build_dataset(self.name, Path(self.raw_dir), **self.builder_kwargs)

        if self.pre_filter is not None:
            if isinstance(data, (list, tuple)):
                data = [d for d in data if self.pre_filter(d)]  # type: ignore[union-attr]
            elif not self.pre_filter(data):
                data = None

        if self.pre_transform is not None:
            if isinstance(data, (list, tuple)):
                data = [self.pre_transform(d) for d in data]  # type: ignore[union-attr]
            elif data is not None:
                data = self.pre_transform(data)

        if isinstance(data, (list, tuple)):
            self.save(list(data), self.processed_paths[0])
        elif data is not None:
            self.save([data], self.processed_paths[0])

    @property
    def target_node_type(self) -> str | None:
        return self._target_node_type

    @property
    def is_hetero(self) -> bool:
        data = self[0]
        return isinstance(data, HeteroData)

    @property
    def node_types(self) -> list[str]:
        data = self[0]
        if isinstance(data, HeteroData):
            return list(data.node_types)
        return []

    @property
    def edge_types(self) -> list[tuple]:
        data = self[0]
        if isinstance(data, HeteroData):
            return list(data.edge_types)
        return []

    @property
    def metadata(self) -> tuple:
        data = self[0]
        if isinstance(data, HeteroData):
            return data.metadata()
        return ([], [])

    @property
    def num_features(self) -> int:
        data: Data | HeteroData = self[0]  # type: ignore[assignment]
        if isinstance(data, HeteroData):
            if self._target_node_type:
                store = data[self._target_node_type]
                if hasattr(store, "x") and store.x is not None:
                    return store.x.size(1)
            return 0
        if hasattr(data, "x") and data.x is not None:
            return data.x.size(1)
        return 0

    @property
    def num_classes(self) -> int:
        data: Data | HeteroData = self[0]  # type: ignore[assignment]
        if hasattr(data, "num_classes") and data.num_classes is not None:
            nc = data.num_classes
            return int(nc.item()) if hasattr(nc, "item") else int(nc)
        if isinstance(data, HeteroData):
            if self._target_node_type:
                store = data[self._target_node_type]
                if hasattr(store, "y") and store.y is not None:
                    return int(store.y.max().item()) + 1
            return 0
        if hasattr(data, "y") and data.y is not None:
            return int(data.y.max().item()) + 1  # type: ignore[union-attr]
        return 0


class HypergraphDataset(GraphDataset):
    def __init__(
        self,
        root: str,
        name: str,
        method_name: str | None = None,
        reprocess: bool = False,
        feature_noise: float | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs,
    ):
        self.feature_noise = feature_noise
        if feature_noise is not None:
            kwargs["feature_noise"] = feature_noise
        super().__init__(
            root,
            name,
            target_node_type=None,
            method_name=method_name,
            reprocess=reprocess,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            **kwargs,
        )

    @property
    def processed_file_names(self) -> list[str]:
        if self.feature_noise is not None and self.feature_noise != 0:
            return [f"data_noise_{self.feature_noise}.pt"]
        return ["data.pt"]

    @property
    def num_hyperedges(self) -> int:
        data: Data = self[0]  # type: ignore[assignment]
        if hasattr(data, "num_hyperedges"):
            nh = data.num_hyperedges
            return int(nh.item()) if hasattr(nh, "item") else int(nh)
        return 0
