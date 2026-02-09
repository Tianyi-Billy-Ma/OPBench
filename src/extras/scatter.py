from torch_geometric.utils import scatter


def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


__all__ = ["scatter", "scatter_add"]
