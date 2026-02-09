import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter


class ExtractV2E(BaseTransform):
    """Extract node-to-hyperedge (V->E) edges only.

    Filters bipartite edge_index to keep only edges where src < num_nodes.
    This removes E->V edges, keeping only the V->E direction.
    """

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        edge_index = data.edge_index
        num_nodes = (
            data.num_nodes
            if data.num_nodes is not None
            else int(edge_index.max().item()) + 1
        )

        _, sorted_idx = torch.sort(edge_index[0])
        edge_index = edge_index[:, sorted_idx]

        mask = edge_index[0] < num_nodes
        data.edge_index = edge_index[:, mask]

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[sorted_idx][mask]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ConstructH(BaseTransform):
    """Build incidence matrix H from edge_index.

    Creates data.H as a dense tensor of shape [num_nodes, num_hyperedges].
    Requires ExtractV2E to be applied first (edge_index should be V->E only).
    """

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        edge_index = data.edge_index.cpu().numpy()
        num_nodes = (
            data.x.shape[0]
            if hasattr(data, "x") and data.x is not None
            else int(edge_index[0].max()) + 1
        )

        he_min = int(edge_index[1].min())
        he_max = int(edge_index[1].max())
        num_hyperedges = he_max - he_min + 1

        H = np.zeros((num_nodes, num_hyperedges), dtype=np.float32)
        cur_idx = 0
        for he in np.unique(edge_index[1]):
            nodes_in_he = edge_index[0][edge_index[1] == he]
            H[nodes_in_he, cur_idx] = 1.0
            cur_idx += 1

        data.H = torch.from_numpy(H).to(data.edge_index.device)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GenerateG(BaseTransform):
    """Generate HGNN propagation matrix from incidence matrix H.

    Computes G = D_V^(-1/2) H W D_E^(-1) H^T D_V^(-1/2)
    where D_V is node degree, D_E is hyperedge degree, W is identity.

    Requires ConstructH to be applied first (data.H must exist).
    Sets data.G as the propagation matrix.
    """

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "H"):
            return data

        H = data.H.cpu().numpy()
        n_edge = H.shape[1]

        W = np.ones(n_edge)
        DV = np.sum(H * W, axis=1)
        DE = np.sum(H, axis=0)

        invDE = np.diag(np.power(DE + 1e-12, -1))
        DV2 = np.diag(np.power(DV + 1e-12, -0.5))
        DV2 = np.nan_to_num(DV2)
        W_diag = np.diag(W)

        G = DV2 @ H @ W_diag @ invDE @ H.T @ DV2
        data.G = torch.from_numpy(G.astype(np.float32)).to(data.H.device)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GenerateNormHNHN(BaseTransform):
    """Compute HNHN degree-based normalizations.

    Sets:
        data.D_v_beta: d_v^beta (node degree to power beta)
        data.D_e_beta_inv: (sum of d_v^beta for nodes in hyperedge)^(-1)
        data.D_e_alpha: d_e^alpha (hyperedge size to power alpha)
        data.D_v_alpha_inv: (sum of d_e^alpha for edges containing node)^(-1)

    Args:
        alpha: Power for hyperedge size normalization.
        beta: Power for node degree normalization.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        edge_index = data.edge_index
        num_nodes = (
            data.num_nodes
            if data.num_nodes is not None
            else int(edge_index.max().item()) + 1
        )
        device = edge_index.device

        v2e_mask = (edge_index[0] < num_nodes) & (edge_index[1] >= num_nodes)
        v2e_index = edge_index[:, v2e_mask]

        if v2e_index.numel() == 0:
            return data

        v_idx = v2e_index[0]
        e_idx = v2e_index[1]

        he_min = int(e_idx.min().item())
        num_hyperedges = int(e_idx.max().item()) - he_min + 1
        e_idx_norm = e_idx - he_min

        d_v = scatter(
            torch.ones(v_idx.size(0), device=device),
            v_idx,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        d_e = scatter(
            torch.ones(e_idx_norm.size(0), device=device),
            e_idx_norm,
            dim=0,
            dim_size=num_hyperedges,
            reduce="sum",
        )

        D_v_beta = d_v.pow(self.beta)

        D_e_beta = scatter(
            D_v_beta[v_idx], e_idx_norm, dim=0, dim_size=num_hyperedges, reduce="sum"
        )
        D_e_beta_inv = D_e_beta.pow(-1)
        D_e_beta_inv[torch.isinf(D_e_beta_inv)] = 0

        D_e_alpha = d_e.pow(self.alpha)

        D_v_alpha = scatter(
            D_e_alpha[e_idx_norm], v_idx, dim=0, dim_size=num_nodes, reduce="sum"
        )
        D_v_alpha_inv = D_v_alpha.pow(-1)
        D_v_alpha_inv[torch.isinf(D_v_alpha_inv)] = 0

        data.D_v_beta = D_v_beta
        data.D_e_beta_inv = D_e_beta_inv
        data.D_e_alpha = D_e_alpha
        data.D_v_alpha_inv = D_v_alpha_inv
        data.num_hyperedges_hnhn = num_hyperedges

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})"


class ConstructNorm(BaseTransform):
    """Compute edge normalization weights.

    Sets data.norm with edge weights based on the specified option.

    Args:
        option: Normalization type.
            - "all_one": All weights = 1 (default)
            - "deg_half_sym": D_V^(-1/2) * D_E^(-1/2) symmetric normalization
    """

    def __init__(self, option: str = "all_one"):
        super().__init__()
        self.option = option

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        edge_index = data.edge_index
        device = edge_index.device

        if self.option == "all_one":
            data.norm = torch.ones(edge_index.size(1), device=device)
        elif self.option == "deg_half_sym":
            edge_weight = torch.ones(edge_index.size(1), device=device)
            cidx = edge_index[1].min()

            Vdeg = scatter(edge_weight, edge_index[0], dim=0, reduce="sum")
            HEdeg = scatter(edge_weight, edge_index[1] - cidx, dim=0, reduce="sum")

            V_norm = Vdeg.pow(-0.5)
            E_norm = HEdeg.pow(-0.5)
            V_norm[torch.isinf(V_norm)] = 0
            E_norm[torch.isinf(E_norm)] = 0

            data.norm = V_norm[edge_index[0]] * E_norm[edge_index[1] - cidx]
        else:
            data.norm = torch.ones(edge_index.size(1), device=device)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(option='{self.option}')"


class BuildHyperedgeDict(BaseTransform):
    """Build hyperedge dictionary for HyperGCN.

    Creates data.He_dict mapping hyperedge_id -> list of node_ids.
    Used for clique expansion in HyperGCN.
    """

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        edge_index = data.edge_index.cpu().numpy()
        edge_index[1, :] = edge_index[1, :] - edge_index[1, :].min()

        He_dict = {}
        for he in np.unique(edge_index[1, :]):
            nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
            He_dict[int(he)] = nodes_in_he

        data.He_dict = He_dict
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
