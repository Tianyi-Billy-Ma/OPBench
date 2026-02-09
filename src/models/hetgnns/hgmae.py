import torch
import torch.nn as nn
import torch.nn.functional as F
from .han import HANEncoder
from ..base import BaseBackbone
from src.hparams import FullArguments
from src.models.registry import ModelRegistry


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow(alpha)
    return loss.mean()


@ModelRegistry.register("hgmae", heterogeneous=True)
class HGMAE(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        metadata = config.data.metadata
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        heads = config.model.num_heads
        dropout = config.model.dropout
        normalization = config.model.normalization
        activation = config.model.activation
        input_norm = config.model.input_norm

        assert output_dim is not None, "output_dim must be set"

        self.mask_rate = config.model.hgmae_mask_rate
        self.metadata = metadata
        feature_dim = config.data.num_features

        self.encoder = HANEncoder(
            metadata=metadata,
            in_channels=feature_dim,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
            input_norm=input_norm,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, feature_dim))
        nn.init.xavier_normal_(self.mask_token)

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def mask_features(self, x_dict):
        masked_x_dict = {}
        mask_nodes_dict = {}

        for node_type, x in x_dict.items():
            num_nodes = x.shape[0]
            perm = torch.randperm(num_nodes, device=x.device)
            num_mask = int(self.mask_rate * num_nodes)
            mask_nodes = perm[:num_mask]

            masked_x = x.clone()
            masked_x[mask_nodes] = self.mask_token

            masked_x_dict[node_type] = masked_x
            mask_nodes_dict[node_type] = mask_nodes

        return masked_x_dict, mask_nodes_dict

    def forward(self, batch) -> dict:
        x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
        masked_x_dict, mask_nodes_dict = self.mask_features(x_dict)
        z_dict = self.encoder(masked_x_dict, edge_index_dict)

        return {
            "embeddings": z_dict,
            "mask_nodes_dict": mask_nodes_dict,
            "x_dict": x_dict,
        }

    def compute_loss(self, batch, outputs: dict) -> dict:
        z_dict = outputs["embeddings"]
        mask_nodes_dict = outputs["mask_nodes_dict"]
        x_dict = outputs["x_dict"]

        recon_loss = 0
        for node_type, z in z_dict.items():
            mask_nodes = mask_nodes_dict[node_type]
            if mask_nodes.numel() > 0:
                recon_feat = self.decoder(z[mask_nodes])
                target_feat = x_dict[node_type][mask_nodes]
                recon_loss += sce_loss(recon_feat, target_feat)

        return {"loss": recon_loss}

    def get_embedding(self, batch):
        return self.encoder(batch.x_dict, batch.edge_index_dict)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        nn.init.xavier_normal_(self.mask_token)
        for layer in self.decoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
