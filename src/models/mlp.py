from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseBackbone
from src.hparams import FullArguments
from src.models.registry import ModelRegistry


class MLP(BaseBackbone):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        normalization: Literal["bn", "ln", "none"] = "bn",
        input_norm: bool = False,
        activation: Literal["relu", "elu", "prelu", "gelu", "id"] = "relu",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.norms.append(self._get_norm(in_channels, normalization, input_norm))
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.norms.append(self._get_norm(in_channels, normalization, input_norm))
            self.lins.append(nn.Linear(in_channels, hidden_channels))

            for _ in range(num_layers - 2):
                self.norms.append(self._get_norm(hidden_channels, normalization, True))
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))

            self.norms.append(self._get_norm(hidden_channels, normalization, True))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.activation = self._get_activation(activation)

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()
        self._reset_norm_parameters(self.norms)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = args[0] if args else kwargs.get("x")
        assert x is not None, "MLP.forward requires input tensor x"
        x = self.norms[0](x)

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            x = self.norms[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, "
            f"hid={self.hidden_channels}, "
            f"out={self.out_channels}, "
            f"layers={self.num_layers})"
        )


@ModelRegistry.register("mlp")
class MLPBackbone(BaseBackbone):
    def __init__(self, config: FullArguments):
        super().__init__()
        self.config = config

        num_features = config.data.num_features
        hidden_dim = config.model.hidden_dim
        output_dim = config.model.output_dim
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        normalization = config.model.normalization
        activation = config.model.activation

        assert output_dim is not None, "output_dim must be set (should be patched)"

        self.mlp = MLP(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
        )

    def forward(self, batch) -> dict:
        h = self.mlp(batch.x)
        return {"embeddings": h}

    def compute_loss(self, batch, outputs: dict) -> dict:
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def reset_parameters(self) -> None:
        self.mlp.reset_parameters()
