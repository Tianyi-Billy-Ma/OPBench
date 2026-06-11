"""Model-related configuration."""

from dataclasses import dataclass, field
from typing import Literal

from .base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Model configuration.

    Controls model architecture and hyperparameters.

    Attributes:
        model_name: Name of the model architecture.
        num_layers: Number of GNN layers.
        hidden_dim: Hidden dimension size.
        dropout: Dropout probability.
        input_dropout: Dropout probability for input features.
        num_heads: Number of attention heads (for attention-based models).
        activation: Activation function type.
        normalization: Normalization type.

        # HGMAE specific
        hgmae_mask_rate: Masking rate for HGMAE pretraining.
        hgmae_replace_rate: Replace rate for HGMAE pretraining.
        hgmae_loss_fn: Loss function for HGMAE.
        hgmae_decoder_type: Decoder type for HGMAE.

        # HyperGCN specific
        hypergcn_mediators: Whether to use mediator nodes in Laplacian.
        hypergcn_fast: Whether to cache the computed structure.

        # AllSet specific
        allset_num_layers: Number of MLP layers within each conv.
        allset_pma: Whether to use PMA (attention) or DeepSets.
        allset_num_heads: Number of attention heads for PMA.
        allset_input_norm: Whether to apply input normalization.
        allset_gpr: Whether to use GPR mechanism.
        allset_learn_mask: Whether to learn edge importance.
        aggregate: Aggregation type ('mean' or 'sum').

        # Classifier configuration
        classifier_num_layers: Number of layers in classifier MLP.
        classifier_hidden: Hidden dimension of classifier MLP.
        classifier_dropout: Dropout for classifier.
    """

    # Model identification
    model_name: str = field(
        default="gcn",
        metadata={"help": "Name of the model architecture (gcn, gat, hgmae, etc.)"},
    )

    # Architecture hyperparameters
    num_layers: int = field(
        default=2,
        metadata={"help": "Number of GNN layers"},
    )
    hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden dimension size"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension size (defaults to hidden_dim)"},
    )
    dropout: float = field(
        default=0.5,
        metadata={"help": "Dropout probability"},
    )
    input_dropout: float = field(
        default=0.2,
        metadata={"help": "Dropout probability for input features"},
    )
    num_heads: int = field(
        default=1,
        metadata={"help": "Number of attention heads (for GAT/HAN)"},
    )
    activation: Literal["relu", "elu", "prelu", "gelu", "id"] = field(
        default="relu",
        metadata={"help": "Activation function: relu, elu, prelu, gelu, id"},
    )
    normalization: Literal["bn", "ln", "none"] = field(
        default="bn",
        metadata={"help": "Normalization type: bn (batch), ln (layer), none"},
    )
    input_norm: bool = field(
        default=True,
        metadata={"help": "Whether to apply normalization to input features"},
    )

    # RGCN specific parameters
    num_relations: int | None = field(
        default=None,
        metadata={
            "help": "Number of relation types for RGCN (inferred from data if None)"
        },
    )
    rgcn_num_bases: int | None = field(
        default=None,
        metadata={
            "help": "Number of bases for RGCN basis-decomposition regularization"
        },
    )

    # HGMAE specific parameters
    hgmae_mask_rate: float = field(
        default=0.5,
        metadata={"help": "Masking rate for HGMAE pretraining"},
    )
    hgmae_replace_rate: float = field(
        default=0.1,
        metadata={"help": "Replace rate for HGMAE pretraining"},
    )
    hgmae_loss_fn: Literal["mse", "sce"] = field(
        default="sce",
        metadata={"help": "Loss function for HGMAE: mse or sce"},
    )
    hgmae_alpha: float = field(
        default=3.0,
        metadata={"help": "Alpha parameter for SCE loss"},
    )
    hgmae_decoder_type: Literal["han", "mlp"] = field(
        default="mlp",
        metadata={"help": "Decoder type for HGMAE: han or mlp"},
    )

    # HyperGCN specific parameters
    hypergcn_mediators: bool = field(
        default=True,
        metadata={"help": "Whether to use mediator nodes in HyperGCN"},
    )
    hypergcn_fast: bool = field(
        default=True,
        metadata={"help": "Whether to cache the computed structure in HyperGCN"},
    )

    # AllSet specific parameters
    allset_num_layers: int = field(
        default=2,
        metadata={"help": "Number of MLP layers within each AllSet conv"},
    )
    allset_pma: bool = field(
        default=True,
        metadata={"help": "Whether to use PMA (attention) or DeepSets in AllSet"},
    )
    allset_num_heads: int = field(
        default=1,
        metadata={"help": "Number of attention heads for PMA in AllSet"},
    )
    allset_input_norm: bool = field(
        default=True,
        metadata={"help": "Whether to apply input normalization in AllSet"},
    )
    allset_gpr: bool = field(
        default=False,
        metadata={"help": "Whether to use GPR mechanism in AllSet"},
    )
    allset_learn_mask: bool = field(
        default=False,
        metadata={"help": "Whether to learn edge importance in AllSet"},
    )
    aggregate: Literal["sum", "mean"] = field(
        default="mean",
        metadata={"help": "Aggregation type for AllSet: sum or mean"},
    )

    # HNHN specific parameters
    hnhn_alpha: float = field(
        default=1.0,
        metadata={"help": "Alpha parameter for HNHN degree normalization"},
    )
    hnhn_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for HNHN degree normalization"},
    )
    hnhn_nonlinear_inbetween: bool = field(
        default=True,
        metadata={"help": "Whether to apply nonlinearity between V2E and E2V in HNHN"},
    )

    # HCHA specific parameters
    hcha_symdegnorm: bool = field(
        default=False,
        metadata={"help": "Whether to use symmetric degree normalization in HCHA"},
    )
    hcha_use_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use attention mechanism in HCHA"},
    )

    # ED-HNN specific parameters
    edhnn_num_layers_1: int = field(
        default=2,
        metadata={"help": "Number of MLP layers for MLP1 in ED-HNN"},
    )
    edhnn_num_layers_2: int = field(
        default=-1,
        metadata={
            "help": "Number of MLP layers for MLP2 in ED-HNN (-1 uses num_layers_1)"
        },
    )
    edhnn_num_layers_3: int = field(
        default=-1,
        metadata={
            "help": "Number of MLP layers for MLP3 in ED-HNN (-1 uses num_layers_1)"
        },
    )
    restart_alpha: float = field(
        default=0.0,
        metadata={"help": "Restart alpha for ED-HNN residual connection"},
    )

    # Classifier configuration (for node classification wrapper)
    classifier_num_layers: int = field(
        default=2,
        metadata={"help": "Number of layers in classifier MLP"},
    )
    classifier_hidden: int = field(
        default=256,
        metadata={"help": "Hidden dimension of classifier MLP"},
    )
    classifier_dropout: float | None = field(
        default=None,
        metadata={"help": "Dropout for classifier (defaults to model dropout)"},
    )

    def __post_init__(self):
        super().__post_init__()
