# OPBench: A Graph Benchmark to Combat the Opioid Crisis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OPBench is a comprehensive graph-based benchmark designed for research on combating the opioid crisis. It provides a unified framework for evaluating Graph Neural Networks (GNNs), Heterogeneous GNNs, and Hypergraph Neural Networks on drug-related detection and classification tasks.

## Datasets

OPBench includes 5 carefully curated datasets spanning heterogeneous graphs, hypergraphs, and multi-relation graphs:

| Dataset | Type | Nodes | Edges | Features | Classes | Task |
|---------|------|-------|-------|----------|---------|------|
| **PDMP** | Heterogeneous | 85,908 | 1.2M+ | 768 | 2 | Opioid Overdose Detection |
| **NHANCE** | Heterogeneous | 12,453 | 89K+ | 768 | 2 | Diet Role Classification |
| **Twitter-HyDrug-Role** | Hypergraph | 3,591 | 11,940 | 200 | 4 | Drug Role Detection |
| **Twitter-HyDrug-Comm** | Hypergraph | 3,591 | 11,940 | 200 | 8 | Community Detection (Multi-label) |
| **Twitter-MRDrug-Role** | Multi-Relation | 27,945 | 436K+ | 384 | 4 | Drug Role Detection |

### Dataset Details

#### PDMP (Prescription Drug Monitoring Program)
A heterogeneous graph constructed from prescription records with 4 node types:
- **Patient**: Target nodes for overdose prediction
- **Prescriber**: Healthcare providers
- **Pharmacy**: Dispensing locations
- **Drug**: Medications prescribed

#### Twitter Datasets
Social media data for drug-related behavior analysis:
- **HyDrug-Role/Comm**: Hypergraph representation of user interactions
- **MRDrug-Role**: Multi-relation graph with 3 edge types (keyword, follow, tweet)

#### NHANCE
A heterogeneous graph for nutrition and health analysis with node types: user, food, ingredient, category, habit.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python environment management.

```bash
# Clone the repository
git clone https://github.com/Tianyi-Billy-Ma/OPBench.git
cd OPBench

# Create and sync environment with uv
uv sync

# Or install with pip
pip install -e .
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- PyTorch Lightning >= 2.0.0

## Quick Start

### Single Run

Train a model on a specific dataset:

```bash
# Using a config file
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.main run ./configs/run/pdmp.yaml

# With command-line overrides
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.main run ./configs/run/pdmp.yaml \
    --lr 0.001 \
    --hidden_dim 256 \
    --epochs 100
```

### Experiment Mode

Run multiple trials with different seeds:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.main exp ./configs/run/pdmp.yaml \
    --num_runs 5
```

### Hyperparameter Sweep

Perform hyperparameter optimization:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.main sweep ./configs/sweep/pdmp_models.yaml
```

## Supported Models

### Standard GNNs
- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GraphSAGE** - Sampling and Aggregation
- **GIN** - Graph Isomorphism Network

### Heterogeneous GNNs
- **HAN** - Heterogeneous Attention Network
- **HGT** - Heterogeneous Graph Transformer
- **RGCN** - Relational Graph Convolutional Network
- **HGMAE** - Heterogeneous Graph Masked Autoencoder

### Hypergraph Neural Networks
- **HGNN** - Hypergraph Neural Network
- **HyperGCN** - Hypergraph Convolution
- **AllSet** - Learning Allset Transformers
- **ED-HNN** - Equivariant Hypergraph Neural Network
- **HNHN** - Hypergraph Networks with Hyperedge Neurons

## Configuration

OPBench uses a hierarchical configuration system with YAML files and command-line overrides.

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | `pdmp` |
| `--model_name` | Model architecture | `gcn` |
| `--hidden_dim` | Hidden layer dimension | `128` |
| `--num_layers` | Number of GNN layers | `2` |
| `--dropout` | Dropout rate | `0.5` |
| `--lr` | Learning rate | `0.01` |
| `--epochs` | Training epochs | `500` |
| `--patience` | Early stopping patience | `50` |
| `--train_prop` | Training split ratio | `0.6` |

### Example Config

```yaml
mode: run
data:
  dataset: pdmp
  train_prop: 0.5
  val_prop: 0.1
model:
  model_name: gcn
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3
train:
  epochs: 500
  lr: 0.001
  weight_decay: 5e-4
  patience: 50
```

## Project Structure

```
OPBench/
├── configs/
│   ├── run/           # Single run configurations
│   └── sweep/         # Hyperparameter sweep configurations
├── datasets/
│   ├── hetgraphs/     # Heterogeneous graph datasets
│   ├── hypergraphs/   # Hypergraph datasets
│   └── graphs/        # Standard graph datasets
├── src/
│   ├── data/          # Data loading and processing
│   ├── models/        # Model implementations
│   ├── train/         # Training logic
│   ├── metrics/       # Evaluation metrics
│   ├── hparams/       # Configuration management
│   └── main.py        # Entry point
└── outputs/           # Experiment results
```

## Output Structure

Results are saved in `./outputs/<run_name>/`:

```
outputs/<run_name>/
├── config/            # Saved configuration
├── pretrain/          # Pretrain checkpoints (if applicable)
├── finetune/          # Finetune checkpoints
├── eval/              # Evaluation results
│   ├── run_results.json
│   └── run_results.md
└── logs/              # Training logs
```

## Citation

If you use OPBench in your research, please cite:

```bibtex
@misc{opbench2025,
  title={OPBench: A Graph Benchmark to Combat the Opioid Crisis},
  author={Ma, Tianyi},
  year={2025},
  url={https://github.com/Tianyi-Billy-Ma/OPBench}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This benchmark builds upon research in graph neural networks and their applications to public health challenges. We thank the open-source community for their contributions to PyTorch Geometric and related libraries.
