This folder contains codes for model implementations. Each model or a group of related models should be implemented in a separate sub-folder for better organization and maintainability, while shared components can be placed in this folder directly.

## Naming Convention

- **backbone**: An encoder network (GNN, HetGNN, HyGNN) that produces embeddings. Examples: `GCN`, `GAT`, `HGMAE`, `AllSet`.
- **model**: A wrapper that combines a backbone with task-specific heads (e.g., classifier). Examples: `NodeModel`, `HetModel`.

In code:
- Backbone classes are named as `XXX` (e.g., `GCN`, `HGMAE`)
- Model wrapper classes are named as `XXXModel` (e.g., `NodeModel`)
- Variables should use `backbone` for encoders and `model` for wrappers

## Structure

`gnns/`: implementations of various Graph Neural Network (GNN) architectures, such as GCN, GAT, GraphSAGE, etc.
`hetgnns/`: implementations of heterogeneous GNN architectures that can handle graphs with multiple types of nodes and edges.
`hygnns/`: implementations of hypergraph neural network architectures for modeling higher-order relationships among nodes.
`transformers/`: implementations of transformer-based models adapted for graph data.
`models/`: wrapper classes for specific type of downstream tasks, such as graph classification, node classification, (hyper-)link prediction, generative tasks, etc.
`base.py`: an abstract base class for all backbones and models, which defines the common interface and shared functionalities.
`mlp.py`: implementation of a simple Multi-Layer Perceptron (MLP) model that can be used as a baseline or for non-graph data.
`registry.py`: a registry for all available backbones in the codebase, allowing for easy model selection and instantiation based on configuration.
