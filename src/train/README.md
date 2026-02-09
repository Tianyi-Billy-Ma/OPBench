This folder contains codes for training models.
Each unique training workflow should be implemented into a separate folder for better modularity and maintainability, while universal or shared training components can be placed in this folder directly.

Below are examples:
`pretrain/`: codes for pretraining models.
`finetune/`: codes for fine-tuning pretrained models on downstream tasks.
`callbacks/`: codes for training callbacks (checkpointing, early stopping, etc.).

## Notes

We use `pytorch-lightning` to train models.

## Callbacks

The `callbacks/setup.py` module provides:
- `setup_callbacks(config, phase)`: Creates EarlyStopping and ModelCheckpoint callbacks based on config.
- `get_checkpoint_callback(callbacks)`: Retrieves the ModelCheckpoint callback from a list.
- `load_checkpoint(callbacks, module, phase)`: Loads the best checkpoint into a module.

Monitor metrics are configurable via `config.train.pretrain_monitor` and `config.train.finetune_monitor`.
