This folder contains the configuration classes for different modules.

`base_config.py` defines the base configuration class.
`log_config.py` defines the logging configuration class, including output paths (`save_dir`, `pretrain_dir`, `finetune_dir`, `eval_dir`, `cfg_dir`, `log_dir`).
`data_config.py` defines the data-related configuration class, such as `dataset`, `train_prop`, `val_prop`, `test_prop`, etc.
`model_config.py` defines the model-related configuration class, such as `model_name`, `hidden_dim`, `num_layers`, etc.
`train_config.py` defines the training-related configuration class, such as `batch_size`, `lr`, `epochs`, `pretrain_monitor`, `finetune_monitor`, etc.
