This folder contains the python codes for configuration loading, patching and saving.

`parser.py` defines the argument parser for command line arguments or configuration files.
`patcher.py` contains functions to patch the loaded configurations if necessary.
`verify.py` contains functions to verify the loaded and patched configurations are correct. This should be the last step before saving and using the configurations.
`configs/` contains the configuration classes for different modules.

## Output Directory Logic

The output directory structure is managed by `LogConfig` and `patcher.py`:

- `output_dir`: Base directory (default `./outputs`).
- `save_dir`: Specific run directory.
  - If manually specified via command line, it overrides automatic naming.
  - Otherwise, constructed as `{output_dir}/{mode}_{dataset}_{model}_{timestamp}` (or `{output_dir}/{run_name}` if `run_name` is set).

Subdirectories automatically created under `save_dir`:
- `pretrain/`: Pretrain checkpoints.
- `finetune/`: Finetune checkpoints.
- `eval/`: Evaluation results.
- `config/`: Saved configuration files.
- `logs/`: Training logs.

The `LogConfig` object provides `pathlib.Path` properties for these directories (`save_path`, `pretrain_path`, `finetune_path`, `eval_path`, `cfg_path`, `log_path`).
