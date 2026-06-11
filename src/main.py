import warnings

warnings.filterwarnings("ignore")

import copy
import itertools
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision("high")

from src.data import OPBenchDataModule, extract_data_info
from src.hparams import (
    FullArguments,
    create_output_dirs,
    parse_args,
    patch_config,
    update_log_paths,
    verify_config,
)
from src.models import ModelRegistry, NodeModel, PTModel
from src.train import FTTrainer, PTTrainer, load_checkpoint, setup_callbacks
from src.utils import (
    save_json,
    save_markdown_results,
    save_metrics_json,
    set_nested_attr,
    set_seed,
    setup_loggers,
)


def run_ft(config: FullArguments, dm: OPBenchDataModule) -> dict[str, float]:
    """Run finetune-only mode: train encoder + classifier end-to-end."""
    print(
        f"Starting finetune run with seed {config.train.seed} in {config.log.save_dir}"
    )

    set_seed(config.train.seed)
    create_output_dirs(config)

    cfg_dir = cast(Path, config.log.cfg_path)
    config.save_configs(cfg_dir)

    backbone = ModelRegistry.build(config.model.model_name, config)
    model = NodeModel(config, backbone=backbone)

    pl_module = FTTrainer(model, config)

    callbacks = setup_callbacks(config, phase="finetune")
    output_dir = cast(Path, config.log.save_path)
    loggers = setup_loggers(config, phase="finetune")

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.train.finetune_epochs,
        accelerator=config.train.accelerator,
        devices=1,
        callbacks=callbacks,
        logger=loggers,
        check_val_every_n_epoch=config.train.eval_step,
        log_every_n_steps=config.log.log_every_n_steps,
    )

    print("Starting Training...")
    trainer.fit(pl_module, datamodule=dm)

    print("Starting Testing...")
    test_metrics_list = trainer.test(pl_module, datamodule=dm)
    metrics = {}
    if test_metrics_list:
        metrics = cast(dict[str, float], test_metrics_list[0])

    eval_path = cast(Path, config.log.eval_path)
    if metrics:
        save_metrics_json(metrics, eval_path / "run_results.json")
        save_markdown_results(metrics, eval_path / "run_results.md", "Run Results")

    return metrics


def run_gen(config: FullArguments, dm: OPBenchDataModule) -> dict[str, float]:
    """Run generative mode: pretrain encoder, then finetune with classifier."""
    print(
        f"Starting generative run with seed {config.train.seed} in {config.log.save_dir}"
    )

    set_seed(config.train.seed)
    create_output_dirs(config)

    cfg_dir = cast(Path, config.log.cfg_path)
    config.save_configs(cfg_dir)

    output_dir = cast(Path, config.log.save_path)

    backbone = ModelRegistry.build(config.model.model_name, config)

    # Phase 1: Pretrain
    print("\n" + "=" * 50)
    print("Phase 1: Pre-training")
    print("=" * 50)

    pt_model = PTModel(config=config, backbone=backbone)
    pt_module = PTTrainer(pt_model, config)

    pt_callbacks = setup_callbacks(config, phase="pretrain")
    pt_loggers = setup_loggers(config, phase="pretrain")

    pt_trainer = pl.Trainer(
        default_root_dir=config.log.pretrain_path,
        max_epochs=config.train.pretrain_epochs,
        accelerator=config.train.accelerator,
        devices=1,
        callbacks=pt_callbacks,
        logger=pt_loggers,
        enable_progress_bar=True,
        log_every_n_steps=config.log.log_every_n_steps,
        check_val_every_n_epoch=1,
    )

    pt_trainer.fit(pt_module, datamodule=dm)

    load_checkpoint(pt_callbacks, pt_module, phase="pretrain")

    # Phase 2: Finetune
    print("\n" + "=" * 50)
    print("Phase 2: Fine-tuning")
    print("=" * 50)

    model = NodeModel(
        config=config,
        backbone=backbone,
        reset_backbone=False,
        freeze_backbone=config.train.freeze_backbone,
    )

    ft_module = FTTrainer(model, config)

    ft_callbacks = setup_callbacks(config, phase="finetune")
    ft_loggers = setup_loggers(config, phase="finetune")

    ft_trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=config.train.finetune_epochs,
        accelerator=config.train.accelerator,
        devices=1,
        callbacks=ft_callbacks,
        logger=ft_loggers,
        check_val_every_n_epoch=config.train.eval_step,
        log_every_n_steps=config.log.log_every_n_steps,
    )

    ft_trainer.fit(ft_module, datamodule=dm)

    print("Starting Testing...")
    test_metrics_list = ft_trainer.test(ft_module, datamodule=dm)
    metrics = {}
    if test_metrics_list:
        metrics = cast(dict[str, float], test_metrics_list[0])

    eval_path = cast(Path, config.log.eval_path)
    if metrics:
        save_metrics_json(metrics, eval_path / "run_results.json")
        save_markdown_results(metrics, eval_path / "run_results.md", "Run Results")

    return metrics


def run(config: FullArguments, dm: OPBenchDataModule) -> dict[str, float]:
    """Dispatch to run_gen or run_ft based on config.train.mode."""
    if config.train.mode == "gen":
        return run_gen(config, dm)
    elif config.train.mode == "ft":
        return run_ft(config, dm)
    else:
        raise ValueError(f"Unknown train mode: {config.train.mode}. Use 'gen' or 'ft'.")


def run_exp(config: FullArguments, dm: OPBenchDataModule) -> dict[str, float]:
    """Executes a set of experiments (multiple runs) with varying seeds."""
    print(f"Running experiment with {config.train.num_runs} runs...")
    results = []
    base_save_dir = cast(Path, config.log.save_path)
    base_seed = config.train.seed

    for i in range(config.train.num_runs):
        print(f"\n=== Run {i}/{config.train.num_runs} ===")
        run_config = copy.deepcopy(config)

        current_seed = base_seed + i
        run_config.train.seed = current_seed

        run_save_dir = base_save_dir / f"run_{i}"
        update_log_paths(run_config, run_save_dir)

        set_seed(current_seed)
        dm.config = run_config
        dm.setup()

        run_metrics = run(run_config, dm)
        results.extend([run_metrics] if run_metrics else [])

    aggregated_results = {}
    if results:
        print("\n=== Experiment Results ===")
        keys = results[0].keys()
        for k in keys:
            values = [res[k] for res in results]
            mean = np.mean(values)
            std = np.std(values)
            aggregated_results[f"{k}_mean"] = float(mean)
            aggregated_results[f"{k}_std"] = float(std)
            print(f"{k}: {mean * 100:.2f} ± {std * 100:.2f}")

        save_metrics_json(aggregated_results, base_save_dir / "exp_results.json")
        save_markdown_results(
            aggregated_results, base_save_dir / "exp_results.md", "Experiment Results"
        )

        cfg_dir = base_save_dir / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        config.save_configs(cfg_dir)

    return aggregated_results


def run_sweep(config: FullArguments, dm: OPBenchDataModule) -> None:
    """Handles sweep mode execution."""
    print("Sweep mode selected.")

    sweep_params = config.sweep.parameters
    if not sweep_params:
        print("No sweep parameters provided in config.sweep.parameters.")
        return

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    base_save_dir = cast(Path, config.log.save_path)
    metric_key = config.sweep.sweep_metric
    higher_is_better = config.sweep.higher_is_better

    if config.train.sweep_type == "grid":
        combinations = list(itertools.product(*param_values))
        configs_to_run = []
        for combo in combinations:
            cfg_update = {name: val for name, val in zip(param_names, combo)}
            configs_to_run.append(cfg_update)

        best_metric_value = -float("inf") if higher_is_better else float("inf")
        best_exp_idx = -1
        best_results = {}
        best_hparams = {}

        for idx, params in enumerate(configs_to_run):
            print(f"\n=== Sweep Experiment {idx + 1}/{len(configs_to_run)} ===")
            print(f"Params: {params}")

            exp_config = copy.deepcopy(config)
            for path, value in params.items():
                set_nested_attr(exp_config, path, value)

            exp_save_dir = base_save_dir / f"exp_{idx}"
            update_log_paths(exp_config, exp_save_dir)

            exp_results = run_exp(exp_config, dm)

            effective_key = metric_key
            if metric_key not in exp_results:
                fallback_key = metric_key.replace("val_", "test_")
                if fallback_key in exp_results:
                    effective_key = fallback_key

            current_metric = exp_results.get(
                effective_key, -float("inf") if higher_is_better else float("inf")
            )

            print(f"Exp {idx} Result ({effective_key}): {current_metric * 100:.2f}")

            is_better = (
                current_metric > best_metric_value
                if higher_is_better
                else current_metric < best_metric_value
            )
            if is_better:
                best_metric_value = current_metric
                best_exp_idx = idx
                best_results = exp_results
                best_hparams = params

        print(
            f"\nBest Experiment: {best_exp_idx} with {metric_key}={best_metric_value * 100:.2f}"
        )

        best_exp_dir = base_save_dir / f"exp_{best_exp_idx}"
        best_dir = base_save_dir / "best"
        if best_exp_dir.exists():
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(best_exp_dir, best_dir)

        for idx in range(len(configs_to_run)):
            exp_dir = base_save_dir / f"exp_{idx}"
            if exp_dir.exists():
                shutil.rmtree(exp_dir)

    elif config.train.sweep_type == "coordinate":
        best_hparams = {}
        best_results = {}
        exp_idx = 0

        for param_name, values in sweep_params.items():
            print(f"\n{'=' * 60}")
            print(f"Sweeping parameter: {param_name}")
            print(f"{'=' * 60}")

            param_best_value = None
            param_best_metric = -float("inf") if higher_is_better else float("inf")
            param_best_results = {}

            for value in values:
                print(f"\n=== Experiment {exp_idx + 1}: {param_name}={value} ===")

                exp_config = copy.deepcopy(config)
                for prev_param, prev_value in best_hparams.items():
                    set_nested_attr(exp_config, prev_param, prev_value)
                set_nested_attr(exp_config, param_name, value)

                exp_save_dir = base_save_dir / f"exp_{exp_idx}"
                update_log_paths(exp_config, exp_save_dir)

                exp_results = run_exp(exp_config, dm)

                effective_key = metric_key
                if metric_key not in exp_results:
                    fallback_key = metric_key.replace("val_", "test_")
                    if fallback_key in exp_results:
                        effective_key = fallback_key

                current_metric = exp_results.get(
                    effective_key, -float("inf") if higher_is_better else float("inf")
                )

                print(f"Result ({effective_key}): {current_metric * 100:.2f}")

                is_better = (
                    current_metric > param_best_metric
                    if higher_is_better
                    else current_metric < param_best_metric
                )
                if is_better:
                    param_best_metric = current_metric
                    param_best_value = value
                    param_best_results = exp_results

                shutil.rmtree(exp_save_dir, ignore_errors=True)
                exp_idx += 1

            best_hparams[param_name] = param_best_value
            best_results = param_best_results
            print(
                f"\nBest {param_name}: {param_best_value} ({metric_key}={param_best_metric * 100:.2f})"
            )

        print(f"\n{'=' * 60}")
        print("Coordinate Sweep Complete")
        print(f"{'=' * 60}")
        print(f"Best hyperparameters: {best_hparams}")

        print("\nRunning final experiment with best hyperparameters...")
        final_config = copy.deepcopy(config)
        for param_name, value in best_hparams.items():
            set_nested_attr(final_config, param_name, value)

        best_dir = base_save_dir / "best"
        update_log_paths(final_config, best_dir)
        best_results = run_exp(final_config, dm)

    else:
        raise ValueError(f"Unknown sweep type: {config.train.sweep_type}")

    save_json(best_hparams, base_save_dir / "best_hparams.json")
    save_metrics_json(best_results, base_save_dir / "sweep_results.json")
    save_markdown_results(
        best_results, base_save_dir / "sweep_results.md", "Best Sweep Results"
    )
    config.sweep.to_yaml(base_save_dir / "sweep_config.yaml")


def main():
    config = parse_args()
    if config.log.verbose:
        print(f"Configuration: {config}")

    dm = OPBenchDataModule(config)
    dm.setup()
    overrides = extract_data_info(dm, config)

    config = patch_config(config, overrides)

    verify_config(config)

    if config.mode == "run":
        run(config, dm)
    elif config.mode == "exp":
        run_exp(config, dm)
    elif config.mode == "sweep":
        run_sweep(config, dm)
    else:
        raise ValueError(f"Unknown execution mode: {config.mode}")


if __name__ == "__main__":
    main()
