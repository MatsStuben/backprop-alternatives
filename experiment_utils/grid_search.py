from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_task_data
from .evaluation import aggregate_grid_summary
from .runtime import METHOD_LABELS, clear_memory, get_device, ordered_methods, setup_matplotlib
from .training import history_rows, summary_row, train_one_run


def build_local_grid(config: dict) -> list[tuple[str, dict]]:
    """Create method/run-config pairs from a documented local grid."""
    items: list[tuple[str, dict]] = []
    for method in ordered_methods(config["methods"]):
        method_grid = config["grid_search"]["local_grid"][method]
        for lr in method_grid["lr"]:
            if method == "bp":
                items.append((method, {"lr": float(lr)}))
                continue
            for sigma in method_grid["sigma"]:
                items.append((method, {"lr": float(lr), "sigma": float(sigma)}))
    return items


def run_training_grid(config: dict, grid_items: list[tuple[str, dict]], epochs: int, seeds: list[int], device=None) -> dict:
    """Run a list of training configurations and aggregate their clean evaluation metrics."""
    device = get_device() if device is None else device
    data = load_task_data(config, device)
    score_key = "acc" if config["task_type"] == "classification" else "r2"
    original_epochs = config["run_epochs"]
    original_print_every = config.get("run_print_every_epoch", None)
    config["run_epochs"] = int(epochs)
    config["run_print_every_epoch"] = int(config["grid_search"].get("print_every_epoch", max(1, epochs // 5)))

    all_history_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    print(
        f"Dataset: train={data['train_size']}, train_eval={data.get('train_eval_size', data['train_size'])}, "
        f"test={data['test_size']} | grid runs={len(grid_items)} | epochs={epochs} | device={device}"
    )

    try:
        for seed_index, seed in enumerate(seeds, start=1):
            print(f"\n===== Grid seed {seed_index}/{len(seeds)}: {seed} =====")
            for run_index, (method, run_config) in enumerate(grid_items, start=1):
                sigma_str = f", sigma={run_config['sigma']:.6g}" if "sigma" in run_config else ""
                print(f"\n--- Grid run {run_index}/{len(grid_items)} | {METHOD_LABELS[method]} | lr={run_config['lr']:.6g}{sigma_str} ---")
                result = train_one_run(method, run_config, data, config, seed, device)
                history = result["history"]
                all_history_rows.extend(history_rows(config["task_key"], method, seed, run_config, history, score_key))
                all_summary_rows.append(summary_row(config["task_key"], method, seed, run_config, history, result, score_key))
                clear_memory()
    finally:
        config["run_epochs"] = original_epochs
        if original_print_every is None:
            config.pop("run_print_every_epoch", None)
        else:
            config["run_print_every_epoch"] = original_print_every

    history_df = pd.DataFrame(all_history_rows)
    summary_seed_df = pd.DataFrame(all_summary_rows)
    grid_summary_df = aggregate_grid_summary(summary_seed_df)
    return {
        "config": config,
        "data": data,
        "history_df": history_df,
        "summary_seed_df": summary_seed_df,
        "grid_summary_df": grid_summary_df,
    }


def run_local_grid_search(config: dict, project_root: Path | None = None, show: bool = True, device=None) -> dict:
    """Run the documented 3x3 local grid search."""
    del project_root
    setup_matplotlib()
    grid_items = build_local_grid(config)
    outputs = run_training_grid(
        config,
        grid_items=grid_items,
        epochs=int(config["grid_search"]["epochs"]),
        seeds=list(config["grid_search"].get("seeds", [0])),
        device=device,
    )
    if show:
        try:
            from IPython.display import display

            display(outputs["grid_summary_df"])
        except Exception:
            print(outputs["grid_summary_df"].to_string(index=False))
    return outputs


def run_full_length_training(config: dict, run_configs: dict[str, dict], project_root: Path | None = None, show: bool = True, device=None) -> dict:
    """Run an editable full-length sweep with user-provided hyperparameters."""
    del project_root
    setup_matplotlib()
    grid_items = [(method, run_configs[method]) for method in ordered_methods([method for method in run_configs if method in config["methods"]])]
    outputs = run_training_grid(
        config,
        grid_items=grid_items,
        epochs=int(config.get("full_run_epochs", config["run_epochs"])),
        seeds=list(config.get("full_run_seeds", [0])),
        device=device,
    )
    if show:
        try:
            from IPython.display import display

            display(outputs["grid_summary_df"])
        except Exception:
            print(outputs["grid_summary_df"].to_string(index=False))
    return outputs


def best_grid_rows(grid_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return one lowest-loss non-diverged row per method."""
    rows = []
    for method, method_df in grid_summary_df.groupby("method"):
        sorted_df = method_df.sort_values(
            ["diverged_runs", "best_test_loss_mean", "best_test_score_mean"],
            ascending=[True, True, False],
        )
        rows.append(sorted_df.iloc[0])
    return pd.DataFrame(rows).sort_values("best_test_loss_mean")


def cartesian_grid(lr_values: list[float], sigma_values: list[float] | None = None) -> list[dict]:
    if sigma_values is None:
        return [{"lr": float(lr)} for lr in lr_values]
    return [{"lr": float(lr), "sigma": float(sigma)} for lr, sigma in product(lr_values, sigma_values)]
