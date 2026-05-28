"""Shared utilities for the thesis hyperparameter-search notebooks.

The notebooks in this directory define task-specific configurations and call
the functions in this module.  The actual model and learning-rule updates are
imported through ``final_config_utils.py``, which in turn imports the trusted
implementations from ``learning_rules_MLP.py``.
"""

from __future__ import annotations

import itertools
import math
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG_DIR = PROJECT_ROOT / "final-config-runs"
for path in [PROJECT_ROOT, FINAL_CONFIG_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import final_config_utils as fcu  # noqa: E402


PERTURBATION_METHODS = ["np", "np_fan_in", "np_fixed", "wp"]
METHOD_ORDER = ["bp", "np", "np_fan_in", "np_fixed", "wp"]
METHOD_LABELS = fcu.METHOD_LABELS
DEFAULT_TOP_K = 10


def build_grid(parameter_grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Return a list of dictionaries from a Cartesian product parameter grid."""
    keys = list(parameter_grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(parameter_grid[key] for key in keys))]


def output_paths(project_root: Path, config: dict) -> dict[str, Path]:
    root = project_root / config["output_dir"]
    paths = {
        "root": root,
        "data": root / "data",
        "tables": root / "tables",
        "archives": root / "archives",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _copy_config_with_epochs(config: dict, epochs: int, print_every: int | None = None) -> dict:
    run_config = dict(config)
    run_config["run_epochs"] = int(epochs)
    if print_every is not None:
        run_config["run_print_every_epoch"] = int(print_every)
    return run_config


def _score_key(config: dict) -> str:
    return "acc" if config["task_type"] == "classification" else "r2"


def _best_value(history: dict, key: str, reducer) -> float:
    values = [float(value) for value in history.get(key, []) if pd.notna(value)]
    return float(reducer(values)) if values else np.nan


def _history_rows(task_key: str, method: str, seed: int, run_config: dict, history: dict, score_key: str) -> list[dict]:
    rows = []
    for index, epoch in enumerate(history["epoch"]):
        rows.append(
            {
                "task": task_key,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "epoch": int(epoch),
                "lr": float(run_config["lr"]),
                "sigma": float(run_config["sigma"]) if "sigma" in run_config else np.nan,
                "train_loss": float(history["train_loss"][index]),
                "test_loss": float(history["test_loss"][index]),
                "train_score": float(history[f"train_{score_key}"][index]),
                "test_score": float(history[f"test_{score_key}"][index]),
            }
        )
    return rows


def _summary_row(task_key: str, method: str, seed: int, run_config: dict, history: dict, result: dict, score_key: str) -> dict:
    final_index = -1 if history.get("epoch") else None
    return {
        "task": task_key,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "lr": float(run_config["lr"]),
        "sigma": float(run_config["sigma"]) if "sigma" in run_config else np.nan,
        "best_train_loss": _best_value(history, "train_loss", min),
        "best_test_loss": _best_value(history, "test_loss", min),
        "best_train_score": _best_value(history, f"train_{score_key}", max),
        "best_test_score": _best_value(history, f"test_{score_key}", max),
        "final_train_loss": float(history["train_loss"][final_index]) if final_index is not None else np.nan,
        "final_test_loss": float(history["test_loss"][final_index]) if final_index is not None else np.nan,
        "final_train_score": float(history[f"train_{score_key}"][final_index]) if final_index is not None else np.nan,
        "final_test_score": float(history[f"test_{score_key}"][final_index]) if final_index is not None else np.nan,
        "convergence_epoch": fcu.convergence_epoch_from_history(history),
        "diverged": bool(result["diverged"]),
        "elapsed_sec": float(result["elapsed_sec"]),
    }


def _aggregate_grid_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df.groupby(["task", "method", "method_label", "lr", "sigma"], dropna=False, as_index=False)
        .agg(
            best_train_loss_mean=("best_train_loss", "mean"),
            best_train_loss_std=("best_train_loss", "std"),
            best_test_loss_mean=("best_test_loss", "mean"),
            best_test_loss_std=("best_test_loss", "std"),
            best_train_score_mean=("best_train_score", "mean"),
            best_train_score_std=("best_train_score", "std"),
            best_test_score_mean=("best_test_score", "mean"),
            best_test_score_std=("best_test_score", "std"),
            final_test_loss_mean=("final_test_loss", "mean"),
            final_test_loss_std=("final_test_loss", "std"),
            convergence_epoch_mean=("convergence_epoch", "mean"),
            convergence_epoch_std=("convergence_epoch", "std"),
            diverged_runs=("diverged", "sum"),
            elapsed_sec_mean=("elapsed_sec", "mean"),
            num_seeds=("seed", "nunique"),
        )
        .sort_values(["method", "best_test_loss_mean", "best_test_score_mean"], ascending=[True, True, False])
    )


def _select_best_per_method(grid_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        method_df = grid_summary_df[grid_summary_df["method"] == method].copy()
        if method_df.empty:
            continue
        method_df = method_df.sort_values(["diverged_runs", "best_test_loss_mean", "best_test_score_mean"], ascending=[True, True, False])
        rows.append(method_df.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def load_grid_search_data(config: dict, device: torch.device) -> dict:
    data = fcu.load_task_data(config, device)
    print(
        f"Dataset: train={data['train_size']}, train_eval={data.get('train_eval_size', data['train_size'])}, "
        f"test={data['test_size']}, dims={config['dimensions']}, batch_size={config['data_kwargs'].get('batch_size')}"
    )
    return data


def run_local_grid_search(config: dict, project_root: Path | None = None) -> dict:
    """Run the documented local 3x3 grid search."""
    fcu.setup_matplotlib()
    project_root = project_root or PROJECT_ROOT
    paths = output_paths(project_root, config)
    device = fcu.get_device()
    data = load_grid_search_data(config, device)
    search_config = config["grid_search"]
    epochs = int(search_config["epochs"])
    seeds = list(search_config.get("seeds", [0]))
    methods = list(search_config.get("methods", config["local_grids"].keys()))
    run_print_every = int(search_config.get("print_every_epoch", max(1, epochs // 4)))
    task_key = config["task_key"]
    score_key = _score_key(config)

    history_rows = []
    summary_rows = []
    start = time.time()

    for method in methods:
        grid = config["local_grids"][method]
        print(f"\n=== Local grid | {METHOD_LABELS[method]} | {len(grid)} configuration(s) ===")
        for grid_index, run_config in enumerate(grid, start=1):
            for seed in seeds:
                sigma_text = f", sigma={run_config['sigma']:.5g}" if "sigma" in run_config else ""
                print(
                    f"\n{METHOD_LABELS[method]} config {grid_index}/{len(grid)} | "
                    f"seed={seed} | lr={run_config['lr']:.5g}{sigma_text}"
                )
                train_config = _copy_config_with_epochs(config, epochs, run_print_every)
                result = fcu.train_one_run(method, run_config, data, train_config, seed, device)
                history = result["history"]
                history_rows.extend(_history_rows(task_key, method, seed, run_config, history, score_key))
                summary_rows.append(_summary_row(task_key, method, seed, run_config, history, result, score_key))

    history_df = pd.DataFrame(history_rows)
    summary_df = pd.DataFrame(summary_rows)
    grid_summary_df = _aggregate_grid_summary(summary_df)
    best_df = _select_best_per_method(grid_summary_df)

    history_df.to_csv(paths["data"] / f"{task_key}_local_grid_history.csv", index=False)
    summary_df.to_csv(paths["data"] / f"{task_key}_local_grid_by_seed.csv", index=False)
    grid_summary_df.to_csv(paths["tables"] / f"{task_key}_local_grid_summary.csv", index=False)
    best_df.to_csv(paths["tables"] / f"{task_key}_local_grid_best_by_method.csv", index=False)
    print(f"\nLocal grid search finished in {(time.time() - start) / 60:.1f} min.")
    return {
        "data": data,
        "history_df": history_df,
        "summary_df": summary_df,
        "grid_summary_df": grid_summary_df,
        "best_df": best_df,
        "paths": paths,
    }


def _sigma_search_loader(data: dict, batch_size: int, device: torch.device) -> DataLoader:
    dataset = TensorDataset(data["x_train"], data["y_train"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))


def _sigma_search_for_seed(config: dict, data: dict, seed: int, device: torch.device) -> pd.DataFrame:
    sigma_config = dict(config)
    sigma_config["analysis"] = dict(config["sigma_search"])
    sigma_config["analysis"]["method_sigmas"] = {
        method: values[len(values) // 2] for method, values in config["sigma_search"]["sigma_grids"].items()
    }
    checkpoint_states = fcu.train_backprop_checkpoint_states(data, sigma_config, seed, device)
    loader = _sigma_search_loader(data, config["sigma_search"]["batch_size"], device)
    rows = []
    num_perturbations = int(config["sigma_search"]["num_perturbations"])
    max_batches = config["sigma_search"].get("max_batches")

    for checkpoint_epoch, state_dict in checkpoint_states.items():
        model = fcu.make_model(config, require_grad=False, device=device)
        model.load_state_dict(state_dict)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        print(f"\nSigma diagnostics | seed={seed} | checkpoint={checkpoint_epoch}")
        for batch_index, (xb, yb) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            true_update = fcu._backprop_update_vector(model, xb, yb)

            for method, sigma_values in config["sigma_search"]["sigma_grids"].items():
                for sigma in sigma_values:
                    sum_estimate = torch.zeros_like(true_update)
                    sum_sample_cosine = 0.0
                    sum_sample_variance = 0.0
                    for _ in range(num_perturbations):
                        estimate = fcu._perturbation_update_vector(method, model, xb, yb, float(sigma))
                        sum_estimate += estimate
                        sum_sample_cosine += fcu.cosine_similarity(estimate, true_update)
                        sum_sample_variance += float((estimate - true_update).pow(2).mean())

                    mean_estimate = sum_estimate / num_perturbations
                    rows.append(
                        {
                            "seed": seed,
                            "checkpoint_epoch": int(checkpoint_epoch),
                            "batch_index": int(batch_index),
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "sigma": float(sigma),
                            "avg_sample_cosine": sum_sample_cosine / num_perturbations,
                            "mean_estimate_cosine": fcu.cosine_similarity(mean_estimate, true_update),
                            "sample_variance": sum_sample_variance / num_perturbations,
                            "batch_variance": float((mean_estimate - true_update).pow(2).mean()),
                        }
                    )

        del model
        fcu.clear_memory()

    return pd.DataFrame(rows)


def _summarize_sigma_search(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_summary = (
        batch_df.groupby(["seed", "checkpoint_epoch", "method", "method_label", "sigma"], as_index=False)
        .agg(
            cosine=("mean_estimate_cosine", "mean"),
            variance=("sample_variance", "mean"),
            batch_variance=("batch_variance", "mean"),
            num_batches=("batch_index", "nunique"),
        )
        .sort_values(["method", "sigma", "seed", "checkpoint_epoch"])
    )
    overall_summary = (
        checkpoint_summary.groupby(["method", "method_label", "sigma"], as_index=False)
        .agg(
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            variance_mean=("variance", "mean"),
            variance_std=("variance", "std"),
            batch_variance_mean=("batch_variance", "mean"),
            num_seeds=("seed", "nunique"),
            num_checkpoints=("checkpoint_epoch", "nunique"),
            num_batches=("num_batches", "mean"),
        )
        .sort_values(["method", "sigma"])
    )
    return checkpoint_summary, overall_summary


def run_sigma_search(config: dict, project_root: Path | None = None) -> dict:
    """Run frozen-backprop estimator diagnostics over five sigma values per method."""
    fcu.setup_matplotlib()
    project_root = project_root or PROJECT_ROOT
    paths = output_paths(project_root, config)
    device = fcu.get_device()
    data = load_grid_search_data(config, device)
    seeds = list(config["sigma_search"].get("seeds", [0]))
    task_key = config["task_key"]
    print("\nSigma grids:")
    for method, values in config["sigma_search"]["sigma_grids"].items():
        print(f"  {METHOD_LABELS[method]}: {values}")

    batch_parts = []
    for seed in seeds:
        print(f"\n===== Sigma search seed {seed} =====")
        batch_parts.append(_sigma_search_for_seed(config, data, seed, device))

    batch_df = pd.concat(batch_parts, ignore_index=True)
    checkpoint_summary_df, overall_summary_df = _summarize_sigma_search(batch_df)
    batch_df.to_csv(paths["data"] / f"{task_key}_sigma_search_batch_metrics.csv", index=False)
    checkpoint_summary_df.to_csv(paths["tables"] / f"{task_key}_sigma_search_checkpoint_summary.csv", index=False)
    overall_summary_df.to_csv(paths["tables"] / f"{task_key}_sigma_search_summary.csv", index=False)
    return {
        "data": data,
        "batch_df": batch_df,
        "checkpoint_summary_df": checkpoint_summary_df,
        "overall_summary_df": overall_summary_df,
        "paths": paths,
    }


def run_full_sweep(config: dict, run_configs: dict[str, dict], project_root: Path | None = None) -> dict:
    """Run selected hyperparameters for the normal full-training epoch count."""
    fcu.setup_matplotlib()
    project_root = project_root or PROJECT_ROOT
    paths = output_paths(project_root, config)
    device = fcu.get_device()
    data = load_grid_search_data(config, device)
    full_config = config["full_run"]
    epochs = int(full_config["epochs"])
    seeds = list(full_config.get("seeds", [0]))
    methods = [method for method in METHOD_ORDER if method in run_configs]
    run_print_every = int(full_config.get("print_every_epoch", max(1, epochs // 10)))
    task_key = config["task_key"]
    score_key = _score_key(config)
    history_rows = []
    summary_rows = []

    for method in methods:
        run_config = run_configs[method]
        for seed in seeds:
            sigma_text = f", sigma={run_config['sigma']:.5g}" if "sigma" in run_config else ""
            print(f"\nFull run | {METHOD_LABELS[method]} | seed={seed} | lr={run_config['lr']:.5g}{sigma_text}")
            train_config = _copy_config_with_epochs(config, epochs, run_print_every)
            result = fcu.train_one_run(method, run_config, data, train_config, seed, device)
            history = result["history"]
            history_rows.extend(_history_rows(task_key, method, seed, run_config, history, score_key))
            summary_rows.append(_summary_row(task_key, method, seed, run_config, history, result, score_key))

    history_df = pd.DataFrame(history_rows)
    summary_df = pd.DataFrame(summary_rows)
    run_summary_df = _aggregate_grid_summary(summary_df)
    history_df.to_csv(paths["data"] / f"{task_key}_full_run_history.csv", index=False)
    summary_df.to_csv(paths["data"] / f"{task_key}_full_run_by_seed.csv", index=False)
    run_summary_df.to_csv(paths["tables"] / f"{task_key}_full_run_summary.csv", index=False)
    return {"data": data, "history_df": history_df, "summary_df": summary_df, "run_summary_df": run_summary_df, "paths": paths}


def archive_grid_search_outputs(config: dict, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    paths = output_paths(project_root, config)
    archive_path = Path(shutil.make_archive(str(paths["root"]), "zip", root_dir=paths["root"]))
    print(f"Archive written to {archive_path}")
    return archive_path


def save_config_snapshot(config: dict, project_root: Path | None = None) -> Path:
    project_root = project_root or PROJECT_ROOT
    paths = output_paths(project_root, config)
    path = paths["data"] / f"{config['task_key']}_grid_search_config.pkl"
    with path.open("wb") as handle:
        pickle.dump(config, handle)
    return path


def top_grid_rows(grid_summary_df: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    return (
        grid_summary_df.sort_values(["method", "diverged_runs", "best_test_loss_mean", "best_test_score_mean"], ascending=[True, True, True, False])
        .groupby("method", group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
