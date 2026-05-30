from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learning_rules_MLP import MLP

from .runtime import CONVERGENCE_FRACTION


def evaluate_deterministic_model(model: MLP, loader: DataLoader, task_type: str, device: torch.device) -> tuple[float, float]:
    """Evaluate the unperturbed model on a full loader."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device, non_blocking=True)
            yb = batch[1].to(device, non_blocking=True)
            output = model(xb)
            loss = F.mse_loss(output, yb, reduction="mean")
            total_loss += loss.item() * xb.size(0)
            total_examples += xb.size(0)
            if task_type == "classification":
                labels = batch[2].to(device, non_blocking=True)
                total_correct += (output.argmax(dim=1) == labels).sum().item()
            else:
                predictions.append(output.detach().cpu())
                targets.append(yb.detach().cpu())

    mean_loss = total_loss / max(total_examples, 1)
    if task_type == "classification":
        score = total_correct / max(total_examples, 1)
    else:
        y_pred = torch.cat(predictions, dim=0)
        y_true = torch.cat(targets, dim=0)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(1e-12)
        score = float(1.0 - ss_res / ss_tot)
    return mean_loss, score


def convergence_epoch_from_history(history: dict, fraction: float = CONVERGENCE_FRACTION) -> float:
    pairs = [
        (int(epoch), float(loss))
        for epoch, loss in zip(history.get("epoch", []), history.get("test_loss", []))
        if np.isfinite(float(loss))
    ]
    if not pairs:
        return np.nan
    initial_loss = pairs[0][1]
    best_loss = min(loss for _, loss in pairs)
    improvement = initial_loss - best_loss
    if improvement <= 1e-12:
        return min(epoch for epoch, loss in pairs if loss == best_loss)
    threshold = initial_loss - fraction * improvement
    for epoch, loss in pairs:
        if loss <= threshold:
            return epoch
    return pairs[-1][0]


def best_history_value(history: dict, key: str, reducer) -> float:
    values = [float(value) for value in history.get(key, []) if pd.notna(value)]
    return float(reducer(values)) if values else np.nan


def aggregate_history(history_df: pd.DataFrame) -> pd.DataFrame:
    return (
        history_df.groupby(["method", "method_label", "epoch"], as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            train_score_mean=("train_score", "mean"),
            train_score_std=("train_score", "std"),
            test_score_mean=("test_score", "mean"),
            test_score_std=("test_score", "std"),
        )
        .sort_values(["method", "epoch"])
    )


def aggregate_diagnostics(diagnostics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_params = diagnostics_df[diagnostics_df["component"] == "all"].copy()
    checkpoint_seed = (
        all_params.groupby(["seed", "checkpoint_epoch", "method"], as_index=False)
        .agg(cosine=("mean_estimate_cosine", "mean"), variance=("sample_variance", "mean"), sigma=("sigma", "mean"))
        .sort_values(["seed", "checkpoint_epoch", "method"])
    )
    checkpoint_summary = (
        checkpoint_seed.groupby(["checkpoint_epoch", "method"], as_index=False)
        .agg(
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            variance_mean=("variance", "mean"),
            variance_std=("variance", "std"),
            sigma=("sigma", "mean"),
        )
        .sort_values(["checkpoint_epoch", "method"])
    )
    seed_over_checkpoints = (
        checkpoint_seed.groupby(["seed", "method"], as_index=False)
        .agg(cosine=("cosine", "mean"), variance=("variance", "mean"), sigma=("sigma", "mean"))
        .sort_values(["seed", "method"])
    )
    overall_summary = (
        seed_over_checkpoints.groupby("method", as_index=False)
        .agg(
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            variance_mean=("variance", "mean"),
            variance_std=("variance", "std"),
            sigma=("sigma", "mean"),
        )
        .sort_values("method")
    )
    return checkpoint_seed, checkpoint_summary, seed_over_checkpoints, overall_summary


def aggregate_results_table(summary_seed_df: pd.DataFrame, seed_over_checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_seed = summary_seed_df.merge(seed_over_checkpoints[["seed", "method", "cosine", "variance"]], on=["seed", "method"], how="left")
    table_seed.loc[table_seed["method"] == "bp", "cosine"] = 1.0
    table_seed.loc[table_seed["method"] == "bp", "variance"] = 0.0
    table_summary = (
        table_seed.groupby(["task", "method", "method_label", "score_metric"], as_index=False)
        .agg(
            best_train_loss_mean=("best_train_loss", "mean"),
            best_train_loss_std=("best_train_loss", "std"),
            best_test_loss_mean=("best_test_loss", "mean"),
            best_test_loss_std=("best_test_loss", "std"),
            best_train_score_mean=("best_train_score", "mean"),
            best_train_score_std=("best_train_score", "std"),
            best_test_score_mean=("best_test_score", "mean"),
            best_test_score_std=("best_test_score", "std"),
            convergence_epoch_mean=("convergence_epoch", "mean"),
            convergence_epoch_std=("convergence_epoch", "std"),
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            variance_mean=("variance", "mean"),
            variance_std=("variance", "std"),
        )
        .sort_values("method")
    )
    return table_seed, table_summary


def aggregate_grid_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
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
