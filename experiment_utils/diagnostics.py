from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from learning_rules_MLP import (
    backprop_step,
    node_perturbation_step,
    node_perturbation_step_fan_in_scaled,
    node_perturbation_step_fixed_sigma,
    weight_perturb_step,
)

from .evaluation import evaluate_deterministic_model
from .runtime import METHOD_LABELS, clear_memory
from .training import build_initial_state_dict, make_model


def train_backprop_checkpoint_states(data: dict, config: dict, seed: int, device: torch.device, analysis_key: str = "analysis") -> dict[int, dict]:
    analysis = config[analysis_key]
    checkpoint_epochs = sorted(set(int(epoch) for epoch in analysis["checkpoint_epochs"]))
    model = make_model(config, require_grad=True, device=device)
    model.load_state_dict(build_initial_state_dict(config, seed, device))
    optimizer = torch.optim.SGD(model.parameters(), lr=analysis["bp_lr"])
    checkpoint_states = {}

    for epoch in range(1, analysis["epochs"] + 1):
        model.train()
        for batch in data["train_loader"]:
            xb = batch[0].to(device, non_blocking=True)
            yb = batch[1].to(device, non_blocking=True)
            backprop_step(model, xb, yb, optimizer=optimizer)

        if epoch in checkpoint_epochs:
            checkpoint_states[epoch] = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        print_every = max(1, analysis["epochs"] // 5)
        if epoch == 1 or epoch == analysis["epochs"] or epoch in checkpoint_epochs or epoch % print_every == 0:
            train_loss, train_score = evaluate_deterministic_model(model, data["train_eval_loader"], config["task_type"], device)
            test_loss, test_score = evaluate_deterministic_model(model, data["test_loader"], config["task_type"], device)
            print(
                f"    BP diagnostic epoch {epoch:4d}/{analysis['epochs']} | "
                f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, test_score={test_score:.4f}"
            )

    del model
    clear_memory()
    return checkpoint_states


def backprop_update_vector(model, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    _, update = backprop_step(model, xb, yb, optimizer=optimizer, return_unscaled_parameter_update_vector=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return update.detach()


def perturbation_update_vector(method: str, model, xb: torch.Tensor, yb: torch.Tensor, sigma: float) -> torch.Tensor:
    step_functions = {
        "np": node_perturbation_step,
        "np_fan_in": node_perturbation_step_fan_in_scaled,
        "np_fixed": node_perturbation_step_fixed_sigma,
        "wp": weight_perturb_step,
    }
    with torch.no_grad():
        _, update = step_functions[method](
            model,
            xb,
            yb,
            eta=0.0,
            sigma=sigma,
            return_unscaled_parameter_update_vector=True,
        )
    return update.detach()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = torch.norm(a) * torch.norm(b)
    if float(denominator) < eps:
        return 0.0
    return float(torch.dot(a, b) / (denominator + eps))


def diagnostic_loader(data: dict, batch_size: int, device: torch.device) -> DataLoader:
    dataset = TensorDataset(data["x_train"], data["y_train"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))


def analyze_frozen_backprop_estimators(
    checkpoint_states: dict[int, dict],
    data: dict,
    config: dict,
    device: torch.device,
    analysis_key: str = "analysis",
    method_sigmas: dict[str, float] | None = None,
) -> pd.DataFrame:
    analysis = config[analysis_key]
    sigmas = dict(method_sigmas or analysis["method_sigmas"])
    loader = diagnostic_loader(data, analysis["batch_size"], device)
    max_batches = analysis.get("max_batches")
    rows = []

    for checkpoint_epoch, state_dict in checkpoint_states.items():
        model = make_model(config, require_grad=False, device=device)
        model.load_state_dict(state_dict)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        print(f"    diagnostics checkpoint {checkpoint_epoch}")
        for batch_index, (xb, yb) in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            true_update = backprop_update_vector(model, xb, yb)

            for method, sigma in sigmas.items():
                sum_estimate = torch.zeros_like(true_update)
                sum_sample_cosine = 0.0
                sum_sample_variance = 0.0
                for _ in range(analysis["num_perturbations"]):
                    estimate = perturbation_update_vector(method, model, xb, yb, float(sigma))
                    sum_estimate += estimate
                    sum_sample_cosine += cosine_similarity(estimate, true_update)
                    sum_sample_variance += float((estimate - true_update).pow(2).mean())

                mean_estimate = sum_estimate / analysis["num_perturbations"]
                rows.append(
                    {
                        "checkpoint_epoch": int(checkpoint_epoch),
                        "batch_index": int(batch_index),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "sigma": float(sigma),
                        "component": "all",
                        "component_label": "All parameters",
                        "layer_index": -1,
                        "avg_sample_cosine": sum_sample_cosine / analysis["num_perturbations"],
                        "mean_estimate_cosine": cosine_similarity(mean_estimate, true_update),
                        "sample_variance": sum_sample_variance / analysis["num_perturbations"],
                        "batch_variance": float((mean_estimate - true_update).pow(2).mean()),
                    }
                )

        del model
        clear_memory()

    return pd.DataFrame(rows)


def summarize_sigma_diagnostics(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
