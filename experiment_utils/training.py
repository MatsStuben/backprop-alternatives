from __future__ import annotations

import math
import time

import torch

from learning_rules_MLP import (
    MLP,
    backprop_step,
    node_perturbation_step,
    node_perturbation_step_fan_in_scaled,
    node_perturbation_step_fixed_sigma,
    weight_perturb_step,
)

from .evaluation import best_history_value, convergence_epoch_from_history, evaluate_deterministic_model
from .runtime import METHOD_LABELS, clear_memory, set_seed


def make_model(config: dict, require_grad: bool, device: torch.device) -> MLP:
    activation_name = config.get("activation", "sigmoid")
    activation = {"sigmoid": torch.sigmoid, "relu": torch.relu}[activation_name]
    return MLP(config["dimensions"], activation=activation, output_activation=None, require_grad=require_grad).to(device)


def model_weight_norm(model: MLP) -> float:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        total = total + layer.weight.detach().pow(2).sum()
    return float(torch.sqrt(total))


def run_training_step(method: str, model: MLP, xb: torch.Tensor, yb: torch.Tensor, run_config: dict, optimizer) -> None:
    if method == "bp":
        backprop_step(model, xb, yb, optimizer=optimizer)
    elif method == "np":
        node_perturbation_step(model, xb, yb, eta=run_config["lr"], sigma=run_config["sigma"])
    elif method == "np_fan_in":
        node_perturbation_step_fan_in_scaled(model, xb, yb, eta=run_config["lr"], sigma=run_config["sigma"])
    elif method == "np_fixed":
        node_perturbation_step_fixed_sigma(model, xb, yb, eta=run_config["lr"], sigma=run_config["sigma"])
    elif method == "wp":
        weight_perturb_step(model, xb, yb, eta=run_config["lr"], sigma=run_config["sigma"])
    else:
        raise ValueError(f"Unknown method: {method}")


def build_initial_state_dict(config: dict, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    model = make_model(config, require_grad=True, device=device)
    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    del model
    clear_memory()
    return state_dict


def train_one_run(
    method: str,
    run_config: dict,
    data: dict,
    config: dict,
    seed: int,
    device: torch.device,
) -> dict:
    base_state = build_initial_state_dict(config, seed, device)
    model = make_model(config, require_grad=(method == "bp"), device=device)
    model.load_state_dict(base_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=run_config["lr"]) if method == "bp" else None

    task_type = config["task_type"]
    score_key = "acc" if task_type == "classification" else "r2"
    history = {"epoch": [], "train_loss": [], f"train_{score_key}": [], "test_loss": [], f"test_{score_key}": [], "weight_norm": []}
    best_test_loss = float("inf")
    best_test_score = -float("inf")
    start_time = time.time()
    diverged = False

    for epoch in range(1, config["run_epochs"] + 1):
        model.train()
        for batch in data["train_loader"]:
            xb = batch[0].to(device, non_blocking=True)
            yb = batch[1].to(device, non_blocking=True)
            run_training_step(method, model, xb, yb, run_config, optimizer)

        train_loss, train_score = evaluate_deterministic_model(model, data["train_eval_loader"], task_type, device)
        test_loss, test_score = evaluate_deterministic_model(model, data["test_loader"], task_type, device)
        best_test_loss = min(best_test_loss, test_loss)
        best_test_score = max(best_test_score, test_score)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history[f"train_{score_key}"].append(train_score)
        history["test_loss"].append(test_loss)
        history[f"test_{score_key}"].append(test_score)
        history["weight_norm"].append(model_weight_norm(model))

        if not math.isfinite(train_loss) or not math.isfinite(test_loss) or test_loss > config.get("divergence_loss_threshold", 5.0):
            diverged = True
            print(f"    diverged at epoch {epoch}/{config['run_epochs']} | train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")
            break

        print_every = config.get("run_print_every_epoch", 25)
        if epoch == 1 or epoch == config["run_epochs"] or epoch % print_every == 0:
            sigma_str = f", sigma={run_config['sigma']:.4g}" if "sigma" in run_config else ""
            print(
                f"    epoch {epoch:4d}/{config['run_epochs']} | {method} | "
                f"lr={run_config['lr']:.4g}{sigma_str} | train_loss={train_loss:.4f}, "
                f"test_loss={test_loss:.4f}, test_score={test_score:.4f}"
            )

    result = {
        "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "history": history,
        "best_test_loss": best_test_loss,
        "best_test_score": best_test_score,
        "diverged": diverged,
        "elapsed_sec": time.time() - start_time,
    }
    del model
    clear_memory()
    return result


def history_rows(task_key: str, method: str, seed: int, run_config: dict, history: dict, score_key: str) -> list[dict]:
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
                "sigma": float(run_config["sigma"]) if "sigma" in run_config else float("nan"),
                "train_loss": float(history["train_loss"][index]),
                "test_loss": float(history["test_loss"][index]),
                "train_score": float(history[f"train_{score_key}"][index]),
                "test_score": float(history[f"test_{score_key}"][index]),
            }
        )
    return rows


def summary_row(task_key: str, method: str, seed: int, run_config: dict, history: dict, result: dict, score_key: str) -> dict:
    final_index = -1 if history.get("epoch") else None
    return {
        "task": task_key,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "lr": float(run_config["lr"]),
        "sigma": float(run_config["sigma"]) if "sigma" in run_config else float("nan"),
        "score_metric": "Accuracy" if score_key == "acc" else "R2",
        "best_train_loss": best_history_value(history, "train_loss", min),
        "best_test_loss": best_history_value(history, "test_loss", min),
        "best_train_score": best_history_value(history, f"train_{score_key}", max),
        "best_test_score": best_history_value(history, f"test_{score_key}", max),
        "final_train_loss": float(history["train_loss"][final_index]) if final_index is not None else float("nan"),
        "final_test_loss": float(history["test_loss"][final_index]) if final_index is not None else float("nan"),
        "final_train_score": float(history[f"train_{score_key}"][final_index]) if final_index is not None else float("nan"),
        "final_test_score": float(history[f"test_{score_key}"][final_index]) if final_index is not None else float("nan"),
        "convergence_epoch": convergence_epoch_from_history(history),
        "diverged": bool(result["diverged"]),
        "elapsed_sec": float(result["elapsed_sec"]),
    }
