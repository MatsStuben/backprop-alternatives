"""Shared utilities for the thesis final-configuration runs.

The notebooks in this directory are intentionally thin: they define one task
configuration and then call the functions in this file.  The learning rules
themselves are imported from ``learning_rules_MLP.py`` so the final runs use the
same implementation as the rest of the thesis code.
"""

from __future__ import annotations

import gc
import math
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning_rules_MLP import (  # noqa: E402
    MLP,
    backprop_step,
    node_perturbation_step,
    node_perturbation_step_fan_in_scaled,
    node_perturbation_step_fixed_sigma,
    weight_perturb_step,
)


METHOD_LABELS = {
    "bp": "BP",
    "np": "IS NP",
    "np_fan_in": "Fan-in NP",
    "np_fixed": "Vanilla NP",
    "wp": "WP",
}
METHOD_COLORS = {
    "bp": "#1f77b4",
    "np": "#ff7f0e",
    "np_fan_in": "#9467bd",
    "np_fixed": "#d62728",
    "wp": "#2ca02c",
}
PERTURBATION_METHODS = ["np", "np_fan_in", "np_fixed", "wp"]
VARIANCE_COLUMN = "sample_variance"
COSINE_COLUMN = "mean_estimate_cosine"
CONVERGENCE_FRACTION = 0.90
SUMMARY_SIGNIFICANT_FIGURES = 3


def setup_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "thesis_final_config_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_output_dirs(project_root: Path, output_dir_name: str) -> dict[str, Path]:
    output_dir = project_root / output_dir_name
    paths = {
        "output": output_dir,
        "figures": output_dir / "figures",
        "data": output_dir / "data",
        "tables": output_dir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _random_subset_indices(n: int, limit: int | None, seed: int) -> torch.Tensor:
    if limit is None or limit >= n:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)[:limit]


def synthetic_target(latent: torch.Tensor) -> torch.Tensor:
    input_dim = latent.shape[1]
    target = 0.6 * torch.sin(math.pi * latent).sum(dim=1, keepdim=True)
    target = target + 0.3 * latent.pow(2).sum(dim=1, keepdim=True)
    target = target + 0.2 * (latent[:, 0:1] * latent[:, 1:2])
    return target / math.sqrt(input_dim)


def load_synthetic_vector_regression_data(
    input_dim: int = 8,
    input_scale: float = 1.0,
    n_train: int = 512,
    n_test: int = 512,
    noise_std: float = 0.1,
    train_eval_limit: int | None = None,
    batch_size: int = 64,
    eval_batch_size: int = 512,
    seed: int = 0,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    latent_train = torch.rand(n_train, input_dim, generator=generator) * 2.0 - 1.0
    x_train = input_scale * latent_train
    y_train = synthetic_target(latent_train) + noise_std * torch.randn(n_train, 1, generator=generator)

    latent_test = torch.rand(n_test, input_dim, generator=generator) * 2.0 - 1.0
    x_test = input_scale * latent_test
    y_test = synthetic_target(latent_test)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_eval_dataset = _maybe_subset_dataset(train_dataset, train_eval_limit, seed + 1)
    return _make_data_dict(
        train_dataset,
        train_eval_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        num_workers,
        device,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        latent_train=latent_train,
        latent_test=latent_test,
        input_scale=input_scale,
    )


def patch_sklearn_urlretrieve_user_agent() -> None:
    """Avoid occasional HTTP 403 errors from sklearn's California Housing host."""
    import shutil as _shutil
    from urllib.request import Request as _Request
    from urllib.request import urlopen as _urlopen

    import sklearn.datasets._base as _sklearn_base

    def _urlretrieve_with_user_agent(url, filename=None, reporthook=None, data=None):
        request = _Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
        with _urlopen(request) as response:
            if filename is None:
                import tempfile as _tempfile

                handle = _tempfile.NamedTemporaryFile(delete=False)
                filename = handle.name
                handle.close()
            with open(filename, "wb") as out_file:
                _shutil.copyfileobj(response, out_file)
            return filename, response.info()

    _sklearn_base.urlretrieve = _urlretrieve_with_user_agent


def load_california_housing(
    test_size: float = 0.2,
    batch_size: int = 256,
    eval_batch_size: int = 4096,
    seed: int = 0,
    num_workers: int = 0,
    data_home: str | None = None,
    device: torch.device | None = None,
) -> dict:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    patch_sklearn_urlretrieve_user_agent()
    dataset = fetch_california_housing(data_home=data_home)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return _make_data_dict(
        train_dataset,
        train_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        num_workers,
        device,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def load_mnist(
    train_limit: int | None = None,
    test_limit: int | None = None,
    train_eval_limit: int | None = 4000,
    batch_size: int = 128,
    eval_batch_size: int = 1024,
    data_dir: str = "./data",
    seed: int = 0,
    mean_center_only: bool = True,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    from torchvision.datasets import MNIST

    train_dataset = MNIST(root=data_dir, train=True, download=True)
    test_dataset = MNIST(root=data_dir, train=False, download=True)
    x_train = train_dataset.data.float() / 255.0
    x_test = test_dataset.data.float() / 255.0
    train_labels = train_dataset.targets.long()
    test_labels = test_dataset.targets.long()
    return _prepare_image_data(
        x_train,
        train_labels,
        x_test,
        test_labels,
        num_classes=10,
        train_limit=train_limit,
        test_limit=test_limit,
        train_eval_limit=train_eval_limit,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        mean_center_only=mean_center_only,
        num_workers=num_workers,
        device=device,
        channel_first=False,
    )


def load_cifar10(
    train_limit: int | None = None,
    test_limit: int | None = None,
    train_eval_limit: int | None = 4000,
    batch_size: int = 128,
    eval_batch_size: int = 1024,
    data_dir: str = "./data",
    seed: int = 0,
    mean_center_only: bool = True,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    from torchvision.datasets import CIFAR10

    train_dataset = CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, download=True)
    x_train = torch.from_numpy(train_dataset.data).float() / 255.0
    x_test = torch.from_numpy(test_dataset.data).float() / 255.0
    train_labels = torch.tensor(train_dataset.targets, dtype=torch.long)
    test_labels = torch.tensor(test_dataset.targets, dtype=torch.long)
    return _prepare_image_data(
        x_train,
        train_labels,
        x_test,
        test_labels,
        num_classes=10,
        train_limit=train_limit,
        test_limit=test_limit,
        train_eval_limit=train_eval_limit,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        mean_center_only=mean_center_only,
        num_workers=num_workers,
        device=device,
        channel_first=True,
    )


def _prepare_image_data(
    x_train: torch.Tensor,
    train_labels: torch.Tensor,
    x_test: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    train_limit: int | None,
    test_limit: int | None,
    train_eval_limit: int | None,
    batch_size: int,
    eval_batch_size: int,
    seed: int,
    mean_center_only: bool,
    num_workers: int,
    device: torch.device | None,
    channel_first: bool,
) -> dict:
    train_idx = _random_subset_indices(len(x_train), train_limit, seed)
    test_idx = _random_subset_indices(len(x_test), test_limit, seed + 1)
    x_train = x_train[train_idx]
    train_labels = train_labels[train_idx]
    x_test = x_test[test_idx]
    test_labels = test_labels[test_idx]

    mean = x_train.mean()
    if mean_center_only:
        x_train = x_train - mean
        x_test = x_test - mean
    else:
        std = x_train.std().clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    if channel_first:
        x_train = x_train.permute(0, 3, 1, 2).contiguous()
        x_test = x_test.permute(0, 3, 1, 2).contiguous()
    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)

    y_train = F.one_hot(train_labels, num_classes=num_classes).float()
    y_test = F.one_hot(test_labels, num_classes=num_classes).float()
    train_dataset = TensorDataset(x_train, y_train, train_labels)
    test_dataset = TensorDataset(x_test, y_test, test_labels)
    train_eval_dataset = _maybe_subset_dataset(train_dataset, train_eval_limit, seed + 2)
    return _make_data_dict(
        train_dataset,
        train_eval_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        num_workers,
        device,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        train_labels=train_labels,
        test_labels=test_labels,
    )


def _maybe_subset_dataset(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = _random_subset_indices(len(dataset), limit, seed)
    return Subset(dataset, indices.tolist())


def _make_data_dict(
    train_dataset,
    train_eval_dataset,
    test_dataset,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device | None,
    **tensors,
) -> dict:
    pin_memory = device is not None and device.type == "cuda"
    return {
        "train_loader": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "train_eval_loader": DataLoader(
            train_eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test_loader": DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "train_size": len(train_dataset),
        "train_eval_size": len(train_eval_dataset),
        "test_size": len(test_dataset),
        **tensors,
    }


DATA_LOADERS = {
    "load_synthetic_vector_regression_data": load_synthetic_vector_regression_data,
    "load_california_housing": load_california_housing,
    "load_mnist": load_mnist,
    "load_cifar10": load_cifar10,
}


def load_task_data(config: dict, device: torch.device) -> dict:
    loader = DATA_LOADERS[config["data_loader"]]
    kwargs = dict(config["data_kwargs"])
    kwargs["device"] = device
    return loader(**kwargs)


def make_model(config: dict, require_grad: bool, device: torch.device) -> MLP:
    activation_name = config.get("activation", "sigmoid")
    activation = {"sigmoid": torch.sigmoid, "relu": torch.relu}[activation_name]
    return MLP(config["dimensions"], activation=activation, output_activation=None, require_grad=require_grad).to(device)


def model_weight_norm(model: MLP) -> float:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.layers:
        total = total + layer.weight.detach().pow(2).sum()
    return float(torch.sqrt(total))


def evaluate_model(model: MLP, loader: DataLoader, task_type: str, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
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


def _run_training_step(method: str, model: MLP, xb: torch.Tensor, yb: torch.Tensor, run_config: dict, optimizer) -> None:
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
            _run_training_step(method, model, xb, yb, run_config, optimizer)

        train_loss, train_score = evaluate_model(model, data["train_eval_loader"], task_type, device)
        test_loss, test_score = evaluate_model(model, data["test_loader"], task_type, device)
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


def _best_history_value(history: dict, key: str, reducer) -> float:
    values = [float(value) for value in history.get(key, []) if pd.notna(value)]
    return float(reducer(values)) if values else np.nan


def train_backprop_checkpoint_states(data: dict, config: dict, seed: int, device: torch.device) -> dict[int, dict]:
    analysis = config["analysis"]
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
            train_loss, train_score = evaluate_model(model, data["train_eval_loader"], config["task_type"], device)
            test_loss, test_score = evaluate_model(model, data["test_loader"], config["task_type"], device)
            print(
                f"    BP diagnostic epoch {epoch:4d}/{analysis['epochs']} | "
                f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, test_score={test_score:.4f}"
            )

    del model
    clear_memory()
    return checkpoint_states


def _backprop_update_vector(model: MLP, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    _, update = backprop_step(model, xb, yb, optimizer=optimizer, return_unscaled_parameter_update_vector=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return update.detach()


def _perturbation_update_vector(method: str, model: MLP, xb: torch.Tensor, yb: torch.Tensor, sigma: float) -> torch.Tensor:
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


def analyze_frozen_backprop_estimators(checkpoint_states: dict[int, dict], data: dict, config: dict, device: torch.device) -> pd.DataFrame:
    analysis = config["analysis"]
    dataset = TensorDataset(data["x_train"], data["y_train"])
    loader = DataLoader(
        dataset,
        batch_size=analysis["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    rows = []

    for checkpoint_epoch, state_dict in checkpoint_states.items():
        model = make_model(config, require_grad=False, device=device)
        model.load_state_dict(state_dict)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        print(f"    diagnostics checkpoint {checkpoint_epoch}")
        for batch_index, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            true_update = _backprop_update_vector(model, xb, yb)

            for method, sigma in analysis["method_sigmas"].items():
                sum_estimate = torch.zeros_like(true_update)
                sum_sample_cosine = 0.0
                sum_sample_variance = 0.0
                for _ in range(analysis["num_perturbations"]):
                    estimate = _perturbation_update_vector(method, model, xb, yb, sigma)
                    sum_estimate += estimate
                    sum_sample_cosine += cosine_similarity(estimate, true_update)
                    sum_sample_variance += float((estimate - true_update).pow(2).mean())

                mean_estimate = sum_estimate / analysis["num_perturbations"]
                rows.append(
                    {
                        "checkpoint_epoch": checkpoint_epoch,
                        "batch_index": batch_index,
                        "method": method,
                        "sigma": sigma,
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


def run_task(config: dict, device: torch.device) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_task_data(config, device)
    print(
        f"Dataset: train={data['train_size']}, train_eval={data['train_eval_size']}, "
        f"test={data['test_size']}, dims={config['dimensions']}, device={device}"
    )

    history_rows = []
    summary_seed_rows = []
    diagnostics = []
    score_key = "acc" if config["task_type"] == "classification" else "r2"

    for seed_index, seed in enumerate(config["seeds"], start=1):
        print(f"\n===== Seed {seed_index}/{len(config['seeds'])}: {seed} =====")
        for method_index, method in enumerate(config["methods"], start=1):
            run_config = config["run_configs"][method]
            sigma_str = f", sigma={run_config['sigma']:.4g}" if "sigma" in run_config else ""
            print(
                f"\n[{seed_index}/{len(config['seeds'])}] Training {METHOD_LABELS[method]} "
                f"({method_index}/{len(config['methods'])}) | lr={run_config['lr']:.4g}{sigma_str}"
            )
            result = train_one_run(method, run_config, data, config, seed, device)
            history = result["history"]
            for row_index, epoch in enumerate(history["epoch"]):
                history_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "epoch": epoch,
                        "train_loss": history["train_loss"][row_index],
                        "test_loss": history["test_loss"][row_index],
                        "train_score": history[f"train_{score_key}"][row_index],
                        "test_score": history[f"test_{score_key}"][row_index],
                    }
                )

            summary_seed_rows.append(
                {
                    "seed": seed,
                    "task": config["display_name"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "score_metric": "Accuracy" if config["task_type"] == "classification" else "R2",
                    "best_train_loss": _best_history_value(history, "train_loss", min),
                    "best_test_loss": _best_history_value(history, "test_loss", min),
                    "best_train_score": _best_history_value(history, f"train_{score_key}", max),
                    "best_test_score": _best_history_value(history, f"test_{score_key}", max),
                    "convergence_epoch": convergence_epoch_from_history(history),
                    "diverged": result["diverged"],
                    "elapsed_sec": result["elapsed_sec"],
                }
            )

        print(
            f"\n[{seed_index}/{len(config['seeds'])}] Diagnostics | "
            f"checkpoints={config['analysis']['checkpoint_epochs']}, "
            f"perturbations={config['analysis']['num_perturbations']}"
        )
        checkpoint_states = train_backprop_checkpoint_states(data, config, seed, device)
        diagnostics_df = analyze_frozen_backprop_estimators(checkpoint_states, data, config, device)
        diagnostics_df.insert(0, "seed", seed)
        diagnostics.append(diagnostics_df)

    return (
        data,
        pd.DataFrame(history_rows),
        pd.DataFrame(summary_seed_rows),
        pd.concat(diagnostics, ignore_index=True),
    )


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
    all_layers = diagnostics_df[diagnostics_df["component"] == "all"].copy()
    checkpoint_seed = (
        all_layers.groupby(["seed", "checkpoint_epoch", "method"], as_index=False)
        .agg(cosine=(COSINE_COLUMN, "mean"), variance=(VARIANCE_COLUMN, "mean"), sigma=("sigma", "mean"))
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


def aggregate_table(summary_seed_df: pd.DataFrame, seed_over_checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_seed = summary_seed_df.merge(
        seed_over_checkpoints[["seed", "method", "cosine", "variance"]],
        on=["seed", "method"],
        how="left",
    )
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


def save_pdf(fig, figure_dir: Path, filename: str) -> Path:
    path = figure_dir / filename
    fig.savefig(path, format="pdf", bbox_inches="tight")
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        pass
    plt.close(fig)
    return path


def _ordered_methods(methods: list[str]) -> list[str]:
    return [method for method in ["bp", "np", "np_fan_in", "np_fixed", "wp"] if method in methods]


def add_curve_legend(ax, methods: list[str]) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=METHOD_COLORS[method], lw=1.5, label=METHOD_LABELS[method]) for method in methods]
    handles.extend(
        [
            Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="Training"),
            Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Test"),
        ]
    )
    ax.legend(handles=handles, ncol=2, frameon=True, framealpha=0.95)


def plot_loss_curves(config: dict, history_df: pd.DataFrame, figure_dir: Path) -> Path:
    suffix = result_suffix(config)
    grouped = history_df.groupby(["method", "epoch"], as_index=False).agg(train_loss=("train_loss", "mean"), test_loss=("test_loss", "mean"))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for method in _ordered_methods(config["methods"]):
        method_df = grouped[grouped["method"] == method].sort_values("epoch")
        ax.plot(method_df["epoch"], method_df["train_loss"], color=METHOD_COLORS[method], lw=1.15, linestyle="-")
        ax.plot(method_df["epoch"], method_df["test_loss"], color=METHOD_COLORS[method], lw=1.15, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    add_curve_legend(ax, _ordered_methods(config["methods"]))
    fig.tight_layout()
    return save_pdf(fig, figure_dir, f"{config['task_key']}_loss_{suffix}.pdf")


def plot_score_curves(config: dict, history_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    if config["task_type"] != "classification":
        return None
    suffix = result_suffix(config)
    grouped = history_df.groupby(["method", "epoch"], as_index=False).agg(train_score=("train_score", "mean"), test_score=("test_score", "mean"))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for method in _ordered_methods(config["methods"]):
        method_df = grouped[grouped["method"] == method].sort_values("epoch")
        ax.plot(method_df["epoch"], method_df["train_score"], color=METHOD_COLORS[method], lw=1.15, linestyle="-")
        ax.plot(method_df["epoch"], method_df["test_score"], color=METHOD_COLORS[method], lw=1.15, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    add_curve_legend(ax, _ordered_methods(config["methods"]))
    fig.tight_layout()
    return save_pdf(fig, figure_dir, f"{config['task_key']}_accuracy_{suffix}.pdf")


def cosine_axis_limits(values: np.ndarray, padding_fraction: float = 0.15) -> tuple[float, float]:
    finite_values = np.asarray([float(value) for value in values if pd.notna(value) and np.isfinite(float(value))])
    if finite_values.size == 0:
        return 0.0, 1.0
    min_value = float(finite_values.min())
    max_value = float(finite_values.max())
    if min_value >= 0.0:
        upper = max_value * (1.0 + padding_fraction) if max_value > 0.0 else 0.05
        return 0.0, min(1.0, upper)
    value_range = max_value - min_value
    padding = value_range * padding_fraction if value_range > 0.0 else 0.05
    return max(-1.0, min_value - padding), min(1.0, max_value + padding)


def plot_diagnostic_bar(summary_df: pd.DataFrame, value_mean_col: str, ylabel: str, filename: str, figure_dir: Path, log_scale: bool = False) -> Path:
    plot_df = summary_df.set_index("method").loc[PERTURBATION_METHODS].reset_index()
    values = plot_df[value_mean_col].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(plot_df))
    for index, row in plot_df.iterrows():
        method = row["method"]
        ax.bar(x[index], row[value_mean_col], color=METHOD_COLORS[method], width=0.72, label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in plot_df["method"]], rotation=20, ha="right")
    ax.set_ylabel(f"{ylabel} (log scale)" if log_scale else ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)
    if log_scale:
        positive = values[values > 0]
        ax.set_yscale("log")
        ax.set_ylim(float(positive.min()) / 3.0, float(positive.max()) * 3.0)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    else:
        ax.set_ylim(*cosine_axis_limits(values))
    fig.tight_layout()
    return save_pdf(fig, figure_dir, filename)


def plot_all_results(config: dict, history_df: pd.DataFrame, diagnostic_summary_df: pd.DataFrame, checkpoint_summary_df: pd.DataFrame, figure_dir: Path) -> list[Path]:
    task_key = config["task_key"]
    suffix = result_suffix(config)
    paths = [plot_loss_curves(config, history_df, figure_dir)]
    score_path = plot_score_curves(config, history_df, figure_dir)
    if score_path is not None:
        paths.append(score_path)
    paths.append(plot_diagnostic_bar(diagnostic_summary_df, "cosine_mean", "Cosine similarity", f"{task_key}_cosine_{suffix}.pdf", figure_dir))
    paths.append(plot_diagnostic_bar(diagnostic_summary_df, "variance_mean", "Estimator variance", f"{task_key}_variance_{suffix}.pdf", figure_dir, log_scale=True))
    for checkpoint_epoch in sorted(checkpoint_summary_df["checkpoint_epoch"].unique()):
        checkpoint_df = checkpoint_summary_df[checkpoint_summary_df["checkpoint_epoch"] == checkpoint_epoch]
        paths.append(
            plot_diagnostic_bar(
                checkpoint_df,
                "cosine_mean",
                "Cosine similarity",
                f"{task_key}_cosine_checkpoint_{int(checkpoint_epoch):03d}_{suffix}.pdf",
                figure_dir,
            )
        )
        paths.append(
            plot_diagnostic_bar(
                checkpoint_df,
                "variance_mean",
                "Estimator variance",
                f"{task_key}_variance_checkpoint_{int(checkpoint_epoch):03d}_{suffix}.pdf",
                figure_dir,
                log_scale=True,
            )
        )
    return paths


def _format_scientific_sigfigs(value: float, significant_figures: int = SUMMARY_SIGNIFICANT_FIGURES) -> str:
    mantissa, exponent = f"{value:.{significant_figures - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_summary_number(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value == 0.0:
        return "0.00"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return _format_scientific_sigfigs(value)
    exponent = int(np.floor(np.log10(abs_value)))
    decimals = SUMMARY_SIGNIFICANT_FIGURES - exponent - 1
    rounded_value = round(value, decimals)
    rounded_abs = abs(rounded_value)
    if rounded_abs == 0.0:
        return "0.00"
    if rounded_abs < 1e-3 or rounded_abs >= 1e4:
        return _format_scientific_sigfigs(rounded_value)
    rounded_exponent = int(np.floor(np.log10(rounded_abs)))
    rounded_decimals = max(0, SUMMARY_SIGNIFICANT_FIGURES - rounded_exponent - 1)
    return f"{rounded_value:.{rounded_decimals}f}"


def format_pm(mean: float, std: float, is_epoch: bool = False) -> str:
    if pd.isna(mean):
        return ""
    if is_epoch:
        if pd.isna(std):
            return str(int(round(float(mean))))
        return f"{int(round(float(mean)))} $\\pm$ {int(round(float(std)))}"
    if pd.isna(std):
        return format_summary_number(mean)
    return f"{format_summary_number(mean)} $\\pm$ {format_summary_number(std)}"


def format_results_table(table_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in table_summary_df.iterrows():
        rows.append(
            {
                "Method": row["method_label"],
                "Best train loss": format_pm(row["best_train_loss_mean"], row["best_train_loss_std"]),
                "Best test loss": format_pm(row["best_test_loss_mean"], row["best_test_loss_std"]),
                "Best train score": format_pm(row["best_train_score_mean"], row["best_train_score_std"]),
                "Best test score": format_pm(row["best_test_score_mean"], row["best_test_score_std"]),
                "Convergence epoch": format_pm(row["convergence_epoch_mean"], row["convergence_epoch_std"], is_epoch=True),
                "Cosine": format_pm(row["cosine_mean"], row["cosine_std"]),
                "Variance": format_pm(row["variance_mean"], row["variance_std"]),
            }
        )
    return pd.DataFrame(rows)


def result_suffix(config: dict) -> str:
    return f"{len(config['seeds'])}seed"


def save_outputs(
    config: dict,
    output_paths: dict[str, Path],
    data: dict,
    history_df: pd.DataFrame,
    summary_seed_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    checkpoint_seed_df: pd.DataFrame,
    checkpoint_summary_df: pd.DataFrame,
    diagnostic_seed_summary_df: pd.DataFrame,
    diagnostic_summary_df: pd.DataFrame,
    table_seed_df: pd.DataFrame,
    table_summary_df: pd.DataFrame,
) -> Path:
    task_key = config["task_key"]
    suffix = result_suffix(config)
    data_dir = output_paths["data"]
    table_dir = output_paths["tables"]
    history_df.to_csv(data_dir / f"{task_key}_history_by_seed.csv", index=False)
    summary_seed_df.to_csv(data_dir / f"{task_key}_training_summary_by_seed.csv", index=False)
    diagnostics_df.to_csv(data_dir / f"{task_key}_diagnostics_by_seed_checkpoint_layer.csv", index=False)
    checkpoint_seed_df.to_csv(data_dir / f"{task_key}_diagnostics_by_seed_checkpoint.csv", index=False)
    diagnostic_seed_summary_df.to_csv(data_dir / f"{task_key}_diagnostics_by_seed_overall.csv", index=False)
    diagnostic_summary_df.to_csv(data_dir / f"{task_key}_diagnostics_summary_{suffix}.csv", index=False)
    checkpoint_summary_df.to_csv(data_dir / f"{task_key}_diagnostics_checkpoint_summary_{suffix}.csv", index=False)

    table_seed_df.to_csv(table_dir / f"{task_key}_results_table_by_seed.csv", index=False)
    table_summary_df.to_csv(table_dir / f"{task_key}_results_table_raw_{suffix}.csv", index=False)
    formatted_table = format_results_table(table_summary_df)
    formatted_table.to_csv(table_dir / f"{task_key}_results_table_{suffix}.csv", index=False)
    formatted_table.to_latex(table_dir / f"{task_key}_results_table_{suffix}.tex", index=False, escape=False)

    with (data_dir / f"{task_key}_{suffix}_outputs.pkl").open("wb") as handle:
        pickle.dump(
            {
                "config": config,
                "data_info": {key: data[key] for key in ["train_size", "train_eval_size", "test_size"] if key in data},
                "history_df": history_df,
                "summary_seed_df": summary_seed_df,
                "diagnostics_df": diagnostics_df,
                "checkpoint_seed_df": checkpoint_seed_df,
                "checkpoint_summary_df": checkpoint_summary_df,
                "diagnostic_seed_summary_df": diagnostic_seed_summary_df,
                "diagnostic_summary_df": diagnostic_summary_df,
                "table_seed_df": table_seed_df,
                "table_summary_df": table_summary_df,
            },
            handle,
        )

    return Path(shutil.make_archive(str(output_paths["output"]), "zip", root_dir=output_paths["output"]))


def run_final_config(config: dict, project_root: Path | None = None) -> dict:
    setup_matplotlib()
    project_root = project_root or PROJECT_ROOT
    device = get_device()
    output_paths = prepare_output_dirs(project_root, config["output_dir"])
    print(f"Output directory: {output_paths['output']}")
    print(f"Device: {device}")
    print(f"Seeds: {config['seeds']}")
    print("Fixed run configs:")
    for method, run_config in config["run_configs"].items():
        print(f"  {method}: {run_config}")

    data, history_df, summary_seed_df, diagnostics_df = run_task(config, device)
    history_summary_df = aggregate_history(history_df)
    checkpoint_seed_df, checkpoint_summary_df, diagnostic_seed_summary_df, diagnostic_summary_df = aggregate_diagnostics(diagnostics_df)
    table_seed_df, table_summary_df = aggregate_table(summary_seed_df, diagnostic_seed_summary_df)
    figure_paths = plot_all_results(config, history_df, diagnostic_summary_df, checkpoint_summary_df, output_paths["figures"])
    archive_path = save_outputs(
        config,
        output_paths,
        data,
        history_df,
        summary_seed_df,
        diagnostics_df,
        checkpoint_seed_df,
        checkpoint_summary_df,
        diagnostic_seed_summary_df,
        diagnostic_summary_df,
        table_seed_df,
        table_summary_df,
    )
    return {
        "config": config,
        "data": data,
        "history_df": history_df,
        "history_summary_df": history_summary_df,
        "summary_seed_df": summary_seed_df,
        "diagnostics_df": diagnostics_df,
        "checkpoint_seed_df": checkpoint_seed_df,
        "checkpoint_summary_df": checkpoint_summary_df,
        "diagnostic_seed_summary_df": diagnostic_seed_summary_df,
        "diagnostic_summary_df": diagnostic_summary_df,
        "table_seed_df": table_seed_df,
        "table_summary_df": table_summary_df,
        "figure_paths": figure_paths,
        "archive_path": archive_path,
        "output_paths": output_paths,
    }
