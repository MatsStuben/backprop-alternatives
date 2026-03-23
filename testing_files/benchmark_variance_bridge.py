from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from learning_rules_MLP import MLP, backprop_step


MC_SAMPLES = 100
EVAL_SUBSET_SIZE = 50
CHECKPOINT_LABELS = ["beginning", "middle", "final"]
METHODS = ["np", "wp"]
VARIANTS = [
    "single_no_baseline",
    "minibatch_no_baseline",
    "minibatch_with_baseline",
]

SINUS_CONFIG = {
    "name": "Sinus",
    "seed": 0,
    "dimensions": (1, 32, 16, 1),
    "batch_size": 64,
    "epochs": 400,
    "lr": 0.05,
    "sigma": 0.2,
    "thresholds": {"middle": 0.2, "final": 0.1},
    "train_samples": 512,
    "test_samples": 512,
    "train_noise_std": 0.1,
}

CALI_CONFIG = {
    "name": "California Housing",
    "seed": 0,
    "dimensions": (8, 128, 64, 1),
    "batch_size": 256,
    "epochs": 80,
    "lr": 0.01,
    "sigma": 0.1,
    "thresholds": {"middle": 0.6, "final": 0.4},
}


def flatten_model_tensors(weight_tensors, bias_tensors):
    pieces = []
    for weight_tensor, bias_tensor in zip(weight_tensors, bias_tensors):
        pieces.append(weight_tensor.reshape(-1))
        pieces.append(bias_tensor.reshape(-1))
    return torch.cat(pieces)


def mse_per_sample(prediction, target):
    loss = F.mse_loss(prediction, target, reduction="none")
    if loss.dim() > 1:
        loss = loss.mean(dim=1)
    return loss.view(-1)


def reward_signal(loss_per_sample, use_baseline):
    reward = -loss_per_sample
    if use_baseline:
        return reward - reward.mean()
    return reward


def evaluate_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return float(F.mse_loss(prediction, y, reduction="mean"))


def make_initial_state(dimensions, seed):
    torch.manual_seed(seed)
    model = MLP(dimensions, activation=torch.sigmoid, require_grad=True)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def build_sinus_dataset(config):
    generator = torch.Generator().manual_seed(config["seed"])
    x_train = (torch.rand(config["train_samples"], 1, generator=generator) * 4.0 - 2.0) * math.pi
    y_train = torch.sin(x_train) + config["train_noise_std"] * torch.randn(x_train.shape, generator=generator)
    x_test = torch.linspace(-2 * math.pi, 2 * math.pi, config["test_samples"]).unsqueeze(1)
    y_test = torch.sin(x_test)
    return x_train, y_train, x_test, y_test


def build_cali_dataset(config):
    dataset = fetch_california_housing()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=config["seed"],
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


def locate_checkpoints(config, x_train, y_train):
    initial_state = make_initial_state(config["dimensions"], config["seed"])
    model = MLP(config["dimensions"], activation=torch.sigmoid, require_grad=True)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    checkpoints = {}
    batch_size = config["batch_size"]

    for _epoch in range(config["epochs"]):
        permutation = torch.randperm(x_train.size(0))
        for batch_start in range(0, x_train.size(0), batch_size):
            batch_end = min(batch_start + batch_size, x_train.size(0))
            batch_indices = permutation[batch_start:batch_end]
            xb = x_train[batch_indices]
            yb = y_train[batch_indices]
            backprop_step(model, xb, yb, optimizer=optimizer)
            train_loss = evaluate_loss(model, x_train, y_train)

            if "beginning" not in checkpoints:
                checkpoints["beginning"] = {
                    "state_dict": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
                    "x_eval": xb[:EVAL_SUBSET_SIZE].detach().clone(),
                    "y_eval": yb[:EVAL_SUBSET_SIZE].detach().clone(),
                    "train_loss": train_loss,
                }

            if "middle" not in checkpoints and train_loss <= config["thresholds"]["middle"]:
                checkpoints["middle"] = {
                    "state_dict": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
                    "x_eval": xb[:EVAL_SUBSET_SIZE].detach().clone(),
                    "y_eval": yb[:EVAL_SUBSET_SIZE].detach().clone(),
                    "train_loss": train_loss,
                }

            if "final" not in checkpoints and train_loss <= config["thresholds"]["final"]:
                checkpoints["final"] = {
                    "state_dict": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
                    "x_eval": xb[:EVAL_SUBSET_SIZE].detach().clone(),
                    "y_eval": yb[:EVAL_SUBSET_SIZE].detach().clone(),
                    "train_loss": train_loss,
                }

            if all(label in checkpoints for label in CHECKPOINT_LABELS):
                return checkpoints

    missing = [label for label in CHECKPOINT_LABELS if label not in checkpoints]
    raise RuntimeError(f"{config['name']} did not reach checkpoints: {missing}")


def node_update(model, xb, yb, sigma, use_baseline):
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb(xb, sigma)
    scalar_signal = reward_signal(mse_per_sample(prediction_noisy, yb), use_baseline=use_baseline)
    weight_updates = []
    bias_updates = []
    for x_in, noise, noise_scale in zip(activations, noises, noise_scales):
        scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
        weight_updates.append(torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0))
        bias_updates.append(scaled_noise.mean(dim=0))
    return flatten_model_tensors(weight_updates, bias_updates)


def weight_update(model, xb, yb, sigma, use_baseline):
    layer_outputs, _, noises = model.forward_weight_perturb(xb, sigma)
    prediction_noisy = layer_outputs[-1]
    scalar_signal = reward_signal(mse_per_sample(prediction_noisy, yb), use_baseline=use_baseline)
    noise_scale = sigma ** 2 + 1e-12
    weight_updates = []
    bias_updates = []
    for weight_noise, bias_noise in noises:
        scaled_weight_noise = scalar_signal.view(-1, 1, 1) * weight_noise / noise_scale
        scaled_bias_noise = scalar_signal.view(-1, 1) * bias_noise / noise_scale
        weight_updates.append(scaled_weight_noise.mean(dim=0))
        bias_updates.append(scaled_bias_noise.mean(dim=0))
    return flatten_model_tensors(weight_updates, bias_updates)


def estimator_update(model, method, xb, yb, sigma, use_baseline):
    if method == "np":
        return node_update(model, xb, yb, sigma, use_baseline)
    return weight_update(model, xb, yb, sigma, use_baseline)


def estimate_variant_variance(model, method, x_eval, y_eval, sigma, variant):
    draws = []
    if variant == "single_no_baseline":
        for _ in range(MC_SAMPLES):
            sample_updates = []
            for idx in range(x_eval.size(0)):
                xb = x_eval[idx:idx + 1]
                yb = y_eval[idx:idx + 1]
                sample_updates.append(estimator_update(model, method, xb, yb, sigma, use_baseline=False))
            draws.append(torch.stack(sample_updates, dim=0).mean(dim=0))
    elif variant == "minibatch_no_baseline":
        for _ in range(MC_SAMPLES):
            draws.append(estimator_update(model, method, x_eval, y_eval, sigma, use_baseline=False))
    elif variant == "minibatch_with_baseline":
        for _ in range(MC_SAMPLES):
            draws.append(estimator_update(model, method, x_eval, y_eval, sigma, use_baseline=True))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    draw_matrix = torch.stack(draws, dim=0)
    return float(draw_matrix.var(dim=0, unbiased=True).mean())


def evaluate_checkpoint(config, checkpoint):
    results = {variant: {} for variant in VARIANTS}
    for method in METHODS:
        model = MLP(config["dimensions"], activation=torch.sigmoid, require_grad=False)
        model.load_state_dict(checkpoint["state_dict"])
        for variant in VARIANTS:
            results[variant][method] = estimate_variant_variance(
                model,
                method,
                checkpoint["x_eval"],
                checkpoint["y_eval"],
                config["sigma"],
                variant,
            )
    return results


def run_dataset(config, dataset_builder):
    x_train, y_train, _, _ = dataset_builder(config)
    checkpoints = locate_checkpoints(config, x_train, y_train)
    metrics = {}
    for checkpoint_label in CHECKPOINT_LABELS:
        metrics[checkpoint_label] = evaluate_checkpoint(config, checkpoints[checkpoint_label])
    return checkpoints, metrics


def main():
    datasets = [
        (SINUS_CONFIG, build_sinus_dataset),
        (CALI_CONFIG, build_cali_dataset),
    ]
    for config, builder in datasets:
        checkpoints, metrics = run_dataset(config, builder)
        print(config["name"])
        for checkpoint_label in CHECKPOINT_LABELS:
            print(f"  {checkpoint_label}: train_loss={checkpoints[checkpoint_label]['train_loss']:.4f}")
            for variant in VARIANTS:
                np_var = metrics[checkpoint_label][variant]["np"]
                wp_var = metrics[checkpoint_label][variant]["wp"]
                print(
                    f"    {variant}: np_var={np_var:.6e}, wp_var={wp_var:.6e}, "
                    f"ratio_wp_over_np={wp_var / max(np_var, 1e-12):.4f}"
                )


if __name__ == "__main__":
    main()
