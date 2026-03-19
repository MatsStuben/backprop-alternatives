from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from learning_rules_MLP import MLP, backprop_step


MC_SAMPLES = 100
EVAL_SUBSET_SIZE = 50
CHECKPOINT_LABELS = ["beginning", "middle", "final"]
METHODS = ["np", "wp"]
METHOD_COLORS = {"np": "C1", "wp": "C2"}
METHOD_LABELS = {"np": "Node Perturbation", "wp": "Weight Perturbation"}

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


def centered_reward_signal(loss_per_sample):
    reward = -loss_per_sample
    return reward - reward.mean()


def cosine_similarity_safe(a, b, eps=1e-12):
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    if a_norm.item() < eps or b_norm.item() < eps:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm + eps))


def evaluate_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return float(F.mse_loss(prediction, y, reduction="mean"))


def true_update_vector(model, xb, yb):
    requires_grad_state = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    prediction = model(xb)
    loss = F.mse_loss(prediction, yb, reduction="mean")
    loss.backward()

    weight_grads = [layer.weight.grad.detach().clone() for layer in model.layers]
    bias_grads = [layer.bias.grad.detach().clone() for layer in model.layers]
    flat_grad = flatten_model_tensors(weight_grads, bias_grads)

    model.zero_grad(set_to_none=True)
    for parameter, old_value in zip(model.parameters(), requires_grad_state):
        parameter.requires_grad_(old_value)

    return -flat_grad


def node_perturbation_update_vector(model, xb, yb, sigma):
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb(xb, sigma)
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))

    weight_updates = []
    bias_updates = []
    for x_in, noise, noise_scale in zip(activations, noises, noise_scales):
        scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
        weight_updates.append(torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0))
        bias_updates.append(scaled_noise.mean(dim=0))

    return flatten_model_tensors(weight_updates, bias_updates)


def weight_perturbation_update_vector(model, xb, yb, sigma):
    layer_outputs, _, noises = model.forward_weight_perturb(xb, sigma)
    prediction_noisy = layer_outputs[-1]
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))
    sigma_safe = sigma + 1e-12

    weight_updates = []
    bias_updates = []
    for weight_noise, bias_noise in noises:
        weight_updates.append((weight_noise * scalar_signal.view(-1, 1, 1)).mean(dim=0) / sigma_safe)
        bias_updates.append((bias_noise * scalar_signal.view(-1, 1)).mean(dim=0) / sigma_safe)

    return flatten_model_tensors(weight_updates, bias_updates)


def estimator_update_vector(model, method, xb, yb, sigma):
    if method == "np":
        return node_perturbation_update_vector(model, xb, yb, sigma)
    if method == "wp":
        return weight_perturbation_update_vector(model, xb, yb, sigma)
    raise ValueError(f"Unknown method: {method}")


def build_sinus_dataset(config):
    generator = torch.Generator().manual_seed(config["seed"])
    x_train = (torch.rand(config["train_samples"], 1, generator=generator) * 4.0 - 2.0) * math.pi
    y_train = torch.sin(x_train) + config["train_noise_std"] * torch.randn_like(x_train, generator=generator)
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


def make_initial_state(dimensions, seed):
    torch.manual_seed(seed)
    model = MLP(dimensions, activation=torch.sigmoid, require_grad=True)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def locate_checkpoints(config, x_train, y_train):
    initial_state = make_initial_state(config["dimensions"], config["seed"])
    model = MLP(config["dimensions"], activation=torch.sigmoid, require_grad=True)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    checkpoints = {}
    batch_size = config["batch_size"]

    for epoch in range(config["epochs"]):
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


def evaluate_true_variance_at_checkpoint(config, checkpoint):
    results = {}
    x_eval = checkpoint["x_eval"]
    y_eval = checkpoint["y_eval"]

    for method in METHODS:
        model = MLP(config["dimensions"], activation=torch.sigmoid, require_grad=False)
        model.load_state_dict(checkpoint["state_dict"])

        true_update = true_update_vector(model, x_eval, y_eval)
        estimates = []
        for _ in range(MC_SAMPLES):
            estimates.append(estimator_update_vector(model, method, x_eval, y_eval, config["sigma"]))

        estimate_matrix = torch.stack(estimates, dim=0)
        estimate_mean = estimate_matrix.mean(dim=0)
        centered = estimate_matrix - estimate_mean

        mean_variance = float(centered.pow(2).mean())
        bias_squared = float((estimate_mean - true_update).pow(2).mean())
        mse_to_true_update = float((estimate_matrix - true_update.unsqueeze(0)).pow(2).mean())
        cosine_to_true_update = cosine_similarity_safe(estimate_mean, true_update)

        results[method] = {
            "mean_variance": mean_variance,
            "bias_squared": bias_squared,
            "mse_to_true_update": mse_to_true_update,
            "cosine_to_true_update": cosine_to_true_update,
        }

    return results


def run_dataset_benchmark(config, dataset_builder):
    x_train, y_train, _, _ = dataset_builder(config)
    checkpoints = locate_checkpoints(config, x_train, y_train)

    dataset_results = {}
    for checkpoint_label in CHECKPOINT_LABELS:
        dataset_results[checkpoint_label] = evaluate_true_variance_at_checkpoint(config, checkpoints[checkpoint_label])

    return checkpoints, dataset_results


def plot_results(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    dataset_names = list(all_results.keys())
    x_positions = list(range(len(CHECKPOINT_LABELS)))

    for row_index, dataset_name in enumerate(dataset_names):
        variance_axis = axes[row_index][0]
        cosine_axis = axes[row_index][1]
        dataset_results = all_results[dataset_name]["metrics"]

        for method in METHODS:
            variance_values = [dataset_results[label][method]["mean_variance"] for label in CHECKPOINT_LABELS]
            cosine_values = [dataset_results[label][method]["cosine_to_true_update"] for label in CHECKPOINT_LABELS]

            variance_axis.plot(
                x_positions,
                variance_values,
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            cosine_axis.plot(
                x_positions,
                cosine_values,
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )

        variance_axis.set_title(f"{dataset_name}: True Minibatch Variance")
        variance_axis.set_ylabel("Mean variance")
        variance_axis.set_xticks(x_positions, CHECKPOINT_LABELS)
        variance_axis.legend()

        cosine_axis.set_title(f"{dataset_name}: Cosine of Mean Estimator")
        cosine_axis.set_ylabel("Cosine")
        cosine_axis.set_xticks(x_positions, CHECKPOINT_LABELS)
        cosine_axis.legend()

    fig.tight_layout()
    plt.show()


def main():
    all_results = {
        SINUS_CONFIG["name"]: {},
        CALI_CONFIG["name"]: {},
    }

    sinus_checkpoints, sinus_metrics = run_dataset_benchmark(SINUS_CONFIG, build_sinus_dataset)
    all_results[SINUS_CONFIG["name"]]["checkpoints"] = sinus_checkpoints
    all_results[SINUS_CONFIG["name"]]["metrics"] = sinus_metrics

    cali_checkpoints, cali_metrics = run_dataset_benchmark(CALI_CONFIG, build_cali_dataset)
    all_results[CALI_CONFIG["name"]]["checkpoints"] = cali_checkpoints
    all_results[CALI_CONFIG["name"]]["metrics"] = cali_metrics

    for dataset_name, result in all_results.items():
        print(dataset_name)
        for checkpoint_label in CHECKPOINT_LABELS:
            train_loss = result["checkpoints"][checkpoint_label]["train_loss"]
            print(f"  {checkpoint_label}: train_loss={train_loss:.4f}")
            for method in METHODS:
                metrics = result["metrics"][checkpoint_label][method]
                print(
                    f"    {method}: variance={metrics['mean_variance']:.6e}, "
                    f"bias_sq={metrics['bias_squared']:.6e}, "
                    f"mse={metrics['mse_to_true_update']:.6e}, "
                    f"cos={metrics['cosine_to_true_update']:.4f}"
                )

    plot_results(all_results)


if __name__ == "__main__":
    main()
