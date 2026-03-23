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


MC_SAMPLES = 200
EVAL_SUBSET_SIZE = 20
CHECKPOINT_LABELS = ["beginning", "middle", "final"]

SINUS_CONFIG = {
    "name": "Sinus",
    "seed": 0,
    "dimensions": (1, 32, 16, 1),
    "batch_size": 64,
    "epochs": 1000,
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
    "epochs": 100,
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


def theorem_estimators_from_weight_noise(model, xb, yb, sigma):
    weight_estimates = []
    bias_estimates = []
    node_weight_estimates = []
    node_bias_estimates = []

    h = xb
    final_layer = len(model.layers) - 1
    prediction = None

    for layer_index, layer in enumerate(model.layers):
        weight_noise = torch.randn_like(layer.weight)
        bias_noise = torch.randn_like(layer.bias)

        augmented_input_sq_norm = 1.0 + h.pow(2).sum()
        induced_noise = sigma * (weight_noise @ h.squeeze(0) + bias_noise)

        pre_activation = layer(h).squeeze(0) + induced_noise
        if layer_index == final_layer:
            next_h = model.output_activation(pre_activation) if model.output_activation else pre_activation
        else:
            next_h = model.activation(pre_activation)

        scalar_factor = h.squeeze(0) / (sigma * augmented_input_sq_norm)
        node_weight = -weight_noise.new_zeros(layer.weight.shape)
        node_bias = -bias_noise.new_zeros(layer.bias.shape)
        for neuron_index in range(layer.weight.shape[0]):
            node_weight[neuron_index] = induced_noise[neuron_index] * scalar_factor
            node_bias[neuron_index] = induced_noise[neuron_index] / (sigma * augmented_input_sq_norm)

        weight_estimates.append(weight_noise)
        bias_estimates.append(bias_noise)
        node_weight_estimates.append(node_weight)
        node_bias_estimates.append(node_bias)

        h = next_h.unsqueeze(0)
        prediction = h

    delta_loss = F.mse_loss(prediction, yb, reduction="mean")
    wp_weights = [-(delta_loss / sigma) * noise for noise in weight_estimates]
    wp_biases = [-(delta_loss / sigma) * noise for noise in bias_estimates]
    np_weights = [-(delta_loss / sigma) * noise for noise in node_weight_estimates]
    np_biases = [-(delta_loss / sigma) * noise for noise in node_bias_estimates]

    return (
        flatten_model_tensors(wp_weights, wp_biases),
        flatten_model_tensors(np_weights, np_biases),
    )


def evaluate_theorem_at_checkpoint(config, checkpoint):
    model = MLP(config["dimensions"], activation=torch.sigmoid, require_grad=False)
    model.load_state_dict(checkpoint["state_dict"])

    wp_coordinate_variances = []
    np_coordinate_variances = []
    coordinate_fraction_holds = []
    sample_mean_gap = []

    for sample_index in range(checkpoint["x_eval"].size(0)):
        xb = checkpoint["x_eval"][sample_index:sample_index + 1]
        yb = checkpoint["y_eval"][sample_index:sample_index + 1]

        wp_draws = []
        np_draws = []
        for _ in range(MC_SAMPLES):
            wp_estimate, np_estimate = theorem_estimators_from_weight_noise(model, xb, yb, config["sigma"])
            wp_draws.append(wp_estimate)
            np_draws.append(np_estimate)

        wp_matrix = torch.stack(wp_draws, dim=0)
        np_matrix = torch.stack(np_draws, dim=0)
        wp_var = wp_matrix.var(dim=0, unbiased=True)
        np_var = np_matrix.var(dim=0, unbiased=True)

        wp_coordinate_variances.append(wp_var.mean())
        np_coordinate_variances.append(np_var.mean())
        coordinate_fraction_holds.append((np_var <= wp_var + 1e-12).float().mean())
        sample_mean_gap.append((wp_var - np_var).mean())

    return {
        "wp_mean_coordinate_variance": float(torch.stack(wp_coordinate_variances).mean()),
        "np_mean_coordinate_variance": float(torch.stack(np_coordinate_variances).mean()),
        "mean_variance_gap": float(torch.stack(sample_mean_gap).mean()),
        "mean_fraction_coordinates_holds": float(torch.stack(coordinate_fraction_holds).mean()),
    }


def run_dataset_benchmark(config, dataset_builder):
    x_train, y_train, _, _ = dataset_builder(config)
    checkpoints = locate_checkpoints(config, x_train, y_train)
    metrics = {}
    for checkpoint_label in CHECKPOINT_LABELS:
        metrics[checkpoint_label] = evaluate_theorem_at_checkpoint(config, checkpoints[checkpoint_label])
    return checkpoints, metrics


def plot_results(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x_positions = list(range(len(CHECKPOINT_LABELS)))

    for row_index, dataset_name in enumerate(all_results.keys()):
        variance_axis = axes[row_index][0]
        fraction_axis = axes[row_index][1]
        metrics = all_results[dataset_name]["metrics"]

        wp_values = [metrics[label]["wp_mean_coordinate_variance"] for label in CHECKPOINT_LABELS]
        np_values = [metrics[label]["np_mean_coordinate_variance"] for label in CHECKPOINT_LABELS]
        fraction_values = [metrics[label]["mean_fraction_coordinates_holds"] for label in CHECKPOINT_LABELS]

        variance_axis.plot(x_positions, wp_values, marker="o", color="C2", label="WP")
        variance_axis.plot(x_positions, np_values, marker="o", color="C1", label="NP")
        variance_axis.set_title(f"{dataset_name}: Coordinate Variance")
        variance_axis.set_ylabel("Mean coordinate variance")
        variance_axis.set_xticks(x_positions, CHECKPOINT_LABELS)
        variance_axis.legend()

        fraction_axis.plot(x_positions, fraction_values, marker="o", color="C0")
        fraction_axis.set_title(f"{dataset_name}: Fraction NP <= WP")
        fraction_axis.set_ylabel("Fraction of coordinates")
        fraction_axis.set_xticks(x_positions, CHECKPOINT_LABELS)

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
            metrics = result["metrics"][checkpoint_label]
            print(
                f"  {checkpoint_label}: train_loss={train_loss:.4f}, "
                f"wp_var={metrics['wp_mean_coordinate_variance']:.6e}, "
                f"np_var={metrics['np_mean_coordinate_variance']:.6e}, "
                f"gap={metrics['mean_variance_gap']:.6e}, "
                f"frac_holds={metrics['mean_fraction_coordinates_holds']:.4f}"
            )

    plot_results(all_results)


if __name__ == "__main__":
    main()
