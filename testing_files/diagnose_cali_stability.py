from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from learning_rules_MLP import MLP, backprop_step


MC_SAMPLES = 80
EVAL_SUBSET_SIZE = 50
CHECKPOINT_LABELS = ["beginning", "middle", "final"]
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


def percentile(tensor, q):
    if tensor.numel() == 0:
        return float("nan")
    if tensor.numel() == 1:
        return float(tensor.item())
    return float(torch.quantile(tensor, q))


def evaluate_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return float(F.mse_loss(prediction, y, reduction="mean"))


def load_cali_dataset(config):
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

    return weight_grads, bias_grads, -flat_grad


def node_update(model, xb, yb, sigma):
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb(xb, sigma)
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))
    weight_updates = []
    bias_updates = []
    for x_in, noise, noise_scale in zip(activations, noises, noise_scales):
        scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
        weight_updates.append(torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0))
        bias_updates.append(scaled_noise.mean(dim=0))
    return weight_updates, bias_updates, flatten_model_tensors(weight_updates, bias_updates)


def weight_update(model, xb, yb, sigma):
    layer_outputs, _, noises = model.forward_weight_perturb(xb, sigma)
    prediction_noisy = layer_outputs[-1]
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))
    noise_scale = sigma ** 2 + 1e-12
    weight_updates = []
    bias_updates = []
    for weight_noise, bias_noise in noises:
        scaled_weight_noise = scalar_signal.view(-1, 1, 1) * weight_noise / noise_scale
        scaled_bias_noise = scalar_signal.view(-1, 1) * bias_noise / noise_scale
        weight_updates.append(scaled_weight_noise.mean(dim=0))
        bias_updates.append(scaled_bias_noise.mean(dim=0))
    return weight_updates, bias_updates, flatten_model_tensors(weight_updates, bias_updates)


def estimator_update(model, method, xb, yb, sigma):
    if method == "np":
        return node_update(model, xb, yb, sigma)
    return weight_update(model, xb, yb, sigma)


def scale_updates(weight_updates, bias_updates, scale):
    return [scale * update for update in weight_updates], [scale * update for update in bias_updates]


def apply_update_and_measure_loss(model, x_eval, y_eval, weight_updates, bias_updates, lr, x_probe, y_probe):
    updated_model = MLP(
        [model.layers[0].in_features] + [layer.out_features for layer in model.layers],
        activation=model.activation,
        output_activation=model.output_activation,
        require_grad=False,
    )
    updated_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        for layer, weight_update, bias_update in zip(updated_model.layers, weight_updates, bias_updates):
            layer.weight += lr * weight_update
            layer.bias += lr * bias_update
    return evaluate_loss(updated_model, x_probe, y_probe) - evaluate_loss(model, x_probe, y_probe)


def summarize_distribution(values):
    values = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_layers(layer_norm_draws):
    summaries = []
    for layer_values in zip(*layer_norm_draws):
        summaries.append(summarize_distribution(layer_values))
    return summaries


def diagnose_checkpoint(config, checkpoint, x_probe, y_probe):
    model = MLP(config["dimensions"], activation=torch.sigmoid, require_grad=False)
    model.load_state_dict(checkpoint["state_dict"])
    x_eval = checkpoint["x_eval"]
    y_eval = checkpoint["y_eval"]
    _, _, true_update = true_update_vector(model, x_eval, y_eval)
    true_norm = float(torch.norm(true_update))

    raw_draws = {"np": [], "wp": []}
    layer_norm_draws = {"np": [], "wp": []}
    for method in ["np", "wp"]:
        for _ in range(MC_SAMPLES):
            weight_updates, bias_updates, flat_update = estimator_update(model, method, x_eval, y_eval, config["sigma"])
            raw_draws[method].append((weight_updates, bias_updates, flat_update))
            layer_norm_draws[method].append(
                [float(weight_update.norm() + bias_update.norm()) for weight_update, bias_update in zip(weight_updates, bias_updates)]
            )

    np_mean_norm = torch.stack([draw[2].norm() for draw in raw_draws["np"]]).mean().item()
    wp_mean_norm = torch.stack([draw[2].norm() for draw in raw_draws["wp"]]).mean().item()
    scale_np = wp_mean_norm / max(np_mean_norm, 1e-12)

    results = {}
    for method in ["np", "wp"]:
        scale = scale_np if method == "np" else 1.0
        effective_lr = config["lr"] * scale
        norms = []
        projections = []
        cosines = []
        loss_jumps_batch = []
        loss_jumps_probe = []
        scaled_layer_norm_draws = []

        for weight_updates, bias_updates, flat_update in raw_draws[method]:
            scaled_flat = scale * flat_update
            norms.append(float(torch.norm(scaled_flat)))
            projections.append(float(torch.dot(scaled_flat, true_update) / (true_norm + 1e-12)))
            cosines.append(float(torch.dot(scaled_flat, true_update) / (torch.norm(scaled_flat) * true_norm + 1e-12)))

            scaled_weight_updates, scaled_bias_updates = scale_updates(weight_updates, bias_updates, scale)
            scaled_layer_norm_draws.append(
                [float(weight_update.norm() + bias_update.norm()) for weight_update, bias_update in zip(scaled_weight_updates, scaled_bias_updates)]
            )
            loss_jumps_batch.append(
                apply_update_and_measure_loss(
                    model,
                    x_eval,
                    y_eval,
                    scaled_weight_updates,
                    scaled_bias_updates,
                    lr=config["lr"],
                    x_probe=x_eval,
                    y_probe=y_eval,
                )
            )
            loss_jumps_probe.append(
                apply_update_and_measure_loss(
                    model,
                    x_eval,
                    y_eval,
                    scaled_weight_updates,
                    scaled_bias_updates,
                    lr=config["lr"],
                    x_probe=x_probe,
                    y_probe=y_probe,
                )
            )

        results[method] = {
            "scale": scale,
            "effective_lr": effective_lr,
            "norms": summarize_distribution(norms),
            "projections": summarize_distribution(projections),
            "cosines": summarize_distribution(cosines),
            "loss_jump_batch": summarize_distribution(loss_jumps_batch),
            "loss_jump_probe": summarize_distribution(loss_jumps_probe),
            "layer_norms": summarize_layers(scaled_layer_norm_draws),
        }

    return {
        "true_update_norm": true_norm,
        "np_raw_mean_norm": np_mean_norm,
        "wp_raw_mean_norm": wp_mean_norm,
        "results": results,
    }


def main():
    x_train, y_train, x_test, y_test = load_cali_dataset(CALI_CONFIG)
    checkpoints = locate_checkpoints(CALI_CONFIG, x_train, y_train)
    x_probe = x_test[:EVAL_SUBSET_SIZE]
    y_probe = y_test[:EVAL_SUBSET_SIZE]

    for checkpoint_label in CHECKPOINT_LABELS:
        diagnosis = diagnose_checkpoint(CALI_CONFIG, checkpoints[checkpoint_label], x_probe, y_probe)
        print(f"{checkpoint_label} train_loss={checkpoints[checkpoint_label]['train_loss']:.4f}")
        print(
            f"  raw_mean_norms: np={diagnosis['np_raw_mean_norm']:.4f}, "
            f"wp={diagnosis['wp_raw_mean_norm']:.4f}, "
            f"np_scale_to_match_wp={diagnosis['results']['np']['scale']:.4f}"
        )
        for method in ["np", "wp"]:
            stats = diagnosis["results"][method]
            print(
                f"  {method}: eff_lr={stats['effective_lr']:.4f}, "
                f"norm_mean={stats['norms']['mean']:.4f}, norm_p95={stats['norms']['p95']:.4f}, "
                f"proj_mean={stats['projections']['mean']:.4f}, proj_p95={stats['projections']['p95']:.4f}, "
                f"proj_min={stats['projections']['min']:.4f}, cos_mean={stats['cosines']['mean']:.4f}, "
                f"cos_p05={stats['cosines']['p05']:.4f}, "
                f"batch_loss_jump_mean={stats['loss_jump_batch']['mean']:.6f}, "
                f"batch_loss_jump_p95={stats['loss_jump_batch']['p95']:.6f}, "
                f"probe_loss_jump_mean={stats['loss_jump_probe']['mean']:.6f}, "
                f"probe_loss_jump_p95={stats['loss_jump_probe']['p95']:.6f}"
            )
            for layer_index, layer_stats in enumerate(stats["layer_norms"]):
                print(
                    f"    layer{layer_index}: norm_mean={layer_stats['mean']:.4f}, "
                    f"norm_p95={layer_stats['p95']:.4f}, norm_p99={layer_stats['p99']:.4f}"
                )


if __name__ == "__main__":
    main()
