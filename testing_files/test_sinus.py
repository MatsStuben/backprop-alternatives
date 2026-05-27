from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from learning_rules_MLP import (
    MLP,
    backprop_step,
    node_perturbation_step,
    node_perturbation_step_fan_in_scaled,
    node_perturbation_step_fixed_sigma,
    weight_perturb_step,
)


METHODS = ["bp", "np", "np_fan_in", "np_fixed", "wp"]
# Edit one lr and one sigma per method here.
METHOD_CONFIG = {
    "bp": {"label": "Backprop", "color": "C0", "lr": 0.1, "sigma": None, "requires_grad": True},
    "np": {"label": "Node Perturbation", "color": "C1", "lr": 0.15, "sigma": 0.2, "requires_grad": False},
    "np_fan_in": {"label": "Node Perturbation Fan-In", "color": "C4", "lr": 0.05, "sigma": 0.1, "requires_grad": False},
    "np_fixed": {"label": "Node Perturbation Fixed Sigma", "color": "C3", "lr": 0.025, "sigma": 0.2, "requires_grad": False},
    "wp": {"label": "Weight Perturbation", "color": "C2", "lr": 0.075, "sigma": 0.2, "requires_grad": False},
}

SEED = 0
TRAIN_SAMPLES = 512
TEST_SAMPLES = 512
TRAIN_NOISE_STD = 0.1
DIMENSIONS = (1, 32, 16, 1)
BATCH_SIZE = 64
EPOCHS = 1000
METRIC_EVERY = 1
PRINT_EVERY = 20
LOSS_PLOT_MAX = 1.0


def generate_sinus_data(n_train, n_test, noise_std, seed):
    generator = torch.Generator().manual_seed(seed)
    x_train = (torch.rand(n_train, 1, generator=generator) * 4.0 - 2.0) * math.pi
    y_train = torch.sin(x_train) + noise_std * torch.randn(x_train.shape, generator=generator)

    x_test = torch.linspace(-2 * math.pi, 2 * math.pi, n_test).unsqueeze(1)
    y_test = torch.sin(x_test)
    return x_train, y_train, x_test, y_test


def flatten_tensors(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


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


def true_gradient(model, xb, yb):
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

    return weight_grads, bias_grads, flat_grad


def node_perturbation_gradient_estimate(model, xb, yb, sigma):
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb(xb, sigma)
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))

    weight_grads = []
    bias_grads = []
    for x_in, noise, noise_scale in zip(activations, noises, noise_scales):
        scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
        weight_grads.append(torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0))
        bias_grads.append(scaled_noise.mean(dim=0))

    return weight_grads, bias_grads, flatten_model_tensors(weight_grads, bias_grads)


def node_perturbation_fan_in_gradient_estimate(model, xb, yb, sigma):
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb_fan_in_scaled(xb, sigma)
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))

    weight_grads = []
    bias_grads = []
    for x_in, noise, noise_scale in zip(activations, noises, noise_scales):
        scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
        weight_grads.append(torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0))
        bias_grads.append(scaled_noise.mean(dim=0))

    return weight_grads, bias_grads, flatten_model_tensors(weight_grads, bias_grads)


def node_perturbation_fixed_sigma_gradient_estimate(model, xb, yb, sigma):
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb_fixed_sigma(xb, sigma)
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))

    weight_grads = []
    bias_grads = []
    for x_in, noise, noise_scale in zip(activations, noises, noise_scales):
        scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
        weight_grads.append(torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0))
        bias_grads.append(scaled_noise.mean(dim=0))

    return weight_grads, bias_grads, flatten_model_tensors(weight_grads, bias_grads)


def weight_perturbation_gradient_estimate(model, xb, yb, sigma):
    layer_outputs, _, noises = model.forward_weight_perturb(xb, sigma)
    prediction_noisy = layer_outputs[-1]
    scalar_signal = centered_reward_signal(mse_per_sample(prediction_noisy, yb))
    noise_scale = sigma ** 2 + 1e-12

    weight_grads = []
    bias_grads = []
    for weight_noise, bias_noise in noises:
        scaled_weight_noise = scalar_signal.view(-1, 1, 1) * weight_noise / noise_scale
        scaled_bias_noise = scalar_signal.view(-1, 1) * bias_noise / noise_scale
        weight_grads.append(scaled_weight_noise.mean(dim=0))
        bias_grads.append(scaled_bias_noise.mean(dim=0))

    return weight_grads, bias_grads, flatten_model_tensors(weight_grads, bias_grads)


def cosine_similarity_safe(a, b, eps=1e-12):
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    if a_norm.item() < eps or b_norm.item() < eps:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm + eps))


def gradient_metrics(unscaled_parameter_update_vector, true_update):
    diff = unscaled_parameter_update_vector - true_update
    cosine = cosine_similarity_safe(unscaled_parameter_update_vector, true_update)
    variance_estimate = float(diff.pow(2).mean())
    estimator_norm = float(torch.norm(unscaled_parameter_update_vector))
    true_update_norm = float(torch.norm(true_update))
    projection = float(torch.dot(unscaled_parameter_update_vector, true_update) / (true_update_norm + 1e-12))
    return cosine, variance_estimate, estimator_norm, true_update_norm, projection


def hidden_activation_norms(model, x):
    model.eval()
    norms = []
    with torch.no_grad():
        h = x
        final_layer = len(model.layers) - 1
        for i, layer in enumerate(model.layers):
            u = layer(h)
            if i == final_layer:
                break
            h = model.activation(u)
            norms.append(float(torch.norm(h, dim=1).mean()))
    return norms


def evaluate_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return float(F.mse_loss(prediction, y, reduction="mean"))


def make_model_copies():
    torch.manual_seed(SEED)
    base_model = MLP(DIMENSIONS, activation=torch.sigmoid, require_grad=True)
    base_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}

    models = {}
    optimizers = {}
    for method in METHODS:
        config = METHOD_CONFIG[method]
        model = MLP(DIMENSIONS, activation=torch.sigmoid, require_grad=config["requires_grad"])
        model.load_state_dict(base_state)
        models[method] = model
        if method == "bp":
            optimizers[method] = torch.optim.SGD(model.parameters(), lr=config["lr"])

    return models, optimizers


def step_method(method, model, optimizer, xb, yb):
    config = METHOD_CONFIG[method]
    if method == "bp":
        return backprop_step(
            model,
            xb,
            yb,
            optimizer=optimizer,
            return_unscaled_parameter_update_vector=True,
        )
    if method == "np":
        return node_perturbation_step(
            model,
            xb,
            yb,
            eta=config["lr"],
            sigma=config["sigma"],
            return_unscaled_parameter_update_vector=True,
        )
    if method == "np_fan_in":
        return node_perturbation_step_fan_in_scaled(
            model,
            xb,
            yb,
            eta=config["lr"],
            sigma=config["sigma"],
            return_unscaled_parameter_update_vector=True,
        )
    if method == "np_fixed":
        return node_perturbation_step_fixed_sigma(
            model,
            xb,
            yb,
            eta=config["lr"],
            sigma=config["sigma"],
            return_unscaled_parameter_update_vector=True,
        )
    if method == "wp":
        return weight_perturb_step(
            model,
            xb,
            yb,
            eta=config["lr"],
            sigma=config["sigma"],
            return_unscaled_parameter_update_vector=True,
        )
    raise ValueError(f"Unknown method: {method}")


def plot_average_gradient_metrics(cosine_history, variance_history, projection_history):
    methods_to_compare = [method for method in METHODS if method in {"np", "np_fan_in", "np_fixed", "wp"}]
    labels = [METHOD_CONFIG[method]["label"] for method in methods_to_compare]
    colors = [METHOD_CONFIG[method]["color"] for method in methods_to_compare]
    mean_cosines = [sum(cosine_history[method]) / max(len(cosine_history[method]), 1) for method in methods_to_compare]
    mean_variances = [sum(variance_history[method]) / max(len(variance_history[method]), 1) for method in methods_to_compare]
    mean_projections = [sum(projection_history[method]) / max(len(projection_history[method]), 1) for method in methods_to_compare]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(labels, mean_cosines, color=colors)
    axes[0].set_title("Average Cosine Similarity")
    axes[0].set_ylabel("Cosine")

    axes[1].bar(labels, mean_variances, color=colors)
    axes[1].set_title("Average Gradient Variance")
    axes[1].set_ylabel("Mean squared error")

    axes[2].bar(labels, mean_projections, color=colors)
    axes[2].set_title("Average Projection onto True Update")
    axes[2].set_ylabel("Signed projection")

    fig.tight_layout()


def plot_cosine_distributions(cosine_history):
    methods_to_compare = [method for method in METHODS if method in {"np", "np_fan_in", "np_fixed", "wp"}]
    fig, axes = plt.subplots(1, len(methods_to_compare), figsize=(10, 4), sharey=True)

    if len(methods_to_compare) == 1:
        axes = [axes]

    for axis, method in zip(axes, methods_to_compare):
        axis.hist(cosine_history[method], bins=30, color=METHOD_CONFIG[method]["color"], alpha=0.8)
        axis.set_title(f"{METHOD_CONFIG[method]['label']} Cosine Distribution")
        axis.set_xlabel("Cosine similarity")
        axis.set_ylabel("Count")

    fig.tight_layout()


def plot_hidden_activation_norms(iterations, hidden_activation_norm_history):
    num_hidden_layers = len(hidden_activation_norm_history[METHODS[0]])
    fig, axes = plt.subplots(num_hidden_layers, 1, figsize=(10, 4 * num_hidden_layers), sharex=True)

    if num_hidden_layers == 1:
        axes = [axes]

    for layer_idx, axis in enumerate(axes):
        for method in METHODS:
            config = METHOD_CONFIG[method]
            axis.plot(
                iterations,
                hidden_activation_norm_history[method][layer_idx],
                label=config["label"],
                color=config["color"],
            )
        axis.set_title(f"Hidden Layer {layer_idx + 1} Activation Norm")
        axis.set_ylabel("L2 norm")
        axis.legend()

    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()


def main():
    x_train, y_train, x_test, y_test = generate_sinus_data(
        n_train=TRAIN_SAMPLES,
        n_test=TEST_SAMPLES,
        noise_std=TRAIN_NOISE_STD,
        seed=SEED,
    )
    models, optimizers = make_model_copies()

    print(
        "Running sinus with configs: "
        + " | ".join(
            f"{method}: lr={config['lr']}"
            + (f", sigma={config['sigma']}" if config["sigma"] is not None else "")
            for method, config in METHOD_CONFIG.items()
        )
    )

    iterations = []
    train_loss_history = {method: [] for method in METHODS}
    test_loss_history = {method: [] for method in METHODS}
    cosine_history = {method: [] for method in METHODS}
    variance_history = {method: [] for method in METHODS}
    estimator_norm_history = {method: [] for method in METHODS}
    true_update_norm_history = {method: [] for method in METHODS}
    projection_history = {method: [] for method in METHODS}
    hidden_layer_count = len(models[METHODS[0]].layers) - 1
    hidden_activation_norm_history = {
        method: [[] for _ in range(hidden_layer_count)] for method in METHODS
    }

    iteration = 0
    batches_per_epoch = (x_train.size(0) + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        permutation = torch.randperm(x_train.size(0))
        for batch_start in range(0, x_train.size(0), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, x_train.size(0))
            batch_indices = permutation[batch_start:batch_end]
            xb = x_train[batch_indices]
            yb = y_train[batch_indices]
            iteration += 1

            for method in METHODS:
                model = models[method]
                _, _, true_grad = true_gradient(model, xb, yb)
                true_update = -true_grad
                _, unscaled_parameter_update_vector = step_method(method, model, optimizers.get(method), xb, yb)
                cosine, variance_estimate, estimator_norm, true_update_norm, projection = gradient_metrics(
                    unscaled_parameter_update_vector,
                    true_update,
                )

                if iteration % METRIC_EVERY == 0:
                    train_loss_history[method].append(evaluate_loss(model, x_train, y_train))
                    test_loss_history[method].append(evaluate_loss(model, x_test, y_test))
                    cosine_history[method].append(cosine)
                    variance_history[method].append(variance_estimate)
                    estimator_norm_history[method].append(estimator_norm)
                    true_update_norm_history[method].append(true_update_norm)
                    projection_history[method].append(projection)
                    for layer_idx, activation_norm in enumerate(hidden_activation_norms(model, x_train)):
                        hidden_activation_norm_history[method][layer_idx].append(activation_norm)

            if iteration % METRIC_EVERY == 0:
                iterations.append(iteration)

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0 or epoch + 1 == EPOCHS:
            status_parts = []
            for method in METHODS:
                status_parts.append(
                    f"{method}: train={train_loss_history[method][-1]:.4f}, "
                    f"test={test_loss_history[method][-1]:.4f}, "
                    f"cos={cosine_history[method][-1]:.4f}, "
                    f"var={variance_history[method][-1]:.4e}, "
                    f"est_norm={estimator_norm_history[method][-1]:.4f}, "
                    f"proj={projection_history[method][-1]:.4f}"
                )
            print(f"Epoch {epoch + 1:4d}/{EPOCHS} | " + " | ".join(status_parts))

    fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    for method in METHODS:
        config = METHOD_CONFIG[method]
        axes[0].plot(iterations, train_loss_history[method], label=f"{config['label']} train", color=config["color"])
        axes[0].plot(iterations, test_loss_history[method], linestyle="--", label=f"{config['label']} test", color=config["color"])
        axes[1].plot(iterations, cosine_history[method], label=config["label"], color=config["color"])
        axes[2].plot(iterations, variance_history[method], label=config["label"], color=config["color"])
        axes[3].plot(iterations, estimator_norm_history[method], label=config["label"], color=config["color"])
        axes[4].plot(iterations, projection_history[method], label=config["label"], color=config["color"])

    axes[0].set_title("Sinus Regression Loss")
    axes[0].set_ylabel("MSE")
    axes[0].set_ylim(0.0, LOSS_PLOT_MAX)
    axes[0].legend()
    axes[1].set_title("Cosine Similarity to True Gradient")
    axes[1].set_ylabel("Cosine")
    axes[1].legend()
    axes[2].set_title("Estimated Mean Gradient Variance")
    axes[2].set_ylabel("Mean squared error")
    axes[2].legend()
    axes[3].set_title("Estimator Norm")
    axes[3].set_ylabel("L2 norm")
    axes[3].legend()
    axes[4].set_title("Projection onto True Update")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("Signed projection")
    axes[4].legend()
    fig.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(x_test.squeeze().numpy(), y_test.squeeze().numpy(), color="black", linewidth=2, label="True signal")
    plt.scatter(x_train.squeeze().numpy(), y_train.squeeze().numpy(), color="0.8", s=10, alpha=0.5, label="Train samples")
    for method in METHODS:
        config = METHOD_CONFIG[method]
        with torch.no_grad():
            prediction = models[method](x_test)
        plt.plot(x_test.squeeze().numpy(), prediction.squeeze().numpy(), color=config["color"], label=config["label"])

    plt.title("Sinus Test Predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plot_hidden_activation_norms(iterations, hidden_activation_norm_history)
    plot_average_gradient_metrics(cosine_history, variance_history, projection_history)
    plot_cosine_distributions(cosine_history)
    plt.show()


if __name__ == "__main__":
    main()
