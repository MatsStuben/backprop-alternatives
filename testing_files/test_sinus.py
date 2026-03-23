from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from learning_rules_MLP import MLP, backprop_step, node_perturbation_step, weight_perturb_step


METHODS = ["bp", "np", "wp"]
PERTURBATION_SIGMA = 0.1
METHOD_CONFIG = {
    "bp": {"label": "Backprop", "color": "C0", "lr": 0.05, "requires_grad": True},
    "np": {"label": "Node Perturbation", "color": "C1", "lr": 0.02, "requires_grad": False},
    "wp": {"label": "Weight Perturbation", "color": "C2", "lr": 0.006, "requires_grad": False},
}
AUTO_MATCH_NP_LR_TO_WP = True
LR_CALIBRATION_BATCHES = 8
LR_CALIBRATION_DRAWS = 16

SEED = 0
TRAIN_SAMPLES = 512
TEST_SAMPLES = 512
TRAIN_NOISE_STD = 0.1
DIMENSIONS = (1, 32, 16, 1)
BATCH_SIZE = 64
EPOCHS = 1000
METRIC_EVERY = 1
PRINT_EVERY = 20


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


def estimator_update_vector(model, method, xb, yb):
    if method == "bp":
        _, _, true_grad = true_gradient(model, xb, yb)
        return -true_grad
    if method == "np":
        _, _, estimator_update = node_perturbation_gradient_estimate(model, xb, yb, PERTURBATION_SIGMA)
        return estimator_update
    if method == "wp":
        _, _, estimator_update = weight_perturbation_gradient_estimate(model, xb, yb, PERTURBATION_SIGMA)
        return estimator_update
    raise ValueError(f"Unknown method: {method}")


def average_estimator_norm(model, method, x, y, batch_size, num_batches, num_draws):
    norms = []
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min(batch_start + batch_size, x.size(0))
        xb = x[batch_start:batch_end]
        yb = y[batch_start:batch_end]
        draw_norms = []
        for _ in range(num_draws):
            estimator_update = estimator_update_vector(model, method, xb, yb)
            draw_norms.append(float(torch.norm(estimator_update)))
        norms.append(sum(draw_norms) / len(draw_norms))
    return sum(norms) / len(norms)


def maybe_calibrate_np_learning_rate(models, x_train, y_train):
    if not AUTO_MATCH_NP_LR_TO_WP:
        return

    np_norm = average_estimator_norm(
        models["np"],
        "np",
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        num_batches=LR_CALIBRATION_BATCHES,
        num_draws=LR_CALIBRATION_DRAWS,
    )
    wp_norm = average_estimator_norm(
        models["wp"],
        "wp",
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        num_batches=LR_CALIBRATION_BATCHES,
        num_draws=LR_CALIBRATION_DRAWS,
    )

    old_np_lr = METHOD_CONFIG["np"]["lr"]
    matched_np_lr = METHOD_CONFIG["wp"]["lr"] * wp_norm / max(np_norm, 1e-12)
    METHOD_CONFIG["np"]["lr"] = matched_np_lr

    print(
        "Calibrated NP learning rate to match WP estimator norm: "
        f"np_norm={np_norm:.4f}, wp_norm={wp_norm:.4f}, "
        f"old_np_lr={old_np_lr:.4f}, new_np_lr={matched_np_lr:.4f}"
    )


def gradient_metrics(model, method, xb, yb):
    _, _, true_grad = true_gradient(model, xb, yb)
    true_update = -true_grad
    if method == "bp":
        estimator_update = true_update
    elif method == "np":
        _, _, estimator_update = node_perturbation_gradient_estimate(model, xb, yb, PERTURBATION_SIGMA)
    elif method == "wp":
        _, _, estimator_update = weight_perturbation_gradient_estimate(model, xb, yb, PERTURBATION_SIGMA)
    else:
        raise ValueError(f"Unknown method: {method}")

    diff = estimator_update - true_update
    cosine = cosine_similarity_safe(estimator_update, true_update)
    variance_estimate = float(diff.pow(2).mean())
    estimator_norm = float(torch.norm(estimator_update))
    true_update_norm = float(torch.norm(true_update))
    projection = float(torch.dot(estimator_update, true_update) / (true_update_norm + 1e-12))
    return cosine, variance_estimate, estimator_norm, true_update_norm, projection


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
        return backprop_step(model, xb, yb, optimizer=optimizer)
    if method == "np":
        return node_perturbation_step(model, xb, yb, eta=config["lr"], sigma=PERTURBATION_SIGMA)
    if method == "wp":
        return weight_perturb_step(model, xb, yb, eta=config["lr"], sigma=PERTURBATION_SIGMA)
    raise ValueError(f"Unknown method: {method}")


def plot_average_gradient_metrics(cosine_history, variance_history):
    methods_to_compare = [method for method in METHODS if method in {"np", "wp"}]
    labels = [METHOD_CONFIG[method]["label"] for method in methods_to_compare]
    colors = [METHOD_CONFIG[method]["color"] for method in methods_to_compare]
    mean_cosines = [sum(cosine_history[method]) / max(len(cosine_history[method]), 1) for method in methods_to_compare]
    mean_variances = [sum(variance_history[method]) / max(len(variance_history[method]), 1) for method in methods_to_compare]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, mean_cosines, color=colors)
    axes[0].set_title("Average Cosine Similarity")
    axes[0].set_ylabel("Cosine")

    axes[1].bar(labels, mean_variances, color=colors)
    axes[1].set_title("Average Gradient Variance")
    axes[1].set_ylabel("Mean squared error")

    fig.tight_layout()


def plot_cosine_distributions(cosine_history):
    methods_to_compare = [method for method in METHODS if method in {"np", "wp"}]
    fig, axes = plt.subplots(1, len(methods_to_compare), figsize=(10, 4), sharey=True)

    if len(methods_to_compare) == 1:
        axes = [axes]

    for axis, method in zip(axes, methods_to_compare):
        axis.hist(cosine_history[method], bins=30, color=METHOD_CONFIG[method]["color"], alpha=0.8)
        axis.set_title(f"{METHOD_CONFIG[method]['label']} Cosine Distribution")
        axis.set_xlabel("Cosine similarity")
        axis.set_ylabel("Count")

    fig.tight_layout()


def main():
    x_train, y_train, x_test, y_test = generate_sinus_data(
        n_train=TRAIN_SAMPLES,
        n_test=TEST_SAMPLES,
        noise_std=TRAIN_NOISE_STD,
        seed=SEED,
    )
    models, optimizers = make_model_copies()
    maybe_calibrate_np_learning_rate(models, x_train, y_train)

    iterations = []
    train_loss_history = {method: [] for method in METHODS}
    test_loss_history = {method: [] for method in METHODS}
    cosine_history = {method: [] for method in METHODS}
    variance_history = {method: [] for method in METHODS}
    estimator_norm_history = {method: [] for method in METHODS}
    true_update_norm_history = {method: [] for method in METHODS}
    projection_history = {method: [] for method in METHODS}

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
                cosine, variance_estimate, estimator_norm, true_update_norm, projection = gradient_metrics(model, method, xb, yb)
                step_method(method, model, optimizers.get(method), xb, yb)

                if iteration % METRIC_EVERY == 0:
                    train_loss_history[method].append(evaluate_loss(model, x_train, y_train))
                    test_loss_history[method].append(evaluate_loss(model, x_test, y_test))
                    cosine_history[method].append(cosine)
                    variance_history[method].append(variance_estimate)
                    estimator_norm_history[method].append(estimator_norm)
                    true_update_norm_history[method].append(true_update_norm)
                    projection_history[method].append(projection)

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
    plot_average_gradient_metrics(cosine_history, variance_history)
    plot_cosine_distributions(cosine_history)
    plt.show()


if __name__ == "__main__":
    main()
