from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

from learning_rules_MLP import (
    MLP,
    backprop_step,
    node_perturbation_step,
    node_perturbation_step_fixed_sigma,
    weight_perturb_step,
)


METHODS = ["bp", "np", "np_fixed", "wp"]

# Edit one lr and one sigma per method here.
METHOD_CONFIG = {
    "bp": {
        "label": "Backprop",
        "color": "C0",
        "lr": 0.03,
        "sigma": None,
        "requires_grad": True,
    },
    "np": {
        "label": "Node Perturbation",
        "color": "C1",
        "lr": 0.02,
        "sigma": 0.01,
        "requires_grad": False,
    },
    "np_fixed": {
        "label": "Node Perturbation Fixed Sigma",
        "color": "C3",
        "lr": 0.005,
        "sigma": 0.01,
        "requires_grad": False,
    },
    "wp": {
        "label": "Weight Perturbation",
        "color": "C2",
        "lr": 0.005,
        "sigma": 0.01,
        "requires_grad": False,
    },
}

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).resolve().parents[1] / ".data" / "fashion_mnist"

NUM_CLASSES = 10
DIMENSIONS = (28 * 28, 256, 128, NUM_CLASSES)
BATCH_SIZE = 256
EPOCHS = 5
METRIC_EVERY = 1
PRINT_EVERY = 1
PRINT_BATCH_EVERY = 1
TRAIN_LIMIT = 20000
TEST_LIMIT = 5000


def flatten_model_tensors(weight_tensors, bias_tensors):
    pieces = []
    for weight_tensor, bias_tensor in zip(weight_tensors, bias_tensors):
        pieces.append(weight_tensor.reshape(-1))
        pieces.append(bias_tensor.reshape(-1))
    return torch.cat(pieces)


def cosine_similarity_safe(a, b, eps=1e-12):
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    if a_norm.item() < eps or b_norm.item() < eps:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm + eps))


def load_fashion_mnist():
    train_dataset = FashionMNIST(root=DATA_DIR, train=True, download=True)
    test_dataset = FashionMNIST(root=DATA_DIR, train=False, download=True)

    x_train = train_dataset.data.float() / 255.0
    x_test = test_dataset.data.float() / 255.0
    train_labels = train_dataset.targets.long()
    test_labels = test_dataset.targets.long()

    if TRAIN_LIMIT is not None:
        x_train = x_train[:TRAIN_LIMIT]
        train_labels = train_labels[:TRAIN_LIMIT]
    if TEST_LIMIT is not None:
        x_test = x_test[:TEST_LIMIT]
        test_labels = test_labels[:TEST_LIMIT]

    # Mean-centering keeps backprop happy while avoiding the very large
    # input norms that make induced node perturbation excessively noisy.
    mean = x_train.mean()
    x_train = x_train - mean
    x_test = x_test - mean

    x_train = x_train.view(x_train.size(0), -1)
    x_test = x_test.view(x_test.size(0), -1)

    y_train = F.one_hot(train_labels, num_classes=NUM_CLASSES).float()
    y_test = F.one_hot(test_labels, num_classes=NUM_CLASSES).float()

    return (
        x_train.to(DEVICE),
        y_train.to(DEVICE),
        train_labels.to(DEVICE),
        x_test.to(DEVICE),
        y_test.to(DEVICE),
        test_labels.to(DEVICE),
    )


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


def gradient_metrics(unscaled_parameter_update_vector, true_update):
    diff = unscaled_parameter_update_vector - true_update
    cosine = cosine_similarity_safe(unscaled_parameter_update_vector, true_update)
    variance_estimate = float(diff.pow(2).mean())
    true_update_norm = float(torch.norm(true_update))
    projection = float(torch.dot(unscaled_parameter_update_vector, true_update) / (true_update_norm + 1e-12))
    return cosine, variance_estimate, projection


def evaluate_split(model, x, y_one_hot, labels):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(F.mse_loss(logits, y_one_hot, reduction="mean"))
        accuracy = float((logits.argmax(dim=1) == labels).float().mean())
    return loss, accuracy


def make_model_copies():
    torch.manual_seed(SEED)
    base_model = MLP(DIMENSIONS, activation=F.relu, require_grad=True).to(DEVICE)
    base_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}

    models = {}
    optimizers = {}
    for method in METHODS:
        config = METHOD_CONFIG[method]
        model = MLP(DIMENSIONS, activation=F.relu, require_grad=config["requires_grad"]).to(DEVICE)
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
    methods_to_compare = [method for method in METHODS if method in {"np", "np_fixed", "wp"}]
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
    methods_to_compare = [method for method in METHODS if method in {"np", "np_fixed", "wp"}]
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
    torch.manual_seed(SEED)
    x_train, y_train, train_labels, x_test, y_test, test_labels = load_fashion_mnist()
    models, optimizers = make_model_copies()

    iterations = []
    train_loss_history = {method: [] for method in METHODS}
    test_loss_history = {method: [] for method in METHODS}
    train_accuracy_history = {method: [] for method in METHODS}
    test_accuracy_history = {method: [] for method in METHODS}
    cosine_history = {method: [] for method in METHODS}
    variance_history = {method: [] for method in METHODS}
    projection_history = {method: [] for method in METHODS}

    iteration = 0
    batches_per_epoch = (x_train.size(0) + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"Running Fashion-MNIST with device={DEVICE}, train_limit={x_train.size(0)}, "
        f"test_limit={x_test.size(0)}, dims={DIMENSIONS}"
    )
    print(
        "Configs: "
        + " | ".join(
            f"{method}: lr={config['lr']}"
            + (f", sigma={config['sigma']}" if config["sigma"] is not None else "")
            for method, config in METHOD_CONFIG.items()
        )
    )

    for epoch in range(EPOCHS):
        permutation = torch.randperm(x_train.size(0), device=x_train.device)
        for batch_start in range(0, x_train.size(0), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, x_train.size(0))
            batch_indices = permutation[batch_start:batch_end]
            xb = x_train[batch_indices]
            yb = y_train[batch_indices]
            iteration += 1
            batch_number = batch_start // BATCH_SIZE + 1

            for method in METHODS:
                model = models[method]
                _, _, true_grad = true_gradient(model, xb, yb)
                true_update = -true_grad
                _, unscaled_parameter_update_vector = step_method(method, model, optimizers.get(method), xb, yb)
                cosine, variance_estimate, projection = gradient_metrics(unscaled_parameter_update_vector, true_update)

                if iteration % METRIC_EVERY == 0:
                    train_loss, train_accuracy = evaluate_split(model, x_train, y_train, train_labels)
                    test_loss, test_accuracy = evaluate_split(model, x_test, y_test, test_labels)
                    train_loss_history[method].append(train_loss)
                    test_loss_history[method].append(test_loss)
                    train_accuracy_history[method].append(train_accuracy)
                    test_accuracy_history[method].append(test_accuracy)
                    cosine_history[method].append(cosine)
                    variance_history[method].append(variance_estimate)
                    projection_history[method].append(projection)

            if iteration % METRIC_EVERY == 0:
                iterations.append(iteration)

            if batch_number % PRINT_BATCH_EVERY == 0 or batch_number == batches_per_epoch:
                batch_status_parts = []
                for method in METHODS:
                    batch_status_parts.append(
                        f"{method}: train={train_loss_history[method][-1]:.4f}, "
                        f"test={test_loss_history[method][-1]:.4f}, "
                        f"acc={test_accuracy_history[method][-1]:.3f}, "
                        f"cos={cosine_history[method][-1]:.4f}, "
                        f"var={variance_history[method][-1]:.4e}, "
                        f"proj={projection_history[method][-1]:.4f}"
                    )
                print(
                    f"  epoch {epoch + 1:3d}/{EPOCHS} | "
                    f"batch {batch_number:3d}/{batches_per_epoch} | "
                    + " | ".join(batch_status_parts)
                )

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0 or epoch + 1 == EPOCHS:
            status_parts = []
            for method in METHODS:
                status_parts.append(
                    f"{method}: train={train_loss_history[method][-1]:.4f}, "
                    f"test={test_loss_history[method][-1]:.4f}, "
                    f"acc={test_accuracy_history[method][-1]:.3f}, "
                    f"cos={cosine_history[method][-1]:.4f}, "
                    f"var={variance_history[method][-1]:.4e}, "
                    f"proj={projection_history[method][-1]:.4f}"
                )
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} "
                f"({batches_per_epoch} batches/epoch) | " + " | ".join(status_parts)
            )

    fig, axes = plt.subplots(5, 1, figsize=(11, 18), sharex=True)
    for method in METHODS:
        config = METHOD_CONFIG[method]
        axes[0].plot(iterations, train_loss_history[method], label=f"{config['label']} train", color=config["color"])
        axes[0].plot(iterations, test_loss_history[method], linestyle="--", label=f"{config['label']} test", color=config["color"])
        axes[1].plot(iterations, train_accuracy_history[method], label=f"{config['label']} train", color=config["color"])
        axes[1].plot(iterations, test_accuracy_history[method], linestyle="--", label=f"{config['label']} test", color=config["color"])
        axes[2].plot(iterations, cosine_history[method], label=config["label"], color=config["color"])
        axes[3].plot(iterations, variance_history[method], label=config["label"], color=config["color"])
        axes[4].plot(iterations, projection_history[method], label=config["label"], color=config["color"])

    axes[0].set_title("Fashion-MNIST Loss")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[1].set_title("Fashion-MNIST Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[2].set_title("Cosine Similarity to True Gradient")
    axes[2].set_ylabel("Cosine")
    axes[2].legend()
    axes[3].set_title("Estimated Mean Gradient Variance")
    axes[3].set_ylabel("Mean squared error")
    axes[3].legend()
    axes[4].set_title("Projection onto True Update")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("Signed projection")
    axes[4].legend()
    fig.tight_layout()

    plot_average_gradient_metrics(cosine_history, variance_history, projection_history)
    plot_cosine_distributions(cosine_history)
    plt.show()


if __name__ == "__main__":
    main()
