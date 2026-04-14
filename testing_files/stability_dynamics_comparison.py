from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from learning_rules_MLP import MLP, backprop_step, node_perturbation_step, weight_perturb_step


METHODS = ["bp", "np", "wp"]
METHOD_CONFIG = {
    "bp": {"label": "Backprop", "color": "C0", "lr": 0.05, "requires_grad": True},
    "np": {"label": "Node Perturbation", "color": "C1", "lr": 0.05, "requires_grad": False},
    "wp": {"label": "Weight Perturbation", "color": "C2", "lr": 0.018, "requires_grad": False},
}
NOISE_CONDITIONS = {
    "noise_free_teacher": {"label": "Noise-Free Teacher", "teacher_noise_std": 0.0},
    "noisy_teacher": {"label": "Noisy Teacher", "teacher_noise_std": 0.1},
}

SEED = 0
DIMENSIONS = (1, 32, 16, 1)
PERTURBATION_SIGMA = 0.1
TRAIN_SAMPLES = 512
TEST_SAMPLES = 512
X_RANGE = (-2.0 * math.pi, 2.0 * math.pi)
EPOCHS = int(os.environ.get("STABILITY_EPOCHS", "50000"))
METRIC_EVERY = int(os.environ.get("STABILITY_METRIC_EVERY", "1"))
PRINT_EVERY = int(os.environ.get("STABILITY_PRINT_EVERY", "500"))


def generate_teacher_student_sinus(n_train, n_test, teacher_noise_std, seed):
    generator = torch.Generator().manual_seed(seed)
    x_train = (torch.rand(n_train, 1, generator=generator) * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0])
    y_train_clean = torch.sin(x_train)
    y_train_noisy = y_train_clean + teacher_noise_std * torch.randn(x_train.shape, generator=generator)

    x_test = torch.linspace(X_RANGE[0], X_RANGE[1], n_test).unsqueeze(1)
    y_test_clean = torch.sin(x_test)
    return x_train, y_train_clean, y_train_noisy, x_test, y_test_clean


def evaluate_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return float(F.mse_loss(prediction, y, reduction="mean"))


def weight_norm(model):
    with torch.no_grad():
        squared_sum = torch.tensor(0.0)
        for layer in model.layers:
            squared_sum = squared_sum + layer.weight.detach().pow(2).sum()
        return float(torch.sqrt(squared_sum))


def bias_norm(model):
    with torch.no_grad():
        squared_sum = torch.tensor(0.0)
        for layer in model.layers:
            squared_sum = squared_sum + layer.bias.detach().pow(2).sum()
        return float(torch.sqrt(squared_sum))


def layer_weight_norms(model):
    with torch.no_grad():
        return [float(torch.norm(layer.weight.detach())) for layer in model.layers]


def make_models():
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


def step_method(method, model, optimizer, x_train, y_train):
    config = METHOD_CONFIG[method]
    if method == "bp":
        return backprop_step(model, x_train, y_train, optimizer=optimizer)
    if method == "np":
        return node_perturbation_step(model, x_train, y_train, eta=config["lr"], sigma=PERTURBATION_SIGMA)
    if method == "wp":
        return weight_perturb_step(model, x_train, y_train, eta=config["lr"], sigma=PERTURBATION_SIGMA)
    raise ValueError(f"Unknown method: {method}")


def initialize_history():
    return {
        method: {
            "train_target_loss": [],
            "train_clean_error": [],
            "test_clean_error": [],
            "weight_norm": [],
            "bias_norm": [],
            "layer_weight_norms": [],
        }
        for method in METHODS
    }


def record_metrics(history, models, x_train, y_train_clean, y_train_target, x_test, y_test_clean):
    for method, model in models.items():
        history[method]["train_target_loss"].append(evaluate_loss(model, x_train, y_train_target))
        history[method]["train_clean_error"].append(evaluate_loss(model, x_train, y_train_clean))
        history[method]["test_clean_error"].append(evaluate_loss(model, x_test, y_test_clean))
        history[method]["weight_norm"].append(weight_norm(model))
        history[method]["bias_norm"].append(bias_norm(model))
        history[method]["layer_weight_norms"].append(layer_weight_norms(model))


def run_condition(condition_name, teacher_noise_std):
    x_train, y_train_clean, y_train_target, x_test, y_test_clean = generate_teacher_student_sinus(
        TRAIN_SAMPLES,
        TEST_SAMPLES,
        teacher_noise_std=teacher_noise_std,
        seed=SEED,
    )
    models, optimizers = make_models()
    history = initialize_history()
    record_metrics(history, models, x_train, y_train_clean, y_train_target, x_test, y_test_clean)

    for epoch in range(1, EPOCHS + 1):
        for method in METHODS:
            step_method(method, models[method], optimizers.get(method), x_train, y_train_target)

        if epoch % METRIC_EVERY == 0:
            record_metrics(history, models, x_train, y_train_clean, y_train_target, x_test, y_test_clean)

        if epoch % PRINT_EVERY == 0:
            print(f"{NOISE_CONDITIONS[condition_name]['label']} | epoch {epoch}")
            for method in METHODS:
                train_clean = history[method]["train_clean_error"][-1]
                test_clean = history[method]["test_clean_error"][-1]
                norm_value = history[method]["weight_norm"][-1]
                print(
                    f"  {METHOD_CONFIG[method]['label']}: "
                    f"train_clean={train_clean:.4f}, test_clean={test_clean:.4f}, weight_norm={norm_value:.4f}"
                )

    return {
        "x_train": x_train,
        "y_train_clean": y_train_clean,
        "y_train_target": y_train_target,
        "x_test": x_test,
        "y_test_clean": y_test_clean,
        "models": models,
        "history": history,
    }


def plot_condition(condition_name, result):
    history = result["history"]
    epochs = [index * METRIC_EVERY for index in range(len(next(iter(history.values()))["weight_norm"]))]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(NOISE_CONDITIONS[condition_name]["label"])

    for method in METHODS:
        config = METHOD_CONFIG[method]
        axes[0, 0].plot(epochs, history[method]["train_target_loss"], color=config["color"], label=config["label"])
        axes[0, 1].plot(epochs, history[method]["train_clean_error"], color=config["color"], label=config["label"])
        axes[0, 2].plot(epochs, history[method]["test_clean_error"], color=config["color"], label=config["label"])
        axes[1, 0].plot(epochs, history[method]["weight_norm"], color=config["color"], label=config["label"])
        axes[1, 1].plot(epochs, history[method]["bias_norm"], color=config["color"], label=config["label"])
        axes[1, 2].plot(
            history[method]["weight_norm"],
            history[method]["train_clean_error"],
            color=config["color"],
            label=config["label"],
            alpha=0.9,
        )

    axes[0, 0].set_title("Train Loss vs Training Targets")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")

    axes[0, 1].set_title("Train Error vs Clean Teacher")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE")

    axes[0, 2].set_title("Test Error vs Clean Teacher")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("MSE")

    axes[1, 0].set_title("Weight Norm")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel(r"$||W||$")

    axes[1, 1].set_title("Bias Norm")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel(r"$||b||$")

    axes[1, 2].set_title("Phase Plot: Clean Error vs Weight Norm")
    axes[1, 2].set_xlabel(r"$||W||$")
    axes[1, 2].set_ylabel("Clean Train Error")

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    axes[0, 0].legend()
    fig.tight_layout()

    layer_fig, layer_axes = plt.subplots(1, len(METHODS), figsize=(18, 4), sharey=True)
    layer_fig.suptitle(f"{NOISE_CONDITIONS[condition_name]['label']} Layer Weight Norms")

    for axis, method in zip(layer_axes, METHODS):
        config = METHOD_CONFIG[method]
        layer_norm_history = history[method]["layer_weight_norms"]
        for layer_index in range(len(layer_norm_history[0])):
            layer_curve = [layer_values[layer_index] for layer_values in layer_norm_history]
            axis.plot(epochs, layer_curve, label=f"Layer {layer_index}")
        axis.set_title(config["label"])
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    layer_axes[0].set_ylabel(r"$||W^\ell||$")
    layer_axes[-1].legend()
    layer_fig.tight_layout()


def print_final_summary(results):
    print("\nFinal stability summary")
    for condition_name, result in results.items():
        print(NOISE_CONDITIONS[condition_name]["label"])
        history = result["history"]
        for method in METHODS:
            train_target = history[method]["train_target_loss"][-1]
            train_clean = history[method]["train_clean_error"][-1]
            test_clean = history[method]["test_clean_error"][-1]
            w_norm = history[method]["weight_norm"][-1]
            b_norm = history[method]["bias_norm"][-1]
            print(
                f"  {METHOD_CONFIG[method]['label']}: "
                f"train_target={train_target:.4f}, train_clean={train_clean:.4f}, "
                f"test_clean={test_clean:.4f}, weight_norm={w_norm:.4f}, bias_norm={b_norm:.4f}"
            )


def main():
    results = {}
    for condition_name, condition in NOISE_CONDITIONS.items():
        results[condition_name] = run_condition(condition_name, teacher_noise_std=condition["teacher_noise_std"])
        plot_condition(condition_name, results[condition_name])

    print_final_summary(results)
    plt.show()


if __name__ == "__main__":
    main()
