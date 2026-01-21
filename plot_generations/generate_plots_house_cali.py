import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from learning_rules_MLP import (
    MLP,
    three_factor_weight_step,
    weight_perturb_step,
    backprop_step,
    three_factor_activation_step,
)

# ---------------------------------------------------
# Data Loading
# ---------------------------------------------------

def load_cali():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1),
    )


# ---------------------------------------------------
# One trial (one seed)
# ---------------------------------------------------

def run_single_trial(seed, X_train, y_train, X_test, y_test, epochs=100, batch_size=256):
    """
    Runs one trial for all methods. Returns:
      - train_losses: dict[method] -> tensor[epochs]
      - test_losses: dict[method] -> tensor[epochs]
      - models: dict[method] -> trained model after last epoch
    """
    torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    dims = (input_dim, 128, 64, 1)

    model_bp = MLP(dims, activation=torch.sigmoid, require_grad=True)
    model_wp = MLP(dims, activation=torch.sigmoid)
    model_wp3 = MLP(dims, activation=torch.sigmoid)
    model_np = MLP(dims, activation=torch.sigmoid)

    optimizer_bp = torch.optim.SGD(model_bp.parameters(), lr=0.02)

    N = X_train.size(0)

    train_losses = {m: [] for m in ["bp", "wp", "wp3", "np"]}
    test_losses = {m: [] for m in ["bp", "wp", "wp3", "np"]}

    for epoch in range(epochs):
        print(f" Epoch {epoch+1}/{epochs} ")
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            # Backprop
            _ = backprop_step(model_bp, X_batch, y_batch, optimizer=optimizer_bp)

            # Weight perturbation (2-factor)
            _ = weight_perturb_step(model_wp, X_batch, y_batch, eta=0.07, sigma=0.1)

            # 3-factor weight perturbation
            _ = three_factor_weight_step(model_wp3, X_batch, y_batch, eta=0.07, sigma=0.1)

            # Node perturbation
            _ = three_factor_activation_step(model_np, X_batch, y_batch, eta=0.02, sigma=0.2)

        # Per-epoch evaluation
        with torch.no_grad():
            train_losses["bp"].append(F.mse_loss(model_bp(X_train), y_train).item())
            train_losses["wp"].append(F.mse_loss(model_wp(X_train), y_train).item())
            train_losses["wp3"].append(F.mse_loss(model_wp3(X_train), y_train).item())
            train_losses["np"].append(F.mse_loss(model_np(X_train), y_train).item())

            test_losses["bp"].append(F.mse_loss(model_bp(X_test), y_test).item())
            test_losses["wp"].append(F.mse_loss(model_wp(X_test), y_test).item())
            test_losses["wp3"].append(F.mse_loss(model_wp3(X_test), y_test).item())
            test_losses["np"].append(F.mse_loss(model_np(X_test), y_test).item())

    for m in train_losses:
        train_losses[m] = torch.tensor(train_losses[m])
        test_losses[m] = torch.tensor(test_losses[m])

    models = {
        "bp": model_bp,
        "wp": model_wp,
        "wp3": model_wp3,
        "np": model_np,
    }

    return train_losses, test_losses, models


# ---------------------------------------------------
# Main multi-run loop
# ---------------------------------------------------

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cali()

    epochs = 100
    batch_size = 256
    num_runs = 1
    seeds = list(range(num_runs))
    MSE_THRESHOLD = 0.6  # choose your threshold

    methods = ["bp", "wp", "wp3", "np"]
    labels = {
        "bp": "Backprop",
        "wp": "Weight perturbation",
        "wp3": "3-factor weight perturbation",
        "np": "3-factor node perturbation",
    }
    colors = {"bp": "C0", "wp": "C1", "wp3": "C2", "np": "C3"}

    N_batches = math.ceil(X_train.size(0) / batch_size)

    # compute units per batch
    cost_batch = {"bp": 2.5, "wp": 1.5, "wp3": 2.0, "np": 2.25}
    cost_epoch = {m: cost_batch[m] * N_batches for m in methods}

    all_train = {m: [] for m in methods}
    all_test = {m: [] for m in methods}

    best_epoch_to_threshold = {m: None for m in methods}
    best_train_curve = {m: None for m in methods}
    best_test_curve = {m: None for m in methods}
    best_state = {m: None for m in methods}
    best_fallback = {m: float("inf") for m in methods}

    # Run trials
    for seed in seeds:
        print(f"Run {seed}")
        train_losses, test_losses, models = run_single_trial(
            seed, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size
        )

        for m in methods:
            all_train[m].append(train_losses[m])
            all_test[m].append(test_losses[m])

        # Select best run per method based on earliest threshold crossing
        for m in methods:
            test_curve = test_losses[m]
            below = torch.nonzero(test_curve <= MSE_THRESHOLD, as_tuple=False)

            if below.numel() > 0:
                eq = int(below[0].item())
                if best_epoch_to_threshold[m] is None or eq < best_epoch_to_threshold[m]:
                    best_epoch_to_threshold[m] = eq
                    best_train_curve[m] = train_losses[m].clone()
                    best_test_curve[m] = test_losses[m].clone()
                    best_state[m] = copy.deepcopy(models[m].state_dict())
            else:
                final_test = float(test_curve[-1].item())
                if best_epoch_to_threshold[m] is None and final_test < best_fallback[m]:
                    best_fallback[m] = final_test
                    best_train_curve[m] = train_losses[m].clone()
                    best_test_curve[m] = test_losses[m].clone()
                    best_state[m] = copy.deepcopy(models[m].state_dict())

    # Restore best models
    best_models = {}
    input_dim = X_train.shape[1]
    dims = (input_dim, 128, 64, 1)
    for m in methods:
        model = MLP(dims, activation=torch.sigmoid, require_grad=(m == "bp"))
        model.load_state_dict(best_state[m])
        best_models[m] = model

    # -----------------------------------------
    # Plot 1: best loss curves per epoch
    # -----------------------------------------
    plt.figure(figsize=(8, 5))
    x = np.arange(epochs)
    for m in methods:
        plt.plot(x, best_train_curve[m], label=f"{labels[m]}", color=colors[m])
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title(f"Cali Housing – best training curves (threshold={MSE_THRESHOLD})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # Plot 2: best loss curves vs compute
    # -----------------------------------------
    plt.figure(figsize=(8, 5))
    for m in methods:
        comp_x = (x + 1) * cost_epoch[m]
        plt.plot(comp_x, best_train_curve[m], label=f"{labels[m]}", color=colors[m])
    plt.xlabel("Compute units (forward-pass equivalents)")
    plt.ylabel("Training MSE")
    plt.title("Cali Housing – best loss vs compute")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # Plot 3: variance shading per method
    # -----------------------------------------

    def shade_plot(x, curves, label, color):
        arr = torch.stack(curves, dim=0).cpu().numpy()
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        plt.plot(x, mean, color=color, label=label)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    for m in methods:
        plt.figure(figsize=(8, 5))
        shade_plot(x, all_train[m], labels[m], colors[m])
        plt.xlabel("Epoch")
        plt.ylabel("Training MSE")
        plt.title(f"Cali Housing – variance: {labels[m]}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -----------------------------------------
    # Plot 4: prediction scatter (best models)
    # -----------------------------------------

    with torch.no_grad():
        preds = {m: best_models[m](X_test) for m in methods}
        test_mse_best = {m: F.mse_loss(preds[m], y_test).item() for m in methods}

    plt.figure(figsize=(12, 10))
    for i, m in enumerate(methods):
        plt.subplot(2, 2, i + 1)
        plt.scatter(
            y_test.numpy(),
            preds[m].numpy(),
            alpha=0.3,
            s=10,
            color=colors[m],
        )
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
        plt.title(f"{labels[m]} (MSE={test_mse_best[m]:.4f})")
        plt.xlabel("True (normalized)")
        plt.ylabel("Predicted (normalized)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Done.")
