from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from learning_rules_MLP import (
    MLP,
    init_signed_lognormal_weights,
    three_factor_weight_step,
    weight_perturb_step,
    weight_perturb_step_multiplicative,
    backprop_step,
    weight_perturb_step_momentum,
    three_factor_activation_step,
    three_factor_activation_step_noisy,
)


if __name__ == "__main__":
    # Change this one line to switch method
    METHOD = "npn"  # options: "bp", "wp", "wp3", "wp-m", "np", "npn", "wp-mult"

    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X_housing = housing.data
    y_housing = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_housing, y_housing, test_size=0.2, random_state=None
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    print(f"Train samples: {X_train_t.size(0)}, Test samples: {X_test_t.size(0)}")
    print(f"Input features: {X_train_t.size(1)}")

    input_dim = X_train.shape[1]
    dimensions = (input_dim, 128, 64, 1)
    model = MLP(dimensions, activation=torch.sigmoid, require_grad=(METHOD == "bp"))
    if METHOD == "wp-mult":
        init_signed_lognormal_weights(model, log_mu=-2.0, log_sigma=1.0, p_inhib=0.2, by_neuron=True)
        for i, layer in enumerate(model.layers):
            w = layer.weight
            print(
                f"Init layer {i} weights finite={torch.isfinite(w).all().item()} max|w|={w.abs().max().item():.4e}"
            )

    optimizer_bp = torch.optim.SGD(model.parameters(), lr=0.01) if METHOD == "bp" else None

    momentum_w = None
    momentum_b = None
    if METHOD == "wp-m":
        momentum_w = [torch.zeros_like(layer.weight) for layer in model.layers]
        momentum_b = [torch.zeros_like(layer.bias) for layer in model.layers]

    step_fns = {
        "bp": lambda xb, yb: backprop_step(model, xb, yb, optimizer=optimizer_bp),
        "wp": lambda xb, yb: weight_perturb_step(model, xb, yb, eta=0.05, sigma=0.1),
        "wp3": lambda xb, yb: three_factor_weight_step(model, xb, yb, eta=0.05, sigma=0.1),
        "np": lambda xb, yb: three_factor_activation_step(model, xb, yb, eta=0.01, sigma=0.1),
        "npn": lambda xb, yb: three_factor_activation_step_noisy(model, xb, yb, eta=0.005, sigma=0.1),
        "wp-mult": lambda xb, yb: weight_perturb_step_multiplicative(
            model, xb, yb, eta=0.05, sigma=0.1, max_mult_step=0.1
        ),
    }

    def step_wp_m(xb, yb):
        nonlocal_momentum_w = momentum_w
        nonlocal_momentum_b = momentum_b
        loss, new_mw, new_mb = weight_perturb_step_momentum(
            model, xb, yb, nonlocal_momentum_w, nonlocal_momentum_b, eta=0.01, sigma=0.1
        )
        return loss, new_mw, new_mb

    epochs = 100
    batch_size = 256
    updates_per_epoch = (X_train_t.size(0) + batch_size - 1) // batch_size

    train_losses = []
    test_losses = []

    print("\nTraining model...")
    for epoch in range(epochs):
        perm = torch.randperm(X_train_t.size(0))
        for start in range(0, X_train_t.size(0), batch_size):
            end = min(start + batch_size, X_train_t.size(0))
            idx = perm[start:end]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            if METHOD == "wp-m":
                loss, momentum_w, momentum_b = step_wp_m(X_batch, y_batch)
            else:
                _ = step_fns[METHOD](X_batch, y_batch)

        with torch.no_grad():
            y_pred_train = model(X_train_t)
            y_pred_test = model(X_test_t)
            train_loss = F.mse_loss(y_pred_train, y_train_t, reduction="mean").item()
            test_loss = F.mse_loss(y_pred_test, y_test_t, reduction="mean").item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(
            f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Method: {METHOD}"
        )

    with torch.no_grad():
        y_pred_test = model(X_test_t)
        final_test_mse = F.mse_loss(y_pred_test, y_test_t, reduction="mean").item()

    print(f"\nFinal Test MSE: {final_test_mse:.6f}")

    for i, layer in enumerate(model.layers):
        w = layer.weight
        print(
            f"Layer {i} weights finite={torch.isfinite(w).all().item()} max|w|={w.abs().max().item():.4e}"
        )

    updates_plot = [(i + 1) * updates_per_epoch for i in range(epochs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(updates_plot, train_losses, label=f"{METHOD} Train", marker="o")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Train MSE")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(updates_plot, test_losses, label=f"{METHOD} Test", marker="s")
    ax2.set_xlabel("Update")
    ax2.set_ylabel("Test MSE")
    ax2.set_title("Test Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_t.numpy(), y_pred_test.numpy(), alpha=0.3, s=10)
    plt.plot(
        [y_test_t.min(), y_test_t.max()],
        [y_test_t.min(), y_test_t.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    plt.xlabel("True Values (normalized)")
    plt.ylabel("Predictions (normalized)")
    plt.title(f"{METHOD}\nTest MSE: {final_test_mse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    model.plot_weight_distributions(
        title=f"{METHOD} weight distributions",
        bins=40,
        include_bias=True,
        show=True,
    )

    print("Done")
