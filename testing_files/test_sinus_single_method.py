from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def generate_data(n=400, noise=0.1, seed=None):
    if seed is None:
        seed = torch.seed()
    torch.manual_seed(seed)
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x, y


if __name__ == "__main__":
    # Change this one line to switch method
    METHOD = "npn"  # options: "bp", "wp", "wp3", "wp-m", "np", "npn", "wp-mult"

    X, Y = generate_data(n=128 * 10, noise=0.1, seed=None)
    dimensions = (1, 8, 4, 1)

    model = MLP(dimensions, activation=torch.sigmoid, require_grad=(METHOD == "bp"))
    if METHOD == "wp-mult":
        init_signed_lognormal_weights(model, log_mu=1.0, log_sigma=0.5, p_inhib=0.2, by_neuron=False)
        for i, layer in enumerate(model.layers):
            w = layer.weight
            print(
                f"Init layer {i} weights finite={torch.isfinite(w).all().item()} max|w|={w.abs().max().item():.4e}"
            )
    optimizer_bp = torch.optim.SGD(model.parameters(), lr=0.1) if METHOD == "bp" else None

    momentum_w = None
    momentum_b = None
    if METHOD == "wp-m":
        momentum_w = [torch.zeros_like(layer.weight) for layer in model.layers]
        momentum_b = [torch.zeros_like(layer.bias) for layer in model.layers]

    step_fns = {
        "bp": lambda xb, yb: backprop_step(model, xb, yb, optimizer=optimizer_bp),
        "wp": lambda xb, yb: weight_perturb_step(model, xb, yb, eta=0.7, sigma=0.2),
        "wp3": lambda xb, yb: three_factor_weight_step(model, xb, yb, eta=0.9, sigma=0.1),
        "np": lambda xb, yb: three_factor_activation_step(model, xb, yb, eta=0.2, sigma=0.1),
        "npn": lambda xb, yb: three_factor_activation_step_noisy(model, xb, yb, eta=0.2, sigma=0.1),
        "wp-mult": lambda xb, yb: weight_perturb_step_multiplicative(
            model, xb, yb, eta=0.7, sigma=0.5, max_mult_step=0.5
        ),
    }

    def step_wp_m(xb, yb):
        nonlocal_momentum_w = momentum_w
        nonlocal_momentum_b = momentum_b
        loss, new_mw, new_mb = weight_perturb_step_momentum(
            model, xb, yb, nonlocal_momentum_w, nonlocal_momentum_b, eta=0.03, sigma=0.1
        )
        return loss, new_mw, new_mb

    epochs = 400
    batch_size = 128

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        for start in range(0, X.size(0), batch_size):
            end = min(start + batch_size, X.size(0))
            idx = perm[start:end]
            X_batch = X[idx]
            Y_batch = Y[idx]

            if METHOD == "wp-m":
                loss, momentum_w, momentum_b = step_wp_m(X_batch, Y_batch)
            else:
                _ = step_fns[METHOD](X_batch, Y_batch)

        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                y_pred = model(X)
                train_loss = F.mse_loss(y_pred, Y, reduction="mean").item()
            print(f"Epoch {epoch:4d}, {METHOD}: {train_loss:.6f}")

    with torch.no_grad():
        xs = torch.linspace(-2 * math.pi, 2 * math.pi, 400).unsqueeze(1)
        ys_pred = model(xs)
        y_true = torch.sin(xs)

    test_mse = F.mse_loss(ys_pred, y_true, reduction="mean").item()
    print(f"Test MSE on grid: {test_mse:.6f}")

    for i, layer in enumerate(model.layers):
        w = layer.weight
        print(
            f"Layer {i} weights finite={torch.isfinite(w).all().item()} max|w|={w.abs().max().item():.4e}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(xs.cpu().numpy(), y_true.cpu().numpy(), label="True sin(x)", color="C0")
    plt.plot(xs.cpu().numpy(), ys_pred.cpu().numpy(), label=f"{METHOD} prediction", color="C1")
    plt.scatter(X.cpu().numpy(), Y.cpu().numpy(), s=10, alpha=0.3, label="Train samples", color="C2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Prediction vs True (MSE={test_mse:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Done")

    model.plot_weight_distributions(
        title=f"{METHOD} weight distributions",
        bins=40,
        include_bias=True,
        show=True,
    )