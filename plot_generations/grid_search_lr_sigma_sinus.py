from pathlib import Path
import sys
import math

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from learning_rules_MLP import (
    MLP,
    init_signed_lognormal_weights,
    three_factor_weight_step,
    three_factor_activation_step,
    three_factor_activation_step_noisy,
    weight_perturb_step,
    weight_perturb_step_momentum,
    weight_perturb_step_multiplicative,
    backprop_step,
)


def generate_sinus_data(n=400, noise=0.1, seed=None):
    if seed is None:
        seed = torch.seed()
    torch.manual_seed(seed)
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x, y


def compute_convergence_metrics(loss_history, tail_frac=0.1):
    n = len(loss_history)
    tail_n = max(1, int(n * tail_frac))
    tail = np.array(loss_history[-tail_n:], dtype=float)

    level = float(np.median(tail))
    var = float(np.var(tail))

    speed = n
    for i, v in enumerate(loss_history, start=1):
        if v <= level:
            speed = i
            break

    return level, speed, var


def run_single_trial(method, X_train, y_train, eta, sigma, epochs, batch_size, seed=None):
    if seed is None:
        seed = torch.seed()
    torch.manual_seed(seed)

    dimensions = (1, 8, 4, 1)
    model = MLP(dimensions, activation=torch.sigmoid, require_grad=(method == "bp"))

    if method == "wp-mult":
        init_signed_lognormal_weights(model, log_mu=0.0, log_sigma=0.5, p_inhib=0.0, by_neuron=False)

    optimizer_bp = torch.optim.SGD(model.parameters(), lr=eta) if method == "bp" else None

    momentum_w = None
    momentum_b = None
    if method == "wp-m":
        momentum_w = [torch.zeros_like(layer.weight) for layer in model.layers]
        momentum_b = [torch.zeros_like(layer.bias) for layer in model.layers]

    loss_history = []
    N = X_train.size(0)

    for _ in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            if method == "bp":
                loss = backprop_step(model, X_batch, y_batch, optimizer=optimizer_bp)
            elif method == "wp":
                loss = weight_perturb_step(model, X_batch, y_batch, eta=eta, sigma=sigma)
            elif method == "wp3":
                loss = three_factor_weight_step(model, X_batch, y_batch, eta=eta, sigma=sigma)
            elif method == "np":
                loss = three_factor_activation_step(model, X_batch, y_batch, eta=eta, sigma=sigma)
            elif method == "npn":
                loss = three_factor_activation_step_noisy(model, X_batch, y_batch, eta=eta, sigma=sigma)
            elif method == "wp-m":
                loss, momentum_w, momentum_b = weight_perturb_step_momentum(
                    model, X_batch, y_batch, momentum_w, momentum_b, eta=eta, sigma=sigma
                )
            elif method == "wp-mult":
                loss = weight_perturb_step_multiplicative(
                    model, X_batch, y_batch, eta=eta, sigma=sigma, max_mult_step=0.5
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            loss_history.append(float(loss))

    return loss_history


def main():
    METHOD = "wp"  # options: bp, wp, wp3, np, npn, wp-m, wp-mult
    EPOCHS = 400
    BATCH_SIZE = 128
    SEEDS = [0, 1, 2]

    ETA_GRID = [0.1, 0.5, 0.7, 0.9]
    SIGMA_GRID = [0.03, 0.1, 0.2, 0.5]

    X, y = generate_sinus_data(n=128 * 10, noise=0.1, seed=None)
    X_train, y_train = X, y

    n_updates = EPOCHS * int(math.ceil(X_train.size(0) / BATCH_SIZE))
    updates_axis = np.arange(1, n_updates + 1)

    levels = np.zeros((len(ETA_GRID), len(SIGMA_GRID)))
    speeds = np.zeros((len(ETA_GRID), len(SIGMA_GRID)))
    variances = np.zeros((len(ETA_GRID), len(SIGMA_GRID)))

    curve_means = {}

    for i, eta in enumerate(ETA_GRID):
        for j, sigma in enumerate(SIGMA_GRID):
            run_levels = []
            run_speeds = []
            run_vars = []
            curves = []
            for seed in SEEDS:
                loss_history = run_single_trial(
                    METHOD,
                    X_train,
                    y_train,
                    eta=eta,
                    sigma=sigma,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    seed=seed,
                )
                level, speed, var = compute_convergence_metrics(loss_history, tail_frac=0.1)
                run_levels.append(level)
                run_speeds.append(speed)
                run_vars.append(var)
                curves.append(np.array(loss_history, dtype=float))

            levels[i, j] = float(np.mean(run_levels))
            speeds[i, j] = float(np.mean(run_speeds))
            variances[i, j] = float(np.mean(run_vars))
            curve_means[(eta, sigma)] = np.mean(np.stack(curves, axis=0), axis=0)

    def plot_table(values, title):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        cell_text = [[f"{v:.4g}" for v in row] for row in values]
        col_labels = [str(v) for v in SIGMA_GRID]
        row_labels = [str(v) for v in ETA_GRID]
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    plot_table(speeds, f"Convergence speed (updates) – {METHOD} / sinus")
    plot_table(levels, f"Convergence level (median last 10%) – {METHOD} / sinus")
    plot_table(variances, f"Variance (last 10%) – {METHOD} / sinus")

    fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True)
    for i, eta in enumerate(ETA_GRID):
        for j, sigma in enumerate(SIGMA_GRID):
            ax = axes[i, j]
            curve = curve_means[(eta, sigma)]
            ax.plot(updates_axis, curve, linewidth=1.0)
            ax.set_title(f"eta={eta}, sigma={sigma}", fontsize=8)
    fig.suptitle(f"Loss curves (mean over seeds) – {METHOD} / sinus", fontsize=12)
    for ax in axes[-1, :]:
        ax.set_xlabel("Update")
    for ax in axes[:, 0]:
        ax.set_ylabel("Train MSE")
    plt.tight_layout()
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
