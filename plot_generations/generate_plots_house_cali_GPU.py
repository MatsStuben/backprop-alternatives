from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
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

def load_cali(device="cpu"):
    """
    Load California Housing, normalize X and y, and return tensors on `device`.
    """
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

    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    return X_train_t, y_train_t, X_test_t, y_test_t


# ---------------------------------------------------
# One trial (one seed)
# ---------------------------------------------------

def run_single_trial(seed,
                     X_train,
                     y_train,
                     X_test,
                     y_test,
                     epochs=100,
                     batch_size=256,
                     device="cpu"):
    """
    Runs one trial for all methods. Returns:
      - train_losses: dict[method] -> tensor[epochs]
      - test_losses: dict[method] -> tensor[epochs]
      - models: dict[method] -> trained model after last epoch
    """
    torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    dims = (input_dim, 128, 64, 1)

    model_bp = MLP(dims, activation=torch.sigmoid, require_grad=True).to(device)
    model_wp = MLP(dims, activation=torch.sigmoid).to(device)
    model_wp3 = MLP(dims, activation=torch.sigmoid).to(device)
    model_np = MLP(dims, activation=torch.sigmoid).to(device)

    optimizer_bp = torch.optim.SGD(model_bp.parameters(), lr=0.02)

    N = X_train.size(0)

    train_losses = {m: [] for m in ["bp", "wp", "wp3", "np"]}
    test_losses =  {m: [] for m in ["bp", "wp", "wp3", "np"]}

    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f"  Seed {seed}: epoch {epoch}/{epochs}", flush=True)

        perm = torch.randperm(N, device=device)
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

    # Convert to tensors [epochs]
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

def save_figure(path_base):
    plt.savefig(f"{path_base}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    print("Hello mate, starting Cali Housing experiment...")
    os.makedirs("plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cali Housing experiment running on:", device, flush=True)

    X_train, y_train, X_test, y_test = load_cali(device=device)

    epochs = 100
    batch_size = 256
    num_runs = 5          # increase if you want better variance estimates
    seeds = list(range(num_runs))
    MSE_THRESHOLD = 0.6   # "good enough" threshold

    methods = ["bp", "wp", "wp3", "np"]
    labels = {
        "bp":  "Backprop",
        "wp":  "Weight perturbation",
        "wp3": "3-factor weight perturbation",
        "np":  "3-factor node perturbation",
    }
    colors = {"bp": "C0", "wp": "C1", "wp3": "C2", "np": "C3"}

    N_batches = math.ceil(X_train.size(0) / batch_size)

    # cost in forward-pass equivalents
    cost_batch = {"bp": 2.0, "wp": 1.5, "wp3": 1.75, "np": 1.6}
    cost_epoch = {m: cost_batch[m] * N_batches for m in methods}

    all_train = {m: [] for m in methods}
    all_test  = {m: [] for m in methods}

    best_epoch_to_threshold = {m: None for m in methods}
    best_train_curve = {m: None for m in methods}
    best_test_curve  = {m: None for m in methods}
    best_state       = {m: None for m in methods}
    best_fallback    = {m: float("inf") for m in methods}

    # Run trials
    for seed in seeds:
        print(f"\nRun {seed}", flush=True)
        train_losses, test_losses, models = run_single_trial(
            seed,
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
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

    # Restore best models on `device`
    best_models = {}
    input_dim = X_train.shape[1]
    dims = (input_dim, 128, 64, 1)

    for m in methods:
        model = MLP(dims, activation=torch.sigmoid, require_grad=(m == "bp")).to(device)
        model.load_state_dict(best_state[m])
        best_models[m] = model

    # =========================================
    # Plot 1: best loss curves per update
    # =========================================
    epochs_arr = np.arange(epochs)
    updates_arr = (epochs_arr + 1) * N_batches

    plt.figure(figsize=(8, 5))
    for m in methods:
        plt.plot(
            updates_arr,
            best_train_curve[m].cpu().numpy(),
            label=f"{labels[m]}",
            color=colors[m],
        )
    plt.xlabel("Update")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.tight_layout()
    save_figure("plots/cali_best_training_vs_epoch")
    plt.close()

    # =========================================
    # Plot 2: best loss curves vs compute
    # =========================================
    plt.figure(figsize=(8, 5))
    for m in methods:
        comp_x = (epochs_arr + 1) * cost_epoch[m]
        plt.plot(
            comp_x,
            best_train_curve[m].cpu().numpy(),
            label=f"{labels[m]}",
            color=colors[m],
        )
    plt.xlabel("Compute units (forward-pass equivalents)")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.tight_layout()
    save_figure("plots/cali_best_training_vs_compute")
    plt.close()

    # =========================================
    # Plot 3: variance shading per method
    # =========================================
    def shade_plot(xvals, curves, label, color):
        arr = torch.stack(curves, dim=0).cpu().numpy()
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        plt.plot(xvals, mean, color=color, label=label)
        plt.fill_between(xvals, mean - std, mean + std, color=color, alpha=0.2)

    for m in methods:
        plt.figure(figsize=(8, 5))
        shade_plot(updates_arr, all_train[m], labels[m], colors[m])
        plt.xlabel("Update")
        plt.ylabel("Training MSE")
        plt.legend()
        plt.tight_layout()
        save_figure(f"plots/cali_variance_training_{m}")
        plt.close()

    # =========================================
    # Plot 4: prediction scatter (best models)
    # =========================================
    with torch.no_grad():
        preds = {m: best_models[m](X_test) for m in methods}
        test_mse_best = {m: F.mse_loss(preds[m], y_test).item() for m in methods}

    y_test_cpu = y_test.cpu().numpy()

    plt.figure(figsize=(12, 10))
    for i, m in enumerate(methods):
        plt.subplot(2, 2, i + 1)
        pm = preds[m].cpu().numpy()
        plt.scatter(
            y_test_cpu,
            pm,
            alpha=0.3,
            s=10,
            color=colors[m],
        )
        ymin, ymax = y_test_cpu.min(), y_test_cpu.max()
        plt.plot([ymin, ymax], [ymin, ymax], "k--")
        plt.xlabel("True (normalized)")
        plt.ylabel("Predicted (normalized)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure("plots/cali_prediction_scatter_best_models")
    plt.close()

    # -----------------------------
    # Plot 5: average loss curves (across runs)
    # -----------------------------
    plt.figure(figsize=(8, 5))
    for m in methods:
        arr = torch.stack(all_train[m], dim=0).cpu().numpy()  # (num_runs, epochs)
        mean = arr.mean(axis=0)
        plt.plot(updates_arr, mean, label=labels[m], color=colors[m])
    plt.xlabel("Update")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.tight_layout()
    save_figure("plots/cali_avg_training_vs_epoch")
    plt.close()

    plt.figure(figsize=(8, 5))
    for m in methods:
        arr = torch.stack(all_train[m], dim=0).cpu().numpy()  # (num_runs, epochs)
        mean = arr.mean(axis=0)
        comp_x = (epochs_arr + 1) * cost_epoch[m]
        plt.plot(comp_x, mean, label=labels[m], color=colors[m])
    plt.xlabel("Compute units (forward-pass equivalents)")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.tight_layout()
    save_figure("plots/cali_avg_training_vs_compute")
    plt.close()

    # -----------------------------
    # Scalar variance summaries (single number per method)
    # -----------------------------
    print("\nVariance summary (across runs):", flush=True)
    for m in methods:
        train_arr = torch.stack(all_train[m], dim=0)  # (num_runs, epochs)
        test_arr = torch.stack(all_test[m], dim=0)    # (num_runs, epochs)

        mean_var_train = train_arr.var(dim=0, unbiased=False).mean().item()
        mean_std_train = train_arr.std(dim=0, unbiased=False).mean().item()

        mean_var_test = test_arr.var(dim=0, unbiased=False).mean().item()
        mean_std_test = test_arr.std(dim=0, unbiased=False).mean().item()

        print(
            f"  {m:>3} ({labels[m]}): "
            f"train mean-var={mean_var_train:.6f}, train mean-std={mean_std_train:.6f} | "
            f"test mean-var={mean_var_test:.6f}, test mean-std={mean_std_test:.6f}",
            flush=True,
        )

    print("Done. Cali plots saved in plots/", flush=True)