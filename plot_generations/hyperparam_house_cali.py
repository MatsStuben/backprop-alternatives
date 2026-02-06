from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import copy
import math
import itertools
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
# One trial for one method/hparams
# ---------------------------------------------------

def run_one_setting(
    seed,
    method,
    hparams,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=100,
    batch_size=256,
    device="cpu",
):
    torch.manual_seed(seed)

    input_dim = X_train.shape[1]
    dims = (input_dim, 128, 64, 1)

    require_grad = (method == "bp")
    model = MLP(dims, activation=torch.sigmoid, require_grad=require_grad).to(device)

    optimizer = None
    if method == "bp":
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams["lr"])

    N = X_train.size(0)
    train_curve = []
    test_curve = []

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            Xb = X_train[idx]
            yb = y_train[idx]

            if method == "bp":
                _ = backprop_step(model, Xb, yb, optimizer=optimizer)
            elif method == "wp":
                _ = weight_perturb_step(model, Xb, yb, eta=hparams["eta"], sigma=hparams["sigma"])
            elif method == "wp3":
                _ = three_factor_weight_step(model, Xb, yb, eta=hparams["eta"], sigma=hparams["sigma"])
            elif method == "np":
                _ = three_factor_activation_step(model, Xb, yb, eta=hparams["eta"], sigma=hparams["sigma"])
            else:
                raise ValueError(f"Unknown method: {method}")

        with torch.no_grad():
            train_curve.append(F.mse_loss(model(X_train), y_train).item())
            test_curve.append(F.mse_loss(model(X_test), y_test).item())

    return (
        torch.tensor(train_curve),
        torch.tensor(test_curve),
        copy.deepcopy(model.state_dict()),
    )


def first_epoch_below(curve, threshold):
    below = torch.nonzero(curve <= threshold, as_tuple=False)
    if below.numel() == 0:
        return None
    return int(below[0].item())


def hparam_str(method, hp):
    if method == "bp":
        return f"lr={hp['lr']}"
    return f"eta={hp['eta']}_sigma={hp['sigma']}"


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    os.makedirs("plots_hparam", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device, flush=True)

    X_train, y_train, X_test, y_test = load_cali(device=device)

    epochs = 100
    batch_size = 256
    num_runs = 5
    seeds = list(range(num_runs))
    MSE_THRESHOLD = 0.6

    methods = ["bp", "wp", "wp3", "np"]
    labels = {
        "bp":  "Backprop",
        "wp":  "Weight perturbation",
        "wp3": "3-factor weight perturbation",
        "np":  "3-factor node perturbation",
    }
    colors = {"bp": "C0", "wp": "C1", "wp3": "C2", "np": "C3"}

    # -----------------------------
    # Hyperparameter grids (as requested)
    # -----------------------------
    grid_bp = [{"lr": lr} for lr in [0.007, 0.01, 0.02]]

    sigmas = [0.05, 0.1, 0.2]

    grid_wp_wp3 = [{"eta": eta, "sigma": s} for eta, s in itertools.product([0.03, 0.05, 0.07], sigmas)]
    grid_np = [{"eta": eta, "sigma": s} for eta, s in itertools.product([0.007, 0.01, 0.02], sigmas)]

    grids = {
        "bp": grid_bp,
        "wp": grid_wp_wp3,
        "wp3": grid_wp_wp3,
        "np": grid_np,
    }

    # For compute axis (reuse the same definition as your earlier script)
    N_batches = math.ceil(X_train.size(0) / batch_size)
    cost_batch = {"bp": 2.0, "wp": 1.5, "wp3": 1.4, "np": 1.8}
    cost_epoch = {m: cost_batch[m] * N_batches for m in methods}

    # Store results
    # results[method] = list of dicts, one per hyperparam setting
    results = {m: [] for m in methods}

    # -----------------------------
    # Sweep
    # -----------------------------
    for m in methods:
        print(f"\n=== Method: {m} ({labels[m]}) ===", flush=True)
        for hp in grids[m]:
            hp_name = hparam_str(m, hp)
            print(f"  -> {hp_name}", flush=True)

            train_curves = []
            test_curves = []

            best_state = None
            best_final_test = float("inf")

            for seed in seeds:
                tr, te, state = run_one_setting(
                    seed=seed,
                    method=m,
                    hparams=hp,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    device=device,
                )
                train_curves.append(tr)
                test_curves.append(te)

                final_te = float(te[-1].item())
                if final_te < best_final_test:
                    best_final_test = final_te
                    best_state = state

            train_stack = torch.stack(train_curves, dim=0)  # (runs, epochs)
            test_stack = torch.stack(test_curves, dim=0)

            mean_train = train_stack.mean(dim=0)
            mean_test = test_stack.mean(dim=0)

            # Metrics you asked for
            # (A) fastest to reach 0.6: based on MEAN test curve
            t_hit = first_epoch_below(mean_test, MSE_THRESHOLD)

            # (B) overall lowest score: take minimum of MEAN test curve (and also final)
            best_mean_test_min = float(mean_test.min().item())
            best_mean_test_final = float(mean_test[-1].item())

            results[m].append({
                "hparams": hp,
                "hparam_str": hp_name,
                "mean_train": mean_train.cpu(),
                "mean_test": mean_test.cpu(),
                "std_train": train_stack.std(dim=0).cpu(),
                "std_test": test_stack.std(dim=0).cpu(),
                "t_hit": t_hit,
                "mean_test_min": best_mean_test_min,
                "mean_test_final": best_mean_test_final,
                "best_state": best_state,  # best single-run final-test state (optional)
            })

    # -----------------------------
    # Print summary tables
    # -----------------------------
    print("\n\n==================== SUMMARY ====================", flush=True)
    for m in methods:
        print(f"\nMethod: {m} ({labels[m]})", flush=True)

        # Fastest to threshold (ignore None)
        candidates_hit = [r for r in results[m] if r["t_hit"] is not None]
        if len(candidates_hit) > 0:
            best_fast = min(candidates_hit, key=lambda r: r["t_hit"])
            print(f"  Fastest to reach test MSE <= {MSE_THRESHOLD}: "
                  f"{best_fast['hparam_str']} at epoch {best_fast['t_hit']}", flush=True)
        else:
            print(f"  Never reached test MSE <= {MSE_THRESHOLD} in {epochs} epochs", flush=True)

        # Lowest mean test min
        best_low = min(results[m], key=lambda r: r["mean_test_min"])
        print(f"  Lowest mean test MSE (min over epochs): {best_low['hparam_str']} "
              f"min={best_low['mean_test_min']:.4f} final={best_low['mean_test_final']:.4f}", flush=True)

    # -----------------------------
    # Plot: mean test loss vs epoch (all hparams) per method
    # -----------------------------
    epochs_arr = np.arange(epochs)
    updates_arr = (epochs_arr + 1) * N_batches
    for m in methods:
        plt.figure(figsize=(10, 6))
        for r in results[m]:
            plt.plot(updates_arr, r["mean_test"].numpy(), alpha=0.35, linewidth=1.5, label=r["hparam_str"])
        plt.xlabel("Update")
        plt.ylabel("Test MSE")
        plt.title(f"{labels[m]} – mean test loss vs update (all hyperparams)")
        plt.grid(alpha=0.3)
        # too many labels -> place outside
        plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"plots_hparam/cali_{m}_mean_test_vs_epoch_all_hparams.png", dpi=200)
        plt.close()

    # -----------------------------
    # Plot: mean test loss vs compute (all hparams) per method
    # -----------------------------
    for m in methods:
        plt.figure(figsize=(10, 6))
        comp_x = (epochs_arr + 1) * cost_epoch[m]
        for r in results[m]:
            plt.plot(comp_x, r["mean_test"].numpy(), alpha=0.35, linewidth=1.5, label=r["hparam_str"])
        plt.xlabel("Compute units (forward-pass equivalents)")
        plt.ylabel("Test MSE")
        plt.title(f"{labels[m]} – mean test loss vs compute (all hyperparams)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"plots_hparam/cali_{m}_mean_test_vs_compute_all_hparams.png", dpi=200)
        plt.close()

    # -----------------------------
    # Plot: “best fast” + “best low” shown clearly per method
    # -----------------------------
    for m in methods:
        candidates_hit = [r for r in results[m] if r["t_hit"] is not None]
        best_fast = min(candidates_hit, key=lambda r: r["t_hit"]) if len(candidates_hit) > 0 else None
        best_low = min(results[m], key=lambda r: r["mean_test_min"])

        plt.figure(figsize=(8, 5))
        if best_fast is not None:
            plt.plot(updates_arr, best_fast["mean_test"].numpy(), color="C0", linewidth=2.5,
                     label=f"fastest: {best_fast['hparam_str']} (t={best_fast['t_hit']})")
        plt.plot(updates_arr, best_low["mean_test"].numpy(), color="C1", linewidth=2.5,
                 label=f"lowest: {best_low['hparam_str']} (min={best_low['mean_test_min']:.3f})")
        plt.axhline(MSE_THRESHOLD, color="k", linestyle="--", linewidth=1, alpha=0.7, label="threshold")
        plt.xlabel("Update")
        plt.ylabel("Test MSE")
        plt.title(f"{labels[m]} – best hyperparams")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots_hparam/cali_{m}_best_hparams_overlay.png", dpi=200)
        plt.close()

    print("\nDone. Hyperparameter tuning plots in plots_hparam/", flush=True)