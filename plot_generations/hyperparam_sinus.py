import os
import copy
import math
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from learning_rules_MLP import (
    MLP,
    three_factor_weight_step,
    weight_perturb_step,
    backprop_step,
    three_factor_activation_step,
)

# ---------------------------------------------------
# Data (sinus)
# ---------------------------------------------------

def generate_sinus(n=128 * 10, noise=0.1, seed=1, device="cpu"):
    torch.manual_seed(seed)
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n, device=device).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x, y

def standardize_train_test(X_train, Y_train, X_test, Y_test, eps=1e-8):
    x_mean = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, keepdim=True).clamp_min(eps)
    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True).clamp_min(eps)

    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    Y_train = (Y_train - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std
    return X_train, Y_train, X_test, Y_test

def make_grid(n=400, device="cpu"):
    xs = torch.linspace(-2 * math.pi, 2 * math.pi, n, device=device).unsqueeze(1)
    ys = torch.sin(xs)
    return xs, ys


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
    epochs=400,
    batch_size=128,
    device="cpu",
):
    torch.manual_seed(seed)

    dims = (1, 8, 4, 1)

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
    os.makedirs("plots_hparam_sinus", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device, flush=True)

    # Train data and a deterministic test grid
    X_train, y_train = generate_sinus(n=128 * 10, noise=0.1, seed=1, device=device)
    X_test, y_test = make_grid(n=400, device=device)

    # Standardize using train statistics
    X_train, y_train, X_test, y_test = standardize_train_test(X_train, y_train, X_test, y_test)

    epochs = 400
    batch_size = 128
    num_runs = 5
    seeds = list(range(num_runs))
    MSE_THRESHOLD = 0.1  # on standardized target

    methods = ["bp", "wp", "wp3", "np"]
    labels = {
        "bp":  "Backprop",
        "wp":  "Weight perturbation",
        "wp3": "3-factor weight perturbation",
        "np":  "3-factor node perturbation",
    }

    # -----------------------------
    # Hyperparameter grids (same style as cali file)
    # -----------------------------
    grid_bp = [{"lr": lr} for lr in [0.05, 0.1, 0.2]]

    sigmas = [0.05, 0.1, 0.2, 0.3]
    grid_wp_wp3 = [{"eta": eta, "sigma": s} for eta, s in itertools.product([0.3, 0.5, 0.7, 0.9], sigmas)]
    grid_np = [{"eta": eta, "sigma": s} for eta, s in itertools.product([0.07, 0.1, 0.2], sigmas)]

    grids = {
        "bp": grid_bp,
        "wp": grid_wp_wp3,
        "wp3": grid_wp_wp3,
        "np": grid_np,
    }

    # Compute axis (same simple model as in your cali file)
    N_batches = math.ceil(X_train.size(0) / batch_size)
    cost_batch = {"bp": 2.0, "wp": 1.5, "wp3": 1.4, "np": 1.8}
    cost_epoch = {m: cost_batch[m] * N_batches for m in methods}

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

            t_hit = first_epoch_below(mean_test, MSE_THRESHOLD)

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
                "best_state": best_state,
            })

    # -----------------------------
    # Print summary tables
    # -----------------------------
    print("\n\n==================== SUMMARY (SINUS) ====================", flush=True)
    for m in methods:
        print(f"\nMethod: {m} ({labels[m]})", flush=True)

        candidates_hit = [r for r in results[m] if r["t_hit"] is not None]
        if len(candidates_hit) > 0:
            best_fast = min(candidates_hit, key=lambda r: r["t_hit"])
            print(f"  Fastest to reach test MSE <= {MSE_THRESHOLD}: "
                  f"{best_fast['hparam_str']} at epoch {best_fast['t_hit']}", flush=True)
        else:
            print(f"  Never reached test MSE <= {MSE_THRESHOLD} in {epochs} epochs", flush=True)

        best_low = min(results[m], key=lambda r: r["mean_test_min"])
        print(f"  Lowest mean test MSE (min over epochs): {best_low['hparam_str']} "
              f"min={best_low['mean_test_min']:.4f} final={best_low['mean_test_final']:.4f}", flush=True)

    # -----------------------------
    # Plots (same style)
    # -----------------------------
    x = np.arange(epochs)

    for m in methods:
        plt.figure(figsize=(10, 6))
        for r in results[m]:
            plt.plot(x, r["mean_test"].numpy(), alpha=0.35, linewidth=1.5, label=r["hparam_str"])
        plt.xlabel("Epoch")
        plt.ylabel("Test MSE")
        plt.title(f"{labels[m]} – mean test loss vs epoch (all hyperparams) [sinus]")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"plots_hparam_sinus/sinus_{m}_mean_test_vs_epoch_all_hparams.png", dpi=200)
        plt.close()

    for m in methods:
        plt.figure(figsize=(10, 6))
        comp_x = (x + 1) * cost_epoch[m]
        for r in results[m]:
            plt.plot(comp_x, r["mean_test"].numpy(), alpha=0.35, linewidth=1.5, label=r["hparam_str"])
        plt.xlabel("Compute units (forward-pass equivalents)")
        plt.ylabel("Test MSE")
        plt.title(f"{labels[m]} – mean test loss vs compute (all hyperparams) [sinus]")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1.0), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"plots_hparam_sinus/sinus_{m}_mean_test_vs_compute_all_hparams.png", dpi=200)
        plt.close()

    for m in methods:
        candidates_hit = [r for r in results[m] if r["t_hit"] is not None]
        best_fast = min(candidates_hit, key=lambda r: r["t_hit"]) if len(candidates_hit) > 0 else None
        best_low = min(results[m], key=lambda r: r["mean_test_min"])

        plt.figure(figsize=(8, 5))
        if best_fast is not None:
            plt.plot(x, best_fast["mean_test"].numpy(), color="C0", linewidth=2.5,
                     label=f"fastest: {best_fast['hparam_str']} (t={best_fast['t_hit']})")
        plt.plot(x, best_low["mean_test"].numpy(), color="C1", linewidth=2.5,
                 label=f"lowest: {best_low['hparam_str']} (min={best_low['mean_test_min']:.3f})")
        plt.axhline(MSE_THRESHOLD, color="k", linestyle="--", linewidth=1, alpha=0.7, label="threshold")
        plt.xlabel("Epoch")
        plt.ylabel("Test MSE")
        plt.title(f"{labels[m]} – best hyperparams [sinus]")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots_hparam_sinus/sinus_{m}_best_hparams_overlay.png", dpi=200)
        plt.close()

    print("\nDone. Hyperparameter tuning plots in plots_hparam_sinus/", flush=True)