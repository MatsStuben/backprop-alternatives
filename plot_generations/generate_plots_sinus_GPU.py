from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import copy
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from learning_rules_MLP import (
    MLP,
    three_factor_weight_step,
    weight_perturb_step,
    backprop_step,
    three_factor_activation_step,
)

# -----------------------------
# Data generation
# -----------------------------

def generate_data(n=400, noise=0.1, device="cpu"):
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n, device=device).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x, y


def generate_test_grid(n=400, device="cpu"):
    xs = torch.linspace(-2 * math.pi, 2 * math.pi, n, device=device).unsqueeze(1)
    ys = torch.sin(xs)
    return xs, ys


# -----------------------------
# One trial (one seed)
# -----------------------------

def run_single_trial(seed, X, Y, X_test, Y_test, epochs=400, batch_size=128, device="cpu"):
    """
    Run training for all four methods on the sinus task for a single random seed.

    Returns:
      - train_losses: dict[method] -> tensor[epochs]
      - test_losses: dict[method] -> tensor[epochs]
      - models: dict[method] -> trained MLP
    """
    torch.manual_seed(seed)

    dimensions = (1, 8, 4, 1)

    # MLP is CPU-only right now (learning_rules_MLP uses CPU tensors internally),
    # so we keep models on the same device as X/Y, which is "device".
    model_bp = MLP(dimensions, activation=torch.sigmoid, require_grad=True).to(device)
    model_wp = MLP(dimensions, activation=torch.sigmoid).to(device)
    model_wp3 = MLP(dimensions, activation=torch.sigmoid).to(device)
    model_np = MLP(dimensions, activation=torch.sigmoid).to(device)

    optimizer_bp = torch.optim.SGD(model_bp.parameters(), lr=0.5)

    N = X.size(0)

    train_losses = {m: [] for m in ["bp", "wp", "wp3", "np"]}
    test_losses = {m: [] for m in ["bp", "wp", "wp3", "np"]}

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            X_batch = X[idx]
            Y_batch = Y[idx]

            # Backprop
            _ = backprop_step(model_bp, X_batch, Y_batch, optimizer=optimizer_bp)

            # Weight perturbation (2-factor)
            _ = weight_perturb_step(
                model_wp, X_batch, Y_batch, eta=0.7, sigma=0.2
            )

            # Three-factor weight perturbation
            _ = three_factor_weight_step(
                model_wp3, X_batch, Y_batch, eta=0.9, sigma=0.1
            )

            # Three-factor node perturbation
            _ = three_factor_activation_step(
                model_np, X_batch, Y_batch, eta=0.2, sigma=0.1
            )

        # Evaluate train and test MSE at end of epoch
        with torch.no_grad():
            y_bp_train = model_bp(X)
            y_wp_train = model_wp(X)
            y_wp3_train = model_wp3(X)
            y_np_train = model_np(X)

            train_losses["bp"].append(F.mse_loss(y_bp_train, Y).item())
            train_losses["wp"].append(F.mse_loss(y_wp_train, Y).item())
            train_losses["wp3"].append(F.mse_loss(y_wp3_train, Y).item())
            train_losses["np"].append(F.mse_loss(y_np_train, Y).item())

            y_bp_test = model_bp(X_test)
            y_wp_test = model_wp(X_test)
            y_wp3_test = model_wp3(X_test)
            y_np_test = model_np(X_test)

            test_losses["bp"].append(F.mse_loss(y_bp_test, Y_test).item())
            test_losses["wp"].append(F.mse_loss(y_wp_test, Y_test).item())
            test_losses["wp3"].append(F.mse_loss(y_wp3_test, Y_test).item())
            test_losses["np"].append(F.mse_loss(y_np_test, Y_test).item())

        if epoch % 100 == 0:
            print(f"Seed {seed}: epoch {epoch}/{epochs}", flush=True)

    # Convert to tensors [epochs] (on CPU for plotting)
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


# -----------------------------
# Main multi-run experiment
# -----------------------------

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    Y_MAX = 1.2  # clip y-axis so spikes don't ruin readability

    # For now, keep this on CPU because learning_rules_MLP is CPU-based.
    # Once we port that file to GPU properly, you can switch to "cuda".
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device, flush=True)

    # Fixed dataset for all runs
    X, Y = generate_data(n=128 * 10, noise=0.1, device=device)
    X_test, Y_test = generate_test_grid(n=400, device=device)

    epochs = 400
    batch_size = 128
    num_runs = 10
    seeds = list(range(num_runs))
    MSE_THRESHOLD = 0.1  # threshold defining "good enough"

    methods = ["bp", "wp", "wp3", "np"]
    method_labels = {
        "bp": "Backprop",
        "wp": "Weight perturbation",
        "wp3": "3-factor weight perturbation",
        "np": "3-factor node perturbation",
    }
    method_colors = {
        "bp": "C0",
        "wp": "C1",
        "wp3": "C2",
        "np": "C3",
    }

    N = X.size(0)
    n_batches = math.ceil(N / batch_size)

    # Approximate cost per batch in forward-pass equivalents
    cost_per_batch = {"bp": 2.0, "wp": 1.5, "wp3": 1.75, "np": 1.6}

    cost_per_epoch = {m: cost_per_batch[m] * n_batches for m in methods}

    # Store all train loss curves for variance
    all_train_losses = {m: [] for m in methods}
    all_test_losses = {m: [] for m in methods}

    # Track best run per method according to earliest crossing of threshold
    best_epoch_to_threshold = {m: None for m in methods}
    best_train_curve = {m: None for m in methods}
    best_test_curve = {m: None for m in methods}
    best_state_dict = {m: None for m in methods}

    # Fallback: lowest final test loss if no run hits threshold
    best_final_test_mse = {m: float("inf") for m in methods}

    for seed in seeds:
        print(f"\nRunning run {seed}...", flush=True)
        train_losses, test_losses, models = run_single_trial(
            seed, X, Y, X_test, Y_test,
            epochs=epochs, batch_size=batch_size, device=device
        )

        # Store for variance plots
        for m in methods:
            all_train_losses[m].append(train_losses[m])
            all_test_losses[m].append(test_losses[m])

        # Select best run according to earliest epoch where test MSE <= threshold
        for m in methods:
            test_curve = test_losses[m]
            below = torch.nonzero(test_curve <= MSE_THRESHOLD, as_tuple=False)
            if below.numel() > 0:
                first_epoch = int(below[0].item())  # 0-based index
                if best_epoch_to_threshold[m] is None or first_epoch < best_epoch_to_threshold[m]:
                    best_epoch_to_threshold[m] = first_epoch
                    best_train_curve[m] = train_losses[m].clone()
                    best_test_curve[m] = test_losses[m].clone()
                    best_state_dict[m] = copy.deepcopy(models[m].state_dict())
            else:
                # Fallback: if no run ever hits threshold, track lowest final test MSE
                final_test_mse = float(test_curve[-1].item())
                if best_epoch_to_threshold[m] is None:  # only if no run crossed threshold yet
                    if final_test_mse < best_final_test_mse[m]:
                        best_final_test_mse[m] = final_test_mse
                        best_train_curve[m] = train_losses[m].clone()
                        best_test_curve[m] = test_losses[m].clone()
                        best_state_dict[m] = copy.deepcopy(models[m].state_dict())

    # Restore best models (still CPU)
    best_models = {}
    for m in methods:
        model = MLP((1, 8, 4, 1), activation=torch.sigmoid, require_grad=(m == "bp")).to(device)
        model.load_state_dict(best_state_dict[m])
        best_models[m] = model

    epochs_arr = np.arange(epochs)
    updates_arr = (epochs_arr + 1) * n_batches

    # -----------------------------
    # Plot 1: Best training loss curve per method (loss vs update)
    # -----------------------------
    plt.figure(figsize=(8, 5))
    for m in methods:
        plt.plot(
            updates_arr,
            best_train_curve[m].cpu().numpy(),
            label=f"{method_labels[m]} (best run)",
            color=method_colors[m],
        )
    plt.ylim(0.0, Y_MAX)
    plt.xlabel("Update")
    plt.ylabel("Training MSE")
    # plt.title(...)  # removed
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sinus_best_training_vs_epoch.pdf")
    plt.close()

    # -----------------------------
    # Plot 1b: Best training loss per method (loss vs compute)
    # -----------------------------
    plt.figure(figsize=(8, 5))
    for m in methods:
        compute_axis = (epochs_arr + 1) * cost_per_epoch[m]
        plt.plot(
            compute_axis,
            best_train_curve[m].cpu().numpy(),
            label=f"{method_labels[m]} (best run)",
            color=method_colors[m],
        )
    plt.ylim(0.0, Y_MAX)
    plt.xlabel("Compute units (approx. forward-pass equivalents)")
    plt.ylabel("Training MSE")
    # plt.title(...)  # removed
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sinus_best_training_vs_compute.pdf")
    plt.close()

    # -----------------------------
    # Plot 2: Variance shading of training loss per method (loss vs update)
    # -----------------------------

    def plot_with_shading(x, curves, label, color):
        arr = torch.stack(curves, dim=0).cpu().numpy()  # [num_runs, epochs]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        plt.plot(x, mean, label=label, color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    for m in methods:
        plt.figure(figsize=(8, 5))
        plot_with_shading(
            updates_arr,
            all_train_losses[m],
            label=method_labels[m],
            color=method_colors[m],
        )
        plt.ylim(0.0, Y_MAX)
        plt.xlabel("Update")
        plt.ylabel("Training MSE")
        # plt.title(...)  # removed
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/sinus_variance_training_{m}.pdf")
        plt.close()

    # -----------------------------
    # Plot 2b: Variance shading of *test* loss per method (loss vs update)
    # -----------------------------
    for m in methods:
        plt.figure(figsize=(8, 5))
        plot_with_shading(
            updates_arr,
            all_test_losses[m],
            label=method_labels[m],
            color=method_colors[m],
        )
        plt.ylim(0.0, Y_MAX)
        plt.xlabel("Update")
        plt.ylabel("Test MSE")
        # plt.title(...)  # removed
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/sinus_variance_test_{m}.pdf")
        plt.close()

    # -----------------------------
    # Plot 3: Prediction quality using best models
    # -----------------------------
    with torch.no_grad():
        y_true = Y_test
        preds = {m: best_models[m](X_test) for m in methods}
        test_mse_best = {m: F.mse_loss(preds[m], y_true).item() for m in methods}

    plt.figure(figsize=(8, 5))
    plt.plot(X_test.cpu().numpy(), y_true.cpu().numpy(), label="True sin(x)", color="k")

    for m in methods:
        plt.plot(
            X_test.cpu().numpy(),
            preds[m].cpu().numpy(),
            label=f"{method_labels[m]} (MSE={test_mse_best[m]:.4f})",
            color=method_colors[m],
        )

    plt.scatter(
        X.cpu().numpy(),
        Y.cpu().numpy(),
        s=10,
        alpha=0.3,
        label="Train samples",
        color="C4",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title(...)  # removed
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/prediction_quality_best_models.pdf")
    plt.close()

    # -----------------------------
    # Plot 4: Average training loss vs update
    # -----------------------------
    plt.figure(figsize=(8, 5))
    for m in methods:
        arr = torch.stack(all_train_losses[m], dim=0).cpu().numpy()
        mean = arr.mean(axis=0)
        plt.plot(
            updates_arr,
            mean,
            label=f"{method_labels[m]} (mean)",
            color=method_colors[m],
        )
    plt.ylim(0.0, Y_MAX)
    plt.xlabel("Update")
    plt.ylabel("Training MSE")
    # plt.title(...)  # removed
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sinus_avg_training_vs_epoch.pdf")
    plt.close()

    # -----------------------------
    # Plot 4b: Average training loss vs compute
    # -----------------------------
    plt.figure(figsize=(8, 5))
    for m in methods:
        arr = torch.stack(all_train_losses[m], dim=0).cpu().numpy()
        mean = arr.mean(axis=0)
        compute_axis = (epochs_arr + 1) * cost_per_epoch[m]
        plt.plot(
            compute_axis,
            mean,
            label=f"{method_labels[m]} (mean)",
            color=method_colors[m],
        )
    plt.ylim(0.0, Y_MAX)
    plt.xlabel("Compute units (approx. forward-pass equivalents)")
    plt.ylabel("Training MSE")
    # plt.title(...)  # removed
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/sinus_avg_training_vs_compute.pdf")
    plt.close()

    # -----------------------------
    # Scalar variance summaries (single number per method)
    # -----------------------------
    print("\nVariance summary (across runs):", flush=True)
    for m in methods:
        train_arr = torch.stack(all_train_losses[m], dim=0)  # (num_runs, epochs)
        test_arr = torch.stack(all_test_losses[m], dim=0)    # (num_runs, epochs)

        # variance across runs for each epoch, then average over epochs
        mean_var_train = train_arr.var(dim=0, unbiased=False).mean().item()
        mean_std_train = train_arr.std(dim=0, unbiased=False).mean().item()

        mean_var_test = test_arr.var(dim=0, unbiased=False).mean().item()
        mean_std_test = test_arr.std(dim=0, unbiased=False).mean().item()

        print(
            f"  {m:>3} ({method_labels[m]}): "
            f"train mean-var={mean_var_train:.6f}, train mean-std={mean_std_train:.6f} | "
            f"test mean-var={mean_var_test:.6f}, test mean-std={mean_std_test:.6f}",
            flush=True,
        )

    print("Done. Plots saved in plots/", flush=True)
