import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from learning_rules_MLP import MLP, three_factor_weight_step, weight_perturb_step, backprop_step, weight_perturb_step_momentum, three_factor_activation_step, three_factor_activation_step_noisy

def generate_data(n=400, noise=0.1, seed=0):
    x = torch.linspace(-2*math.pi, 2*math.pi, n).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x, y



if __name__ == "__main__":
    X, Y = generate_data(n=128*10, noise=0.1, seed=1)
    dimensions = (1, 8, 4, 1)
    model_three_factor = MLP(dimensions, activation=torch.sigmoid)
    model_two_factor = MLP(dimensions, activation=torch.sigmoid)
    model_backprop = MLP(dimensions, activation=torch.sigmoid, require_grad=True)
    model_two_factor_momentum = MLP(dimensions, activation=torch.sigmoid)
    model_node_perturb = MLP(dimensions, activation=torch.sigmoid)
    model_node_perturb_noisy = MLP(dimensions, activation=torch.sigmoid)
    momentum_w = [torch.zeros_like(layer.weight) for layer in model_two_factor_momentum.layers]
    momentum_b = [torch.zeros_like(layer.bias) for layer in model_two_factor_momentum.layers]

    optimizer_bp = torch.optim.SGD(model_backprop.parameters(), lr=0.1)
    epochs = 400
    batch_size = 128
    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        for start in range(0, X.size(0), batch_size):
            end = min(start + batch_size, X.size(0))
            idx = perm[start:end]
            X_batch = X[idx]
            Y_batch = Y[idx]

            loss_backprop = backprop_step(model_backprop, X_batch, Y_batch, optimizer=optimizer_bp)
            loss_two_factor = weight_perturb_step(model_two_factor, X_batch, Y_batch,
                                   eta=0.7, sigma=0.2)
            loss_three_factor = three_factor_weight_step(model_three_factor, X_batch, Y_batch, 
                            eta=0.9, sigma=0.1)
            loss_momentum, momentum_w, momentum_b = weight_perturb_step_momentum(
                model_two_factor_momentum, X_batch, Y_batch,
                momentum_w, momentum_b, eta=0.03, sigma=0.1
            )

            loss_node_perturb = three_factor_activation_step(model_node_perturb, X_batch, Y_batch,
                                   eta=0.2, sigma=0.1)
            loss_node_perturb_noisy = three_factor_activation_step_noisy(model_node_perturb_noisy, X_batch, Y_batch,
                                   eta=0.7, sigma=0.2)

        if epoch % 20 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                y_pred_three_factor = model_three_factor(X)
                y_pred_two_factor = model_two_factor(X)
                y_pred_backprop = model_backprop(X)
                y_pred_momentum = model_two_factor_momentum(X)
                y_pred_node_perturb = model_node_perturb(X)
                y_pred_node_perturb_noisy = model_node_perturb_noisy(X)
                train_loss_three_factor = F.mse_loss(y_pred_three_factor, Y, reduction='mean').item()
                train_loss_two_factor = F.mse_loss(y_pred_two_factor, Y, reduction='mean').item()
                train_loss_backprop = F.mse_loss(y_pred_backprop, Y, reduction='mean').item()
                train_loss_momentum = F.mse_loss(y_pred_momentum, Y, reduction='mean').item()
                train_loss_node_perturb = F.mse_loss(y_pred_node_perturb, Y, reduction='mean').item()
                train_loss_node_perturb_noisy = F.mse_loss(y_pred_node_perturb_noisy, Y, reduction='mean').item()
            print(f"Epoch {epoch:4d}, 3F: {train_loss_three_factor:.6f}, 2F: {train_loss_two_factor:.6f}, BP: {train_loss_backprop:.6f}, 2F-M: {train_loss_momentum:.6f}, NP: {train_loss_node_perturb:.6f}, NP-N: {train_loss_node_perturb_noisy:.6f}")
    with torch.no_grad():
        xs = torch.linspace(-2*math.pi, 2*math.pi, 400).unsqueeze(1)
        ys_three_factor = model_three_factor(xs)
        ys_two_factor = model_two_factor(xs)
        ys_backprop = model_backprop(xs)
        ys_momentum = model_two_factor_momentum(xs)
        ys_node_perturb = model_node_perturb(xs)
        ys_node_perturb_noisy = model_node_perturb_noisy(xs)
        y_true = torch.sin(xs)

    test_mse_three_factor = F.mse_loss(ys_three_factor, y_true, reduction='mean').item()
    test_mse_two_factor = F.mse_loss(ys_two_factor, y_true, reduction='mean').item()
    test_mse_backprop = F.mse_loss(ys_backprop, y_true, reduction='mean').item()
    test_mse_momentum = F.mse_loss(ys_momentum, y_true, reduction='mean').item()
    test_mse_node_perturb = F.mse_loss(ys_node_perturb, y_true, reduction='mean').item()
    test_mse_node_perturb_noisy = F.mse_loss(ys_node_perturb_noisy, y_true, reduction='mean').item()
    print(f"Test MSE on grid: {test_mse_three_factor:.6f}, {test_mse_two_factor:.6f}, {test_mse_backprop:.6f}, {test_mse_momentum:.6f}, {test_mse_node_perturb:.6f}, {test_mse_node_perturb_noisy:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(xs.cpu().numpy(), y_true.cpu().numpy(), label='True sin(x)', color='C0')
    plt.plot(xs.cpu().numpy(), ys_three_factor.cpu().numpy(), label='Three-Factor Model prediction', color='C1')
    plt.plot(xs.cpu().numpy(), ys_two_factor.cpu().numpy(), label='Two-Factor Model prediction', color='C2')
    plt.plot(xs.cpu().numpy(), ys_backprop.cpu().numpy(), label='Backprop Model prediction', color='C3')
    plt.plot(xs.cpu().numpy(), ys_momentum.cpu().numpy(), label='Two-Factor Momentum Model prediction', color='C5')
    plt.plot(xs.cpu().numpy(), ys_node_perturb.cpu().numpy(), label='Node-Perturbation Model prediction', color='C6')
    plt.plot(xs.cpu().numpy(), ys_node_perturb_noisy.cpu().numpy(), label='Node-Perturbation Noisy prediction', color='C7')
    plt.scatter(X.cpu().numpy(), Y.cpu().numpy(), s=10, alpha=0.3, label='Train samples', color='C4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Prediction vs True (MSE={test_mse_three_factor:.4f}, {test_mse_two_factor:.4f}, {test_mse_backprop:.4f}, {test_mse_momentum:.4f}, {test_mse_node_perturb:.4f}, {test_mse_node_perturb_noisy:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Done")