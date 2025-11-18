import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from three_factor_MLP import MLP, three_factor_weight_step, weight_perturb_step, backprop_step, weight_perturb_step_momentum, three_factor_activation_step

if __name__ == "__main__":
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X_housing = housing.data
    y_housing = housing.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_housing, y_housing, test_size=0.2, random_state=42
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
    model_three_factor = MLP(dimensions, activation=torch.sigmoid)
    model_two_factor = MLP(dimensions, activation=torch.sigmoid)
    model_backprop = MLP(dimensions, activation=torch.sigmoid, require_grad=True)
    model_two_factor_momentum = MLP(dimensions, activation=torch.sigmoid)
    model_node_perturb = MLP(dimensions, activation=torch.sigmoid)
    optimizer_bp = torch.optim.SGD(model_backprop.parameters(), lr=0.01)
    momentum_w = [torch.zeros_like(layer.weight) for layer in model_two_factor_momentum.layers]
    momentum_b = [torch.zeros_like(layer.bias) for layer in model_two_factor_momentum.layers]
    
    epochs = 100
    batch_size = 256
    
    train_losses_3f = []
    train_losses_2f = []
    train_losses_bp = []
    train_losses_2f_momentum = []
    train_losses_node_perturb = []
    test_losses_3f = []
    test_losses_2f = []
    test_losses_bp = []
    test_losses_2f_momentum = []
    test_losses_node_perturb = []
    print("\nTraining models...")
    for epoch in range(epochs):
        perm = torch.randperm(X_train_t.size(0))
        for start in range(0, X_train_t.size(0), batch_size):
            end = min(start + batch_size, X_train_t.size(0))
            idx = perm[start:end]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]
            
            loss_two_factor = weight_perturb_step(model_two_factor, X_batch, y_batch,
                                             eta=0.05, sigma=0.1)
            loss_backprop = backprop_step(model_backprop, X_batch, y_batch, optimizer=optimizer_bp)
            loss_three_factor = three_factor_weight_step(model_three_factor, X_batch, y_batch, 
                                                 eta=0.05, sigma=0.1)
            loss_momentum, momentum_w, momentum_b = weight_perturb_step_momentum(
                model_two_factor_momentum, X_batch, y_batch,
                momentum_w, momentum_b, eta=0.01, sigma=0.1
            )
            loss_node_perturb = three_factor_activation_step(model_node_perturb, X_batch, y_batch,
                                   eta=0.01, sigma=0.1)

        if epoch % 1 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                y_pred_3f_train = model_three_factor(X_train_t)
                y_pred_2f_train = model_two_factor(X_train_t)
                y_pred_bp_train = model_backprop(X_train_t)
                y_pred_momentum_train = model_two_factor_momentum(X_train_t)
                y_pred_node_perturb_train = model_node_perturb(X_train_t)
                
                train_loss_3f = F.mse_loss(y_pred_3f_train, y_train_t, reduction='mean').item()
                train_loss_2f = F.mse_loss(y_pred_2f_train, y_train_t, reduction='mean').item()
                train_loss_bp = F.mse_loss(y_pred_bp_train, y_train_t, reduction='mean').item()
                train_loss_momentum = F.mse_loss(y_pred_momentum_train, y_train_t, reduction='mean').item()
                train_loss_node_perturb = F.mse_loss(y_pred_node_perturb_train, y_train_t, reduction='mean').item()

                y_pred_3f_test = model_three_factor(X_test_t)
                y_pred_2f_test = model_two_factor(X_test_t)
                y_pred_bp_test = model_backprop(X_test_t)
                y_pred_momentum_test = model_two_factor_momentum(X_test_t)
                y_pred_node_perturb_test = model_node_perturb(X_test_t)
                
                test_loss_3f = F.mse_loss(y_pred_3f_test, y_test_t, reduction='mean').item()
                test_loss_2f = F.mse_loss(y_pred_2f_test, y_test_t, reduction='mean').item()
                test_loss_bp = F.mse_loss(y_pred_bp_test, y_test_t, reduction='mean').item()
                test_loss_momentum = F.mse_loss(y_pred_momentum_test, y_test_t, reduction='mean').item()
                test_loss_node_perturb = F.mse_loss(y_pred_node_perturb_test, y_test_t, reduction='mean').item()

                train_losses_3f.append(train_loss_3f)
                train_losses_2f.append(train_loss_2f)
                train_losses_bp.append(train_loss_bp)
                train_losses_2f_momentum.append(train_loss_momentum)
                train_losses_node_perturb.append(train_loss_node_perturb)
                test_losses_3f.append(test_loss_3f)
                test_losses_2f.append(test_loss_2f)
                test_losses_bp.append(test_loss_bp)
                test_losses_2f_momentum.append(test_loss_momentum)
                test_losses_node_perturb.append(test_loss_node_perturb)
                print(f"Epoch {epoch:3d} | Train - 3F: {train_loss_3f:.4f}, 2F: {train_loss_2f:.4f}, BP: {train_loss_bp:.4f}, 2F-M: {train_loss_momentum:.4f}, NP: {train_loss_node_perturb:.4f} | "
                      f"Test - 3F: {test_loss_3f:.4f}, 2F: {test_loss_2f:.4f}, BP: {test_loss_bp:.4f}, 2F-M: {test_loss_momentum:.4f}, NP: {test_loss_node_perturb:.4f}")

    with torch.no_grad():
        y_pred_3f = model_three_factor(X_test_t)
        y_pred_2f = model_two_factor(X_test_t)
        y_pred_bp = model_backprop(X_test_t)
        y_pred_momentum = model_two_factor_momentum(X_test_t)
        y_pred_node_perturb = model_node_perturb(X_test_t)

        final_test_mse_3f = F.mse_loss(y_pred_3f, y_test_t, reduction='mean').item()
        final_test_mse_2f = F.mse_loss(y_pred_2f, y_test_t, reduction='mean').item()
        final_test_mse_bp = F.mse_loss(y_pred_bp, y_test_t, reduction='mean').item()
        final_test_mse_momentum = F.mse_loss(y_pred_momentum, y_test_t, reduction='mean').item()
        final_test_mse_node_perturb = F.mse_loss(y_pred_node_perturb, y_test_t, reduction='mean').item()

    print(f"\nFinal Test MSE:")
    print(f"  Three-Factor: {final_test_mse_3f:.6f}")
    print(f"  Two-Factor:   {final_test_mse_2f:.6f}")
    print(f"  Backprop SGD: {final_test_mse_bp:.6f}")
    print(f"  Two-Factor Momentum: {final_test_mse_momentum:.6f}")
    print(f"  Node-Perturbation: {final_test_mse_node_perturb:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_plot = list(range(epochs))
    
    ax1.plot(epochs_plot, train_losses_3f, label='Three-Factor', marker='o')
    ax1.plot(epochs_plot, train_losses_2f, label='Two-Factor', marker='s')
    ax1.plot(epochs_plot, train_losses_bp, label='Backprop SGD', marker='^')
    ax1.plot(epochs_plot, train_losses_2f_momentum, label='Two-Factor Momentum', marker='v')
    ax1.plot(epochs_plot, train_losses_node_perturb, label='Node-Perturbation', marker='x')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train MSE')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_plot, test_losses_3f, label='Three-Factor', marker='o')
    ax2.plot(epochs_plot, test_losses_2f, label='Two-Factor', marker='s')
    ax2.plot(epochs_plot, test_losses_bp, label='Backprop SGD', marker='^')
    ax2.plot(epochs_plot, test_losses_2f_momentum, label='Two-Factor Momentum', marker='v')
    ax2.plot(epochs_plot, test_losses_node_perturb, label='Node-Perturbation', marker='x')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test MSE')
    ax2.set_title('Test Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = [
        (y_pred_3f, 'Three-Factor', final_test_mse_3f),
        (y_pred_2f, 'Two-Factor', final_test_mse_2f),
        (y_pred_bp, 'Backprop SGD', final_test_mse_bp),
        (y_pred_momentum, 'Two-Factor Momentum', final_test_mse_momentum)
        , (y_pred_node_perturb, 'Node-Perturbation', final_test_mse_node_perturb)
    ]
    
    for ax, (y_pred, name, mse) in zip(axes, models):
        ax.scatter(y_test_t.numpy(), y_pred.numpy(), alpha=0.3, s=10)
        ax.plot([y_test_t.min(), y_test_t.max()], 
                [y_test_t.min(), y_test_t.max()], 
                'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel('True Values (normalized)')
        ax.set_ylabel('Predictions (normalized)')
        ax.set_title(f'{name}\nTest MSE: {mse:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Done")