import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

device = torch.device("cpu")

class SimpleMLP(nn.Module):
    def __init__(self, dims=(1, 64, 32, 1), activation=torch.tanh):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=True))
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        h = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < last:
                h = self.activation(h)
        return h

def _get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def _set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

def generate_data(n=400, noise=0.1, seed=0):
    x = torch.linspace(-2*math.pi, 2*math.pi, n).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x.to(device), y.to(device)

def train_perturbation(model, X, Y, epochs=300, batch_size=128, sigma=0.1, eta=0.1, print_every=50):
    """
    Each perturbation is evaluated on a single (different) sample.
    One epoch = covering the whole dataset (N) in chunks of `batch_size` perturbations.
    """
    model.to(device)
    param_vec = _get_flat_params(model)
    num_params = param_vec.numel()
    x_size = X.size(0)

    for epoch in range(epochs):
        perm = torch.randperm(x_size, device=X.device)
        for start in range(0, x_size, batch_size):
            end = min(start + batch_size, x_size)
            idx = perm[start:end]
            current_batch_size = end - start

            eps = torch.randn(current_batch_size, num_params, device=device)
            losses = torch.empty(current_batch_size, device=device)

            for k in range(current_batch_size):
                x_k = X[idx[k]]   
                y_k = Y[idx[k]]

                p_k = param_vec + sigma * eps[k]
                _set_flat_params(model, p_k)
                with torch.no_grad():
                    y_pred = model(x_k)
                    losses[k] = F.mse_loss(y_pred, y_k, reduction='mean')

            rewards = -losses
            norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-12)

            grad_est = (norm_rewards.unsqueeze(1) * eps).mean(dim=0) / (sigma + 1e-12)

            param_vec = param_vec + eta * grad_est
            _set_flat_params(model, param_vec)

        if (epoch % print_every) == 0 or epoch == epochs - 1:
            with torch.no_grad():
                loss_unpert = F.mse_loss(model(X), Y, reduction='mean').item()
            avg_step_per_dim = grad_est.norm() / math.sqrt(grad_est.numel())
            print(f"Epoch {epoch:4d}  loss={loss_unpert:.6f}  grad_norm={grad_est.norm().item():.6f}  avg_step_per_dim={avg_step_per_dim:.6f}")

    return model

if __name__ == "__main__":
    X, Y = generate_data(n=400, noise=0.1, seed=1)
    model = SimpleMLP((1,64,32,1), activation=torch.tanh)
    trained = train_perturbation(model, X, Y, epochs=400, batch_size=128, sigma=0.1, eta=0.005)
    with torch.no_grad():
        xs = torch.linspace(-2*math.pi, 2*math.pi, 400).unsqueeze(1)
        ys = trained(xs)
        y_true = torch.sin(xs)

    # compute test MSE on the grid
    test_mse = F.mse_loss(ys, y_true, reduction='mean').item()
    print(f"Test MSE on grid: {test_mse:.6f}")

    # Plot predictions vs true function and training data
    plt.figure(figsize=(8, 5))
    plt.plot(xs.cpu().numpy(), y_true.cpu().numpy(), label='True sin(x)', color='C0')
    plt.plot(xs.cpu().numpy(), ys.cpu().numpy(), label='Model prediction', color='C1')
    plt.scatter(X.cpu().numpy(), Y.cpu().numpy(), s=10, alpha=0.3, label='Train samples', color='C2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Prediction vs True (MSE={test_mse:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Done")