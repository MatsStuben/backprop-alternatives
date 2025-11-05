import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt



class MLP_PERT_METHODS(nn.Module):
    def __init__(self, dimensions, activation=torch.sigmoid, output_activation=None): 
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i+1], bias=True)
                                     for i in range(len(dimensions)-1)])
        self.activation = activation
        self.output_activation = output_activation  
        for p in self.parameters(): 
            p.requires_grad_(False)
    
    def forward(self, x):
        """
        Forward pass during inference: use mean weights without perturbations.
        """
        final_layer = len(self.layers) - 1
        h = x
        for i, layer in enumerate(self.layers):
            
            w = layer.weight
            b = layer.bias
            u = h @ w.T + b
            if i == final_layer:
                h = self.output_activation(u) if self.output_activation else u
            else:
                h = self.activation(u)
        return h

    def forward_training(self, x, sigma):
        """
        Forward pass during training: add noise to weights and biases.
        """
        batch_size = x.shape[0]
        xs = [x]   
        ys = []
        noises = []
        h = x
        for i, layer in enumerate(self.layers):
            w = layer.weight        
            b = layer.bias
            in_neurons = layer.in_features
            scaling_factor = math.sqrt(1.0 / in_neurons)
            eps_w = torch.randn(batch_size, *w.shape) * sigma #* scaling_factor
            eps_b = torch.randn(batch_size, *b.shape) * sigma #* scaling_factor * 0.5

            w_used = w.unsqueeze(0) + eps_w 
            b_used = b.unsqueeze(0) + eps_b
            u = torch.bmm(h.unsqueeze(1), w_used.transpose(1, 2)).squeeze(1) + b_used
            final_layer = len(self.layers) - 1
            if i == final_layer:
                h = self.output_activation(u) if self.output_activation else u
                ys.append(h)
            else:
                h = self.activation(u)
                ys.append(h)
                xs.append(h)

            noises.append((eps_w, eps_b))
        return ys, xs, noises

def three_factor_step(model, X, target, eta=0.005, sigma=0.1):
    """
    One training step using your three-factor rule on a mini-batch.
    """
    model.train()

    ys, xs, noises = model.forward_training(X, sigma)

    y_out = ys[-1]
    loss_vec = F.mse_loss(y_out, target, reduction='none')
    if loss_vec.dim() > 1:
        loss_vec = loss_vec.mean(dim=1)
    loss_vec = loss_vec.view(-1)

    with torch.no_grad():
        reward = -loss_vec.squeeze()   
        reward_bar = reward.mean()
        norm_reward = (reward - reward_bar) / (reward.std() + 1e-8)
        
        h_in = xs   
        h_out = ys
        
        for layer_idx, (layer, (eps_w, eps_b), x_in, x_out) in enumerate(zip(model.layers, noises, h_in, h_out)):

            num_layers = len(model.layers)
            if layer_idx == 0 or layer_idx == num_layers - 1:
                eligibility_w = torch.ones_like(torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1)).transpose(1, 2))
            else:
                eligibility_w = torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1)).transpose(1, 2)
            eligibility_w = torch.ones_like(torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1)).transpose(1, 2))
            dW = (eps_w * eligibility_w * norm_reward.view(-1, 1, 1)).mean(dim=0) / (sigma + 1e-12) * eta
            
            new_weights = layer.weight + dW
            layer.weight.copy_(new_weights)

            eligibility_b = torch.ones_like(x_out)
            db = (eps_b * eligibility_b * norm_reward.unsqueeze(1)).mean(dim=0) / (sigma + 1e-12) * eta
            
            new_bias = layer.bias + db
            layer.bias.copy_(new_bias)

    return loss_vec.mean().item()

def two_factor_step(model, X, target, eta=0.005, sigma=0.1):
    """
    One training step using your two-factor rule on a mini-batch.
    """
    model.train()

    ys, xs, noises = model.forward_training(X, sigma)

    y_out = ys[-1]
    loss_vec = F.mse_loss(y_out, target, reduction='none')
    if loss_vec.dim() > 1:
        loss_vec = loss_vec.mean(dim=1)
    loss_vec = loss_vec.view(-1)

    with torch.no_grad():
        reward = -loss_vec.squeeze()   
        reward_bar = reward.mean()
        norm_reward = (reward - reward_bar) / (reward.std() + 1e-8)
        
        h_in = xs   
        h_out = ys
        
        for layer_idx, (layer, (eps_w, eps_b), x_in, x_out) in enumerate(zip(model.layers, noises, h_in, h_out)):

            dW = (eps_w  * norm_reward.view(-1, 1, 1)).mean(dim=0) / (sigma + 1e-12) * eta
            new_weights = layer.weight + dW
            layer.weight.copy_(new_weights)

            db = (eps_b  * norm_reward.unsqueeze(1)).mean(dim=0) / (sigma + 1e-12) * eta
            new_bias = layer.bias + db
            layer.bias.copy_(new_bias)

    return loss_vec.mean().item()

def backprop_step(model, X, target, loss_fn=F.mse_loss, eta=0.01, optimizer=torch.optim.SGD):
    """
    One training step using backpropagation on a mini-batch.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, target, reduction='mean')
    loss.backward()
    optimizer.step()

    return loss.item()

def normalize_layer_weights(weights, n_in, c=1.0):
    """
    Normalize weights to have target mean=0 and std=sqrt(c/n_in).
    """
    target_std = math.sqrt(c / n_in)
    target_mean = 0.0
    
    W_mean = weights.mean()
    W_std = weights.std().clamp_min(1e-8)
    
    W_normalized = (weights - W_mean) * (target_std / W_std) + target_mean
    
    return W_normalized

def normalize_layer_bias(bias, n_in, c=1.0):
    """
    Normalize biases to have mean=0 and std=sqrt(c/(4*n_in)).
    """
    if bias.numel() == 1:
        return bias  # Single bias, no normalization needed
    target_std = math.sqrt(c / (4 * n_in))
    target_mean = 0.0

    b_mean = bias.mean()
    b_std = bias.std().clamp_min(1e-8)

    b_normalized = (bias - b_mean) * (target_std / b_std) + target_mean
    return b_normalized

def generate_data(n=400, noise=0.1, seed=0):
    x = torch.linspace(-2*math.pi, 2*math.pi, n).unsqueeze(1)
    y = torch.sin(x) + noise * torch.randn_like(x)
    return x, y

if __name__ == "__main__":
    X, Y = generate_data(n=128*4, noise=0.1, seed=1)
    dimensions = (1, 64, 32, 1)
    model_three_factor = MLP_PERT_METHODS(dimensions, activation=torch.sigmoid)
    model_two_factor = MLP_PERT_METHODS(dimensions, activation=torch.sigmoid)
    model_backprop = MLP_PERT_METHODS(dimensions, activation=torch.sigmoid)
    optimizer_bp = torch.optim.SGD(model_backprop.parameters(), lr=0.1)
    epochs = 400
    batch_size = 128
    # Train both using three factor and two factor and compare results
    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        for start in range(0, X.size(0), batch_size):
            end = min(start + batch_size, X.size(0))
            idx = perm[start:end]
            X_batch = X[idx]
            Y_batch = Y[idx]
            loss_three_factor = three_factor_step(model_three_factor, X_batch, Y_batch, 
                                     eta=0.5, sigma=0.1)
            loss_two_factor = two_factor_step(model_two_factor, X_batch, Y_batch,
                                   eta=0.5, sigma=0.1)
            loss_backprop = backprop_step(model_backprop, X_batch, Y_batch, optimizer=optimizer_bp)

        if epoch % 50 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                y_pred_three_factor = model_three_factor(X)
                y_pred_two_factor = model_two_factor(X)
                y_pred_backprop = model_backprop(X)
                train_loss_three_factor = F.mse_loss(y_pred_three_factor, Y, reduction='mean').item()
                train_loss_two_factor = F.mse_loss(y_pred_two_factor, Y, reduction='mean').item()
                train_loss_backprop = F.mse_loss(y_pred_backprop, Y, reduction='mean').item()
            print(f"Epoch {epoch:4d}, 3F: {train_loss_three_factor:.6f}, 2F: {train_loss_two_factor:.6f}, BP: {train_loss_backprop:.6f}")
    with torch.no_grad():
        xs = torch.linspace(-2*math.pi, 2*math.pi, 400).unsqueeze(1)
        ys_three_factor = model_three_factor(xs)
        ys_two_factor = model_two_factor(xs)
        ys_backprop = model_backprop(xs)
        y_true = torch.sin(xs)

    test_mse_three_factor = F.mse_loss(ys_three_factor, y_true, reduction='mean').item()
    test_mse_two_factor = F.mse_loss(ys_two_factor, y_true, reduction='mean').item()
    test_mse_backprop = F.mse_loss(ys_backprop, y_true, reduction='mean').item()
    print(f"Test MSE on grid: {test_mse_three_factor:.6f}, {test_mse_two_factor:.6f}, {test_mse_backprop:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(xs.cpu().numpy(), y_true.cpu().numpy(), label='True sin(x)', color='C0')
    plt.plot(xs.cpu().numpy(), ys_three_factor.cpu().numpy(), label='Three-Factor Model prediction', color='C1')
    plt.plot(xs.cpu().numpy(), ys_two_factor.cpu().numpy(), label='Two-Factor Model prediction', color='C2')
    plt.plot(xs.cpu().numpy(), ys_backprop.cpu().numpy(), label='Backprop Model prediction', color='C3')
    plt.scatter(X.cpu().numpy(), Y.cpu().numpy(), s=10, alpha=0.3, label='Train samples', color='C4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Prediction vs True (MSE={test_mse_three_factor:.4f}, {test_mse_two_factor:.4f}, {test_mse_backprop:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Done")


     