import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt



class MLP(nn.Module):
    def __init__(self, dimensions, activation=torch.sigmoid, output_activation=None, require_grad = False): 
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i+1], bias=True)
                                     for i in range(len(dimensions)-1)])
        self.activation = activation
        self.output_activation = output_activation  
        if not require_grad:
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

    def forward_perturb(self, x, sigma):
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
            eps_w = torch.randn(batch_size, *w.shape) * sigma 
            eps_b = torch.randn(batch_size, *b.shape) * sigma

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

def three_factor_weight_step(model, X, target, eta=0.5, sigma=0.1):

    model.train()
    ys, xs, noises = model.forward_perturb(X, sigma)
    y_out = ys[-1]
    loss_vec = F.mse_loss(y_out, target, reduction='none')

    if loss_vec.dim() > 1:
        loss_vec = loss_vec.mean(dim=1)
    loss_vec = loss_vec.view(-1)

    with torch.no_grad():
        reward = -loss_vec.squeeze()   
        reward_bar = reward.mean()
        norm_reward = (reward - reward_bar)
        
        h_in = xs   
        h_out = ys
        
        for layer_idx, (layer, (eps_w, eps_b), x_in, x_out) in enumerate(zip(model.layers, noises, h_in, h_out)):

            num_layers = len(model.layers)
            if layer_idx == 0 or layer_idx == num_layers - 1:
                eligibility_w = torch.ones_like(torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1)).transpose(1, 2))
            else:
                eligibility_w = torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1)).transpose(1, 2)

            dW = ((eps_w* eligibility_w * norm_reward.view(-1, 1, 1)).mean(dim=0) / (sigma + 1e-12)) * eta
            
            new_weights = layer.weight + dW
            layer.weight.copy_(new_weights)

            eligibility_b = torch.ones_like(x_out)
            db = ((eps_b * eligibility_b * norm_reward.unsqueeze(1)).mean(dim=0) / (sigma + 1e-12)) * eta
            
            new_bias = layer.bias + db
            layer.bias.copy_(new_bias)

    return loss_vec.mean().item()

def three_factor_activation_step(model, X, target, eta=0.5, sigma=0.1):
    """
    Node perturbation + three-factor rule:
      reward = -loss (noisy forward)
      baseline = mean(reward)
      gradient estimate: a_i * phi'(z_j) * ( (R - b) * ξ_j / σ )
    Final layer kept linear if model.output_activation is None.
    """
    model.train()
    batch_size = X.shape[0]
    final_layer = len(model.layers) - 1

    # ----- 1) Clean forward (store pre-activation z and activation a) -----
    xs = []          # inputs to each layer
    acts = []        # activations of each layer
    preacts = []     # pre-activations z_j (needed for derivatives if custom activation)
    h = X
    for i, layer in enumerate(model.layers):
        xs.append(h)
        z = h @ layer.weight.T + layer.bias        # (B, out)
        preacts.append(z)
        if i == final_layer:
            h = model.output_activation(z) if model.output_activation else z
        else:
            h = model.activation(z)
        acts.append(h)
    y_clean = acts[-1]

    # Clean loss (for logging)
    loss_vec_clean = F.mse_loss(y_clean, target, reduction='none')
    if loss_vec_clean.dim() > 1:
        loss_vec_clean = loss_vec_clean.mean(dim=1)
    L_clean = loss_vec_clean.view(-1)  # (B,)

    # ----- 2) Noisy forward: add noise to activations (node perturbation) -----
    noise_list = []
    h_tilde = X
    for i, layer in enumerate(model.layers):
        z = h_tilde @ layer.weight.T + layer.bias
        if i == final_layer:
            a = model.output_activation(z) if model.output_activation else z
        else:
            a = model.activation(z)

        eps = torch.randn_like(a)          # ξ_j
        noise_list.append(eps)
        a_tilde = a + sigma * eps
        h_tilde = a_tilde
    y_tilde = h_tilde

    loss_vec_noisy = F.mse_loss(y_tilde, target, reduction='none')
    if loss_vec_noisy.dim() > 1:
        loss_vec_noisy = loss_vec_noisy.mean(dim=1)
    L_noisy = loss_vec_noisy.view(-1)

    # ----- 3) Reward signal -----
    with torch.no_grad():
        reward = -L_noisy                      # higher reward = lower loss
        baseline = reward.mean()
        norm_reward = reward - baseline        # (B,)

        # ----- 4) Weight updates -----
        for i, layer in enumerate(model.layers):
            x_in  = xs[i]                      # (B, in_dim)
            a     = acts[i]                    # (B, out_dim) clean activation
            eps   = noise_list[i]              # (B, out_dim)

            # dR/da_j ≈ (R - b) * ξ_j / σ
            dR_da_est = norm_reward.view(-1, 1) * eps / (sigma + 1e-8)

            # Activation derivative φ'(z_j):
            if i == final_layer:
                # Linear output (unless custom output_activation provided)
                if model.output_activation is None:
                    phi_prime = torch.ones_like(a)
                else:
                    # If you later use a custom output activation, add its derivative here
                    if model.output_activation is torch.sigmoid:
                        phi_prime = a * (1 - a)
                    else:
                        phi_prime = torch.ones_like(a)  # fallback
            else:
                # Hidden layer derivative
                if model.activation is torch.sigmoid:
                    phi_prime = a * (1 - a)
                elif model.activation is torch.tanh:
                    phi_prime = 1 - a.pow(2)
                else:
                    # ReLU or other: approximate derivative
                    if model.activation is torch.relu:
                        phi_prime = (preacts[i] > 0).float()
                    else:
                        phi_prime = torch.ones_like(a)

            delta_j_R = dR_da_est * phi_prime         # (B, out_dim)

            # dW_ij = mean_b[ a_i_in * delta_j_R ]
            dW = torch.bmm(delta_j_R.unsqueeze(2), x_in.unsqueeze(1))  # (B, out_dim, in_dim)
            dW = dW.mean(dim=0)                                        # (out_dim, in_dim)

            db = delta_j_R.mean(dim=0)                                 # (out_dim,)

            layer.weight += eta * dW
            layer.bias   += eta * db

    return L_clean.mean().item()

def weight_perturb_step(model, X, target, eta=0.5, sigma=0.1):

    model.train()
    ys, xs, noises = model.forward_perturb(X, sigma)
    y_out = ys[-1]
    loss_vec = F.mse_loss(y_out, target, reduction='none')

    if loss_vec.dim() > 1:
        loss_vec = loss_vec.mean(dim=1)
    loss_vec = loss_vec.view(-1)

    with torch.no_grad():
        reward = -loss_vec.squeeze()   
        reward_bar = reward.mean()
        norm_reward = (reward - reward_bar)

        for layer_idx, (layer, (eps_w, eps_b), x_in, x_out) in enumerate(zip(model.layers, noises, xs, ys)):

            dW = ((eps_w * norm_reward.view(-1, 1, 1)).mean(dim=0) / (sigma + 1e-12)) * eta
            
            new_weights = layer.weight + dW
            layer.weight.copy_(new_weights)

            db = ((eps_b  * norm_reward.unsqueeze(1)).mean(dim=0) / (sigma + 1e-12)) * eta
            
            new_bias = layer.bias + db
            layer.bias.copy_(new_bias)

    return loss_vec.mean().item()

def weight_perturb_step_momentum(model, X, target, momentum_w, momentum_b, eta=0.5, sigma=0.1):

    model.train()
    ys, xs, noises = model.forward_perturb(X, sigma)
    y_out = ys[-1]
    loss_vec = F.mse_loss(y_out, target, reduction='none')

    if loss_vec.dim() > 1:
        loss_vec = loss_vec.mean(dim=1)
    loss_vec = loss_vec.view(-1)

    with torch.no_grad():
        reward = -loss_vec.squeeze()   
        reward_bar = reward.mean()
        norm_reward = (reward - reward_bar)

        for layer_idx, (layer, (eps_w, eps_b), x_in, x_out) in enumerate(zip(model.layers, noises, xs, ys)):
            momentum_w_l = momentum_w[layer_idx]
            momentum_b_l = momentum_b[layer_idx]

            dW = ((eps_w * norm_reward.view(-1, 1, 1)).mean(dim=0) / (sigma + 1e-12)) * eta

            momentum_w_l = 0.9 * momentum_w_l + dW
            new_weights = layer.weight + momentum_w_l
            layer.weight.copy_(new_weights)


            db = ((eps_b  * norm_reward.unsqueeze(1)).mean(dim=0) / (sigma + 1e-12)) * eta

            momentum_b_l = 0.9 * momentum_b_l + db
            new_bias = layer.bias + momentum_b_l
            layer.bias.copy_(new_bias)

            momentum_w[layer_idx] = momentum_w_l
            momentum_b[layer_idx] = momentum_b_l

    return loss_vec.mean().item(), momentum_w, momentum_b

def backprop_step(model, X, target, optimizer, loss_fn=F.mse_loss):

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

