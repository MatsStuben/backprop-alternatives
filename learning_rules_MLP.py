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

    def forward_weight_perturb(self, x, sigma):
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
            eps_w = torch.randn(batch_size, *w.shape, device=w.device, dtype=w.dtype) * sigma
            eps_b = torch.randn(batch_size, *b.shape, device=b.device, dtype=b.dtype) * sigma

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

    def forward_activation_perturb(self, x, sigma):
        acts = [x]
        noises = []
        last = len(self.layers) - 1
        a_noisy = x

        for i, layer in enumerate(self.layers):
            z_clean = layer(acts[-1])
            if i == last:
                a_clean = self.output_activation(z_clean) if self.output_activation else z_clean
            else:
                a_clean = self.activation(z_clean)
            acts.append(a_clean)

            z_noisy = layer(a_noisy)
            if i == last:
                a_noisy = self.output_activation(z_noisy) if self.output_activation else z_noisy
            else:
                a_noisy = self.activation(z_noisy)

            eps = torch.randn_like(a_noisy, device=a_noisy.device, dtype=a_noisy.dtype)
            noises.append(eps)
            a_noisy = a_noisy + sigma * eps

        return acts, noises, a_noisy

    def forward_activation_perturb_noisy(self, x, sigma):
        acts = [x]
        noises = []
        last = len(self.layers) - 1
        a_noisy = x

        for i, layer in enumerate(self.layers):
            z_noisy = layer(a_noisy)
            eps = torch.randn_like(z_noisy, device=z_noisy.device, dtype=z_noisy.dtype)
            noises.append(eps)
            z_noisy = z_noisy + sigma * eps
            if i == last:
                a_noisy = self.output_activation(z_noisy) if self.output_activation else z_noisy
            else:
                a_noisy = self.activation(z_noisy)
            acts.append(a_noisy)

        return acts, noises, a_noisy

def three_factor_weight_step(model, X, target, eta=0.5, sigma=0.1):

    model.train()
    ys, xs, noises = model.forward_weight_perturb(X, sigma)
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
    Three-factor node-perturbation update:
    grad ≈ (reward - baseline) * (eps / sigma) * activation_derivative * pre-activity
    """
    model.train()
    last = len(model.layers) - 1

    acts, noises, y_noisy = model.forward_activation_perturb(X, sigma)
    clean_output = acts[-1]

    loss_noisy = F.mse_loss(y_noisy, target, reduction='none')
    if loss_noisy.dim() > 1:
        loss_noisy = loss_noisy.mean(dim=1)
    loss_vec = loss_noisy.view(-1)

    with torch.no_grad():
        reward = -loss_vec
        baseline = reward.mean()
        advantage = reward - baseline  # (B,)

        for i, layer in enumerate(model.layers):
            x_in = acts[i]        # layer input
            act = acts[i+1]       # layer clean output
            eps = noises[i]

            act_deriv = _activation_derivative(
                act,
                model.output_activation if i == last else model.activation
            )

            delta = advantage.view(-1, 1) * eps / (sigma + 1e-12)
            delta *= act_deriv

            dW = torch.bmm(delta.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0)
            db = delta.mean(dim=0)

            layer.weight += eta * dW
            layer.bias += eta * db

    return F.mse_loss(clean_output, target, reduction='mean').item()

def three_factor_activation_step_noisy(model, X, target, eta=0.5, sigma=0.1):
    model.train()
    last = len(model.layers) - 1

    acts, noises, y_noisy = model.forward_activation_perturb_noisy(X, sigma)

    loss_noisy = F.mse_loss(y_noisy, target, reduction='none')
    if loss_noisy.dim() > 1:
        loss_noisy = loss_noisy.mean(dim=1)
    loss_vec = loss_noisy.view(-1)

    with torch.no_grad():
        reward = -loss_vec
        baseline = reward.mean()
        advantage = reward - baseline

        for i, layer in enumerate(model.layers):
            x_in = acts[i]
            act = acts[i+1]
            eps = noises[i]

            act_deriv = _activation_derivative(
                act,
                model.output_activation if i == last else model.activation
            )

            delta = advantage.view(-1, 1) * eps / (sigma + 1e-12)
            delta *= act_deriv

            dW = torch.bmm(delta.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0)
            db = delta.mean(dim=0)

            layer.weight += eta * dW
            layer.bias += eta * db

    return loss_vec.mean().item()

def weight_perturb_step(model, X, target, eta=0.5, sigma=0.1):

    model.train()
    ys, xs, noises = model.forward_weight_perturb(X, sigma)
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
    ys, xs, noises = model.forward_weight_perturb(X, sigma)
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

def _activation_derivative(act, activation):
    """
    Returns φ'(z) expressed via the activation output `act`.
    Supports None, sigmoid, tanh, relu, and defaults to 1.
    """
    if activation is None:
        return torch.ones_like(act)
    if activation is torch.sigmoid:
        return act * (1 - act)
    if activation is torch.tanh:
        return 1 - act.pow(2)
    if activation is torch.relu:
        return (act > 0).float()
    return torch.ones_like(act)

