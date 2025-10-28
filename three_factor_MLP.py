import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class THREE_FACTOR_MLP(nn.Module):
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
        xs = [x]   # x^0
        ys = []
        noises = []
        h = x
        for i, layer in enumerate(self.layers):
            w = layer.weight        
            b = layer.bias
            in_neurons = layer.in_features
            scaling_factor = math.sqrt(1.0 / in_neurons)
            eps_w = torch.randn(batch_size, *w.shape) * sigma * scaling_factor
            eps_b = torch.randn(batch_size, *b.shape) * sigma * scaling_factor * 0.5

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

def three_factor_step(model, X, target, loss_fn, eta=1e-3, sigma=0.01):
    """
    One training step using your three-factor rule on a mini-batch.
    loss_fn must return per-sample losses (reduction='none').
    """
    model.train()

    ys, xs, noises = model.forward_training(X, sigma)

    y_out = ys[-1]
    loss_vec = loss_fn(y_out, target)       

    with torch.no_grad():
        reward = -loss_vec.squeeze()   
        reward_bar = reward.mean()
        norm_reward = (reward - reward_bar) / (reward.std() + 1e-8)  
        h_in = xs   
        h_out = ys  
        
        for layer_idx, (layer, (eps_w, eps_b), x_in, x_out) in enumerate(zip(model.layers, noises, h_in, h_out)):
            scaling_factor = math.sqrt(1.0 / layer.in_features)
            eps_w = eps_w / ((scaling_factor) + 1e-8)
            eps_b = eps_b / ((scaling_factor*0.5) + 1e-8)
 
            eligibility_w = torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1))
            # I want eligibility_2 to be just 1, but need the correct shape. 
            dW = (eps_w * eligibility_w.transpose(1, 2) * norm_reward.view(-1, 1, 1) * eta).mean(dim=0)
            new_weights = layer.weight + dW
            normalized_weights = normalize_layer_weights(new_weights, layer.in_features, c=1)

            layer.weight.copy_(normalized_weights)

            eligibility_b = x_out
            db = (eps_b * eligibility_b * norm_reward.unsqueeze(1) * eta).mean(dim=0)
            new_bias = layer.bias + db
            normalized_bias = normalize_layer_bias(new_bias, layer.in_features, c=1)
            layer.bias.copy_(normalized_bias)

    return loss_vec.mean().item()

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

