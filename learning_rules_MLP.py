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

    def plot_weight_distributions(
        self,
        model,
        bins=50,
        include_bias=True,
        title=None,
        save_path=None,
        show=True,
    ):
        """
        Plot per-layer weight (and optional bias) distributions for a model with `.layers`.
        Returns (fig, axes).
        """
        n_layers = len(model.layers)
        if n_layers == 0:
            raise ValueError("Model has no layers to plot.")

        fig, axes = plt.subplots(n_layers, 1, figsize=(7, 3 * n_layers), squeeze=False)
        axes = axes.flatten()

        for idx, layer in enumerate(model.layers):
            w = layer.weight.detach().cpu().numpy().ravel()
            ax = axes[idx]
            ax.hist(w, bins=bins, alpha=0.7, label=f"Layer {idx} weights")

            if include_bias and layer.bias is not None:
                b = layer.bias.detach().cpu().numpy().ravel()
                ax.hist(b, bins=bins, alpha=0.7, label=f"Layer {idx} bias")

            ax.set_ylabel("Count")
            ax.set_xlabel("Value")
            ax.legend(loc="best")

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        if save_path:
            import os

            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes

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

    def forward_node_perturb(self, x, sigma):
        """
        Forward pass for node perturbation using the exact pre-activation noise
        variance induced by weight and bias perturbations with standard deviation `sigma`.
        """
        acts = [x]
        noises = []
        noise_scales = []
        last = len(self.layers) - 1
        a_noisy = x

        for i, layer in enumerate(self.layers):
            x_in = a_noisy
            z_noisy = layer(a_noisy)
            eps = torch.randn_like(z_noisy, device=z_noisy.device, dtype=z_noisy.dtype)
            noise_scale = sigma * torch.sqrt(1.0 + x_in.pow(2).sum(dim=1, keepdim=True))
            noises.append(eps)
            noise_scales.append(noise_scale)
            z_noisy = z_noisy + noise_scale * eps
            if i == last:
                a_noisy = self.output_activation(z_noisy) if self.output_activation else z_noisy
            else:
                a_noisy = self.activation(z_noisy)
            acts.append(a_noisy)

        return acts, noises, noise_scales, a_noisy

def _mean_loss_per_sample(prediction, target):
    loss = F.mse_loss(prediction, target, reduction='none')
    if loss.dim() > 1:
        loss = loss.mean(dim=1)
    return loss.view(-1)


def _centered_reward_signal(loss_per_sample):
    reward = -loss_per_sample
    reward_bar = reward.mean()
    return reward - reward_bar


def node_perturbation_step(model, X, target, eta=0.5, sigma=0.1):
    """
    Node perturbation with a centered reward signal from the noisy forward pass.
    """
    model.train()
    activations, noises, noise_scales, prediction_noisy = model.forward_node_perturb(X, sigma)
    loss_per_sample = _mean_loss_per_sample(prediction_noisy, target)
    scalar_signal = _centered_reward_signal(loss_per_sample)

    with torch.no_grad():
        for layer, x_in, noise, noise_scale in zip(model.layers, activations, noises, noise_scales):
            scaled_noise = scalar_signal.view(-1, 1) * noise / (noise_scale + 1e-12)
            weight_update = eta * torch.bmm(scaled_noise.unsqueeze(2), x_in.unsqueeze(1)).mean(dim=0)
            bias_update = eta * scaled_noise.mean(dim=0)

            layer.weight += weight_update
            layer.bias += bias_update

    return loss_per_sample.mean().item()


def weight_perturb_step(model, X, target, eta=0.5, sigma=0.1):
    """
    Weight perturbation with a centered reward signal from the noisy forward pass.
    """
    model.train()
    layer_outputs, _, noises = model.forward_weight_perturb(X, sigma)
    prediction_noisy = layer_outputs[-1]
    loss_per_sample = _mean_loss_per_sample(prediction_noisy, target)
    scalar_signal = _centered_reward_signal(loss_per_sample)
    noise_scale = sigma ** 2 + 1e-12

    with torch.no_grad():
        for layer, (weight_noise, bias_noise) in zip(model.layers, noises):
            scaled_weight_noise = scalar_signal.view(-1, 1, 1) * weight_noise / noise_scale
            scaled_bias_noise = scalar_signal.view(-1, 1) * bias_noise / noise_scale
            weight_update = eta * scaled_weight_noise.mean(dim=0)
            bias_update = eta * scaled_bias_noise.mean(dim=0)

            layer.weight += weight_update
            layer.bias += bias_update

    return loss_per_sample.mean().item()



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

