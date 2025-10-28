import torch
import numpy as np
import matplotlib.pyplot as plt
from three_factor_MLP import THREE_FACTOR_MLP, three_factor_step

import torch.nn as nn

def generate_data(n_samples=1000):
    x = torch.linspace(-2*np.pi, 2*np.pi, n_samples).unsqueeze(1)
    y = torch.sin(x) + 0.1 * torch.randn_like(x)  # sin(x) + small noise
    return x, y

X_train, y_train = generate_data(800)
X_test, y_test = generate_data(200)

# Three-factor MLP
print("Training Three-Factor MLP...")
model_3f = THREE_FACTOR_MLP([1, 64, 32, 1], activation=torch.sigmoid)
loss_fn = nn.MSELoss(reduction='none')

losses_3f = []
for epoch in range(10000):
    batch_size = 32
    idx = torch.randperm(len(X_train))[:batch_size]
    X_batch = X_train[idx]
    y_batch = y_train[idx]
    
    loss = three_factor_step(model_3f, X_batch, y_batch, loss_fn, 
                           eta=0.01, sigma=0.01)
    losses_3f.append(loss)
    
    if epoch % 2000 == 0:
        print(f"Variance in weights in layer 1:{model_3f.layers[0].weight.var().item():.6f} in layer 2: {model_3f.layers[1].weight.var().item():.6f} in layer 3: {model_3f.layers[2].weight.var().item():.6f}")
        print(f"Variance in biases in layer 1:{model_3f.layers[0].bias.var().item():.6f}")
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Standard MLP with backprop
print("\nTraining Standard MLP with Backprop...")
class StandardMLP(nn.Module):
    def __init__(self, dimensions, activation=torch.sigmoid):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i+1], bias=True)
                                     for i in range(len(dimensions)-1)])
        self.activation = activation
    
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # No activation on output layer
                h = self.activation(h)
        return h

model_bp = StandardMLP([1, 64, 32, 1], activation=torch.sigmoid)
optimizer = torch.optim.Adam(model_bp.parameters(), lr=0.001)
loss_fn_bp = nn.MSELoss()

losses_bp = []
for epoch in range(10000):
    batch_size = 32
    idx = torch.randperm(len(X_train))[:batch_size]
    X_batch = X_train[idx]
    y_batch = y_train[idx]
    
    optimizer.zero_grad()
    y_pred = model_bp(X_batch)
    loss = loss_fn_bp(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    
    losses_bp.append(loss.item())
    
    if epoch % 2000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluate both models
model_3f.eval()
model_bp.eval()
with torch.no_grad():
    y_pred_3f = model_3f(X_test)
    test_loss_3f = nn.MSELoss()(y_pred_3f, y_test)
    
    y_pred_bp = model_bp(X_test)
    test_loss_bp = nn.MSELoss()(y_pred_bp, y_test)
    
    print(f'\nThree-Factor Test Loss: {test_loss_3f:.4f}')
    print(f'Backprop Test Loss: {test_loss_bp:.4f}')

# Create figure with comparison plots
fig = plt.figure(figsize=(18, 10))

# Training loss comparison
plt.subplot(3, 3, 1)
plt.plot(losses_3f, label='Three-Factor', alpha=0.7)
plt.plot(losses_bp, label='Backprop', alpha=0.7)
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')

# Three-factor predictions
plt.subplot(3, 3, 2)
X_sorted, idx = torch.sort(X_test.squeeze())
y_true_sorted = y_test[idx].squeeze()
y_pred_3f_sorted = y_pred_3f[idx].squeeze()

plt.plot(X_sorted, y_true_sorted, 'b-', label='True', alpha=0.7)
plt.plot(X_sorted, y_pred_3f_sorted, 'r-', label='Three-Factor')
plt.title(f'Three-Factor Predictions (Loss: {test_loss_3f:.4f})')
plt.xlabel('x')
plt.ylabel('sin(x) + noise')
plt.legend()

# Backprop predictions
plt.subplot(3, 3, 3)
y_pred_bp_sorted = y_pred_bp[idx].squeeze()

plt.plot(X_sorted, y_true_sorted, 'b-', label='True', alpha=0.7)
plt.plot(X_sorted, y_pred_bp_sorted, 'g-', label='Backprop')
plt.title(f'Backprop Predictions (Loss: {test_loss_bp:.4f})')
plt.xlabel('x')
plt.ylabel('sin(x) + noise')
plt.legend()

# Weight distributions for three-factor model
for i, layer in enumerate(model_3f.layers):
    plt.subplot(3, 3, 4 + i)
    weights = layer.weight.detach().cpu().numpy().flatten()
    plt.hist(weights, bins=50, alpha=0.7, edgecolor='black', color='red')
    plt.title(f'3F Layer {i+1} Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    mean_w = weights.mean()
    std_w = weights.std()
    plt.text(0.02, 0.98, f'Mean: {mean_w:.4f}\nStd: {std_w:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Weight distributions for backprop model
for i, layer in enumerate(model_bp.layers):
    plt.subplot(3, 3, 7 + i)
    weights = layer.weight.detach().cpu().numpy().flatten()
    plt.hist(weights, bins=50, alpha=0.7, edgecolor='black', color='green')
    plt.title(f'BP Layer {i+1} Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    mean_w = weights.mean()
    std_w = weights.std()
    plt.text(0.02, 0.98, f'Mean: {mean_w:.4f}\nStd: {std_w:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print weight statistics for both models
print("\nThree-Factor Weight Statistics:")
for i, layer in enumerate(model_3f.layers):
    weights = layer.weight.detach().cpu().numpy()
    print(f"Layer {i+1}: shape={weights.shape}, mean={weights.mean():.4f}, "
          f"std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")

print("\nBackprop Weight Statistics:")
for i, layer in enumerate(model_bp.layers):
    weights = layer.weight.detach().cpu().numpy()
    print(f"Layer {i+1}: shape={weights.shape}, mean={weights.mean():.4f}, "
          f"std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")

# ---- New: plot sigmoid output distributions per layer ----
def collect_sigmoid_activations(model, X):
    model.eval()
    with torch.no_grad():
        h = X
        acts = []
        for i, layer in enumerate(model.layers):
            pre = layer(h)                    # pre-activation
            a = torch.sigmoid(pre)            # sigmoid output for this layer
            acts.append(a)
            # propagate using the model's typical behavior: hidden uses activation, output is linear
            if i < len(model.layers) - 1:
                h = a
            else:
                h = pre
        return acts

with torch.no_grad():
    acts_3f = collect_sigmoid_activations(model_3f, X_test)
    acts_bp = collect_sigmoid_activations(model_bp, X_test)

fig2 = plt.figure(figsize=(18, 6))
num_layers = len(model_3f.layers)

# Three-Factor activations
for i, a in enumerate(acts_3f):
    plt.subplot(2, num_layers, i + 1)
    vals = a.detach().cpu().numpy().ravel()
    plt.hist(vals, bins=50, range=(0, 1), alpha=0.8, edgecolor='black', color='red')
    plt.title(f'3F Sigmoid L{i+1}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xlim(0, 1)

# Backprop activations
for i, a in enumerate(acts_bp):
    plt.subplot(2, num_layers, num_layers + i + 1)
    vals = a.detach().cpu().numpy().ravel()
    plt.hist(vals, bins=50, range=(0, 1), alpha=0.8, edgecolor='black', color='green')
    plt.title(f'BP Sigmoid L{i+1}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xlim(0, 1)

plt.tight_layout()
plt.show()