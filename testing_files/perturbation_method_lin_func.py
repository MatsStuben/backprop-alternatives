import numpy as np

def generate_linear_data(a, b, n=200, x_min=-1.0, x_max=1.0, noise_std=0.1, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_min, x_max, size=(n, 1))
    y = a * x + b + rng.randn(n, 1) * noise_std
    return x, y

def train_perturbation(x, y, epochs=200, batch_size=64, sigma=0.1, eta=0.1, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn()  
    b = rng.randn()

    N = x.shape[0]

    for epoch in range(epochs):
        eps_a = rng.randn(batch_size)
        eps_b = rng.randn(batch_size)

        a_perturb = a + sigma * eps_a
        b_perturb = b + sigma * eps_b
        y_pred = x @ a_perturb.reshape(1, batch_size) + b_perturb.reshape(1, batch_size)

        mse = ((y_pred - y)**2).mean(axis=0) 

        rewards = -mse
        baseline = rewards.mean()
        advantages = rewards - baseline 

        grad_a = (advantages * eps_a).mean() / (sigma + 1e-12)
        grad_b = (advantages * eps_b).mean() / (sigma + 1e-12)

        a += eta * grad_a
        b += eta * grad_b

        if (epoch % 10) == 0 or epoch == epochs - 1:
            y_hat = a * x + b
            loss = ((y_hat - y)**2).mean()
            print(f"Epoch {epoch:4d}: loss={loss.item():.6f}, a={a:.6f}, b={b:.6f}")
            print(f"    grads: grad_a={grad_a:.6f}, grad_b={grad_b:.6f}")

    return a, b

if __name__ == "__main__":
    x, y = generate_linear_data(a=2.0, b=0.5, n=200, noise_std=0.0, seed=1)
    a_final, b_final = train_perturbation(x, y,
                                          epochs=400,
                                          batch_size=128,
                                          sigma=0.1,
                                          eta=0.1,
                                          seed=2)
    print(f"Trained params: a={a_final:.4f}, b={b_final:.4f}")