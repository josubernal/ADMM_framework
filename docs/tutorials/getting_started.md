# Getting Started

Welcome to the **ADMM Optimizer Framework**. This guide will take you from installation to training your first Alternating Direction Method of Multipliers (ADMM) network in less than two minutes.

## 1. Installation

Install the framework using pip:

```bash
pip install admm

```

## 2. Your First ADMM Network

Unlike traditional PyTorch networks that rely on `loss.backward()` and Backpropagation, this framework uses the ADMM manager to handle forward passes and localized spatial/temporal weight updates.

Here is a complete, runnable example of a 2-layer network:

```python
import torch

from admm import ADMM, ADMM_Config, ADMM_FeedForward, ADMM_ReLU

# 1. Setup Environment and Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, in_features, hidden, out_features = 32, 784, 128, 10

dummy_x = torch.randn(batch_size, in_features, device=device)
dummy_y = torch.randn(batch_size, out_features, device=device)

# 2. Define the Global Configuration
# This dictates initialization and overall configuration for the manager.
config = ADMM_Config(init="pytorch", train_method="vectorized")

# 3. Create ADMM-compatible Layers
layer1 = ADMM_FeedForward(in_features, hidden, h=ADMM_ReLU(), rho=1.0, beta=1.0)
layer2 = ADMM_FeedForward(hidden, out_features, h=ADMM_ReLU(), rho=1.0, beta=1.0)

# 4. Wrap in the ADMM Manager
model = ADMM(layers=[layer1, layer2], device=device, config=config)

# 5. Execute an Optimization Step
# Forward pass
output = model.forward_model(dummy_x)

# ADMM Weight Update (Replaces Backprop!)
model.fit(dummy_x, dummy_y)

print("Forward pass and ADMM updates completed successfully!")

```

## 3. What just happened?

If you are coming from standard PyTorch, you might have noticed a few differences:

* **No `torch.optim` or `loss.backward()`:** ADMM does not use chain-rule gradients. Instead, the `model.fit()` method solves a series of subproblems to update the weights.
* **The Manager (`ADMM`):** Because ADMM requires layers to coordinate auxiliary variables (like the $z$ and $u$ penalties), you pass your layers into the `ADMM` manager. The manager acts as the conductor, orchestrating the forward sweeps and the backward spatial updates.
* **Global Config:** Instead of setting initialization strategies on every single layer, you define an `ADMMConfig` once and the manager automatically cascades it down to all layers.

## Next Steps

Now that you have the basics running, check out the [Layers API][src.admm.layers] to see how to build Convolutional or Spiking networks!

```
