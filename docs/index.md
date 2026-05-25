---
hide:
  - navigation
---
# ADMM Optimizer

**An Alternating Direction Method of Multipliers (ADMM) optimizer for Neural Networks (NNs) and Spiking Neural Networks (SNNs).**

Standard gradient-based training methods face several [fundamental challenges](theory/motivation.md):

* Inefficiencies when scaling and parallelizing across distributed hardware.
* Vulnerability to vanishing gradients, saddle points, and poor conditioning.
* Difficulty handling the non-differentiable nature of spiking neurons.

This package sidesteps backpropagation entirely by reframing neural network training as a distributed, layer-wise constrained optimization problem. While ADMM for deep learning remains an active area of research and may not yet match state-of-the-art gradient baselines in all scenarios, our goal is to provide a robust foundation for experimentation and to explore alternative paradigms in neural training.

To learn more about ADMM, and use the package correctly, we highly recommend to read or [theorical guide](theory/motivation.md) first.

## Key Features

* **PyTorch Native:** Built entirely on `torch.Tensor` operations. Layers, weights, and states seamlessly integrate with standard PyTorch device management.

---

## Installation

Install the package via `pip` (or your preferred package manager like `uv`):

```bash
pip install admm
```

*(Note: Requires PyTorch 2.0.0 or higher).*

---

## Quick Start

Training an ADMM network is different from standard PyTorch. Instead of `loss.backward()` and `optimizer.step()`, you initialize an ADMM orchestrator that handles the proximal updates and dual variables automatically.

Here is how to train a basic 2-layer Spiking Neural Network:

```python
import torch
from admm import ADMM, ADMM_FeedForward, ADMM_ReLU

# 1. Define your network geometry
in_features = 784
hidden_features = 128
out_features = 10
batch_size = 32


# 2. Instantiate ADMM Layers
layer1 = ADMM_FeedForward(in_features, hidden_features, rho=1, beta=1, h=ADMM_ReLU())
layer2 = ADMM_FeedForward(hidden_features, out_features, rho=1, beta=1)

# 3. Initialize the Orchestrator
model = ADMM(
    layers=[layer1, layer2], device="cuda" if torch.cuda.is_available() else "cpu"
)

# 4. Train (ADMM replaces standard forward/backward passes)
inputs = torch.randn(batch_size, in_features, device=model.device)
labels = torch.ones(batch_size, out_features, device=model.device)

# A single call executes the full layer-wise optimization loop
model.fit(inputs, labels)

```

---

## Where to go next?

* **[Tutorials](tutorials/getting_started.md):** Step-by-step guides from basic networks to complex spatiotemporal convolutions.
* **[Theory](theory/motivation.md):** Deep dive into the mathematical formulation of the ADMM splitting scheme and SNN temporal unrolling.
* **[API Reference](api/manager.md):** Comprehensive documentation of all layers, managers, and functional math operations.


***
