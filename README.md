# ADMM Optimizer

[![PyPI version](https://badge.fury.io/py/snn-admm-optimizer.svg)](https://badge.fury.io/py/snn-admm-optimizer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/snn-admm-optimizer/badge/?version=latest)](https://snn-admm-optimizer.readthedocs.io/en/latest/?badge=latest)

**SNN ADMM Optimizer** is a PyTorch-native optimization library that applies the Alternating Direction Method of Multipliers (ADMM) to train both standard and Spiking Neural Networks (SNNs). 

By decomposing the complex global loss landscape into highly parallelizable, localized layer-wise subproblems, this library provides exact numerical solvers for spatial and temporal activation updates, bypassing the limitations of traditional backpropagation through time (BPTT).

## ✨ Features

* **PyTorch Native:** Built directly on top of `torch` and `torch.nn` for seamless integration with your existing models.
* **Spiking & Static Network Support:** Includes specialized managers for Affine layers (`Linear`, `Conv2d`) alongside a powerful `SpikingMixin` for modeling temporal leakage and spike resets.
* **Advanced Solvers:** Utilizes highly optimized linear algebra routines, including Woodbury identity, Cholesky decomposition, and Conjugate Gradient (CG) methods for least-squares weight updates.
* **Vectorized & Unrolled Training:** Flexible training modes allowing you to balance memory constraints with raw compute speed.
* **Lagrangian Multipliers:** Full support for dual variables and penalty parameter ($\rho$, $\beta$) tuning to enforce layer constraints.

## 📦 Installation

You can install the package directly via pip:

```bash
pip install admm

```

*(Note: If you are installing from source or for development)*

```bash
git clone [https://github.com/yourusername/snn-admm-optimizer.git](https://github.com/yourusername/snn-admm-optimizer.git)
cd admm
pip install -e .

```

## 🚀 Quick Start

Here is a minimal example of how to wrap your layers and initiate the ADMM optimization process.

```python
import torch
import torch.nn as nn
from admm import ADMM_AffineLayer, ADMM_Spiking, ADMM_LayerConfig

# 1. Define your layer configuration
config = ADMM_LayerConfig(
    rho=1.0, 
    beta=0.5, 
    deltas=0.9,  # Temporal leakage
    thetas=1.0,  # Spiking threshold
    use_reset=True,
    use_bias=True
)

# 2. Initialize an ADMM-managed Spiking Layer
class MySpikingLayer(ADMM_Spiking, ADMM_AffineLayer):
    def __init__(self):
        super().__init__(config=config)
        self._init_weights_and_bias(weight_shape=(100, 10), bias_shape=(10,))

layer = MySpikingLayer()

# 3. Perform a forward pass (temporal simulation)
dummy_input = torch.rand(10, 32, 100) # (Time, Batch, Features)
output = layer.forward(dummy_input)

print(f"Output shape: {output.shape}")

```

## 🧠 Architecture Overview

The core of the library is divided into two main domains:

* **Spatial Mathematics (`affine_layer.py`):** Manages trainable parameters ($W$, $b$). Computes $W^T W$ matrix covariances, spatial activation updates ($a$), and target tensors ($v$).
* **Temporal Mechanics (`spiking_mixin.py`):** Overrides standard spatial hooks to inject SNN mechanics. It handles time-unrolled matrix multiplications, temporal dependencies, and spiking linear algebra systems.

## 📚 Documentation

For full API references, tutorials, and mathematical derivations of the ADMM update steps, please visit our documentation:

👉 **[Read the Docs: SNN ADMM Optimizer](https://www.google.com/search?q=https://your-docs-link.com)**

To build the documentation locally, ensure you have `mkdocs` and `mkdocstrings` installed, then run:

```bash
mkdocs serve

```

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the solvers, add new layer types, or fix a bug, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

Please ensure all tests pass before submitting your PR:

```bash
python -m pytest -s tests/

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```
---

### Tips for this README:
1. **Badges:** Replace the URLs in the badges at the top with your actual GitHub/PyPI URLs once they exist. They instantly signal that the project is well-maintained.
2. **Quick Start:** The quick start uses the exact classes you provided (`ADMM_AffineLayer`, `ADMM_Spiking`). Adjust the `ADMM_LayerConfig` block if your configuration class is named differently.
3. **Documentation:** It explicitly mentions `mkdocs`, which ties back to your previous errors, letting users know exactly how to view the local docs.

```