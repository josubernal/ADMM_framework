# ADMM Optimizer

[![PyPI version](https://badge.fury.io/py/snn-admm-optimizer.svg)](https://badge.fury.io/py/snn-admm-optimizer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/snn-admm-optimizer/badge/?version=latest)](https://snn-admm-optimizer.readthedocs.io/en/latest/?badge=latest)

**SNN ADMM Optimizer** is a PyTorch-native optimization library that applies the Alternating Direction Method of Multipliers (ADMM) to train both standard and Spiking Neural Networks (SNNs).

By decomposing the complex global loss landscape into highly parallelizable, localized layer-wise subproblems, this library provides exact numerical solvers for spatial and temporal activation updates, bypassing the limitations of traditional backpropagation through time (BPTT).

## ✨ Features

* **PyTorch Native:** Built directly on top of `torch` and `torch.nn` to facilitate the learning curve.
* **Spiking & Static Network Support:** Includes specialized managers for Affine layers (`Linear`, `Conv2d`) alongside a powerful `SpikingMixin` for modeling temporal leakage and spike resets.
* **Vectorized, Decoupled & Unrolled Training:** Flexible training modes allowing you to balance memory constraints with raw compute speed.
* **Lagrangian Multipliers:** Full support for dual variables and penalty parameter ($\rho$, $\beta$) tuning to enforce layer constraints.

## 📦 Installation

You can install the package directly via pip:

```bash
pip install admm

```

*(Note: If you are installing from source or for development)*

```bash
git clone https://github.com/Neuromorphic-Intelligence-Systems-Lab/snn_paper_private/tree/affine
pip install -e .

```

## 🧠 Architecture Overview

The codebase is modularly designed to separate the mathematical solvers from the temporal mechanics and global memory management.

* Core Optimizer (`src/admm/`)

* Benchmarks & Scripts (`benchmarks/` / `scripts/`)

    A comprehensive suite of automated pipelines used to profile ADMM's hardware footprint and convergence behaviors.

*  Testing Suite (`tests/`)

* Documentation
    - **`README.md`**: The entry point for the project, detailing installation, features, and high-level architecture.
    - **`QUICK_GUIDE.md`**: A rapid-reference guide explaining the intuition behind global hyperparameters (`train_method`, state injectors) and layer-specific physics (penalty parameters $\rho$/$\beta$, leakage $\delta$, and thresholds $\theta$).
    - **`LIMITATIONS.md`**: A quick review of the current limitations of the framework, and possible future updates.

## 📚 Documentation

For full API references, tutorials, and mathematical derivations of the ADMM update steps, please visit our documentation:

👉 **[Read the Docs: SNN ADMM Optimizer](https://www.google.com/search?q=https://your-docs-link.com)**

To build the documentation locally, ensure you have `mkdocs` and `mkdocstrings` installed, then run:

```bash
mkdocs serve

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.