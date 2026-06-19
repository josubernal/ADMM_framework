# ADMM Framework: Quick Configuration Guide

This guide provides a high-level overview of the configuration parameters used in the ADMM training framework. It is designed to quickly explain the intuition behind specific settings, making it easier to interpret training plots and experimental results.

---

## 1. Global Network Configurations (`ADMM_Config`)
These parameters dictate the global behavior of the ADMM optimization loop, the sequence of updates, and the mathematical solvers deployed.

### Training Methods (`train_method`)
Defines the algorithmic sequence used to update the network's states over time.
* **`vectorized`**: The default for static (non-spiking) networks. Processes the entire batch simultaneously. For spiking breaks the temporal dependencies apart, solving time-steps in parallel. (Jacobi update) 
* **`unrolled`** *(SNNs only)*: Solves the network strictly step-by-step across the time dimension ($t=0$ to $t=T$). Highly accurate but computationally heavier. (Gauss-Seidel update) 
* **`decoupled`** *(SNNs only)*: Breaks the temporal dependencies apart, solving the a-update in parallel. Performs z-update step-by-step.
It is a middleground between the other two approaches.

### Layer Ordering (`layer_order`)
The sequence in which layers are optimized during a single ADMM sweep.
* **`backwards`**: Computes from output to input (similar to backpropagation).
* **`sequential`**: Computes from input to output.
* **`random`** : Shuffles the layer update order.
* **`random-last`**: Shuffles the first n layer update order but secures updating the last layer the last.

### Time Ordering (`time_order`)
The sequence in which time are optimized during ADMM.
* **`backwards`**: Computes from output to input (similar to backpropagation).
* **`sequential`**: Computes from input to output.
* **`random`** : Shuffles the time update order.
* **`random-last`**: Shuffles the first n time update order but secures updating the last timestep the last.

### Weight Solvers (`solver`)
The mathematical method used to solve the least-squares problem for weight ($W$) updates.
* **`standard`**: Uses standard dense matrix inversion / pseudo-inverses.
* **`conjugate-gradient`**: Iterative solver.
* **`cholesky`**: Highly stable solver for positive-definite systems.

### Initialization (`init`)
In standard deep learning (backpropagation), initial weights determine where you start on the loss landscape. **In ADMM, initial weights don't matter** because the algorithm completely overwrites them in the very first step. 

Instead, ADMM needs starting values for the **auxiliary states** (the pre-activations $z$ and activations $a$). A better initialization jump-starts the ADMM consensus, reducing massive loss spikes at the beginning of training.

**Forward-Pass Seeders**
These strategies temporarily initialize the weights, push a "dummy" batch of data through the network, and capture the resulting $z$ and $a$ to use as the starting states.
* **`pytorch`**: Uses PyTorch’s standard Kaiming Uniform distribution.
* **`xavier`**: Uses Xavier Normal distribution for the warm-up pass.
* **`zeros`**: Starts all states entirely at zero. 

**Direct State Injectors**
These strategies bypass the forward pass entirely and inject raw random noise directly into the memory states.
* **`z-uniform`**: Injects random Gaussian noise into the pre-activations ($z$), and then calculates the corresponding activations ($a = h(z)$).
* **`s-uniform` (Default)**: Completely decouples the math. Both $z$ and $a$ are initialized with independent uniform noise $[0, 1)$.

**Spiking-Specific Injectors**
* **`relaxed`**: Designed specifically for Spiking Neural Networks (SNNs). Gives $z$ Gaussian noise to simulate membrane potential variance, but forces the activations ($a$) into a dense field of random, discrete spikes (0s and 1s).

### Sub-step Order (`update_z_first`)
* **`False` (Default)**: Updates activations ($a$) first, then pre-activations ($z$).
* **`True`**: Reverses the sub-step order.

---

## 2. Layer-Specific Configurations (`ADMM_LayerConfig`)
These parameters are tuned *per-layer*. They control the physical properties of the neurons and the rigidity of the ADMM math.

### Penalty Parameters (The "Rubber Bands")
* **`rho` ($\rho$)**: The affine penalty. It enforces the constraint $z = Wx + b$. 
    * *High $\rho$*: Forces the auxiliary variables to strictly obey the linear weights. 
* **`beta` ($\beta$)**: The activation penalty. It enforces the non-linear constraint $a = h(z)$. 
    * *High $\beta$*: Forces the network to strictly obey the activation function.

### Spiking Physics (SNNs Only)
* **`thetas` ($\theta$)**: The spiking threshold. The membrane potential required for the neuron to emit a spike.
* **`deltas` ($\delta$)**: Temporal leakage. Determines how much membrane potential is retained between time steps (e.g., $0.8 = 80\%$ retained).
* **`use_reset`**: If `True`, applies a soft-reset to the voltage after a spike is emitted. **Last layer must of SNNs be marked as False.**(This should be automatically done in a future update). 

### Algorithmic Toggles
* **`use_lagrange`**: Enables dual variable ($\lambda$) updates for the layer.
* **`use_bias`**: Enables a trainable bias vector for the layer.
* **`use_fft`**: If `True` (and used on Conv to Conv layers), solves the heavy $O(N^3)$ spatial bottlenecks entirely in the Fourier domain, massively accelerating the $a$-updates.
---
*For a deeper dive into the specific mathematical derivations or API usage, please refer to the main [ADMM Dataclasses Documentation](src.admm.dataclasses).*