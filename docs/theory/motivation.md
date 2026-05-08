# ADMM Optimization Theory Motivation

The Alternating Direction Method of Multipliers (ADMM) offers a gradient-free approach to training deep learning models. This section outlines the mathematical motivation and theoretical framework behind using ADMM for both standard and Spiking Neural Networks (SNNs).

## The Limitations of Backpropagation

Most modern neural networks are trained using Stochastic Gradient Descent (SGD) algorithms driven by backpropagation[^1]. While highly optimized for single-GPU execution, gradient-based methods face several fundamental challenges:

*   **Poor Scaling:** SGD relies on frequent, computationally inexpensive updates using small data batches[^1]. This process causes heavy communication bottlenecks when distributed across multiple CPU cores, preventing linear scaling in cluster environments[^1].
*   **Optimization Landscape:** Gradient methods frequently suffer from vanishing gradients, poor conditioning, and a tendency to stall near saddle points[^1].
*   **Non-Differentiability in SNNs:** Spiking Neural Networks rely on a discrete, non-differentiable step function[^2]. To use BPTT (Backpropagation Through Time), researchers must use "surrogate gradients," which introduce approximation errors, require extra memory, and exacerbate the vanishing gradient problem[^2].

## The ADMM Splitting Scheme

The core principle of the ADMM approach is to decouple the network's layers[^1]. Instead of optimizing all weights simultaneously via the chain rule, ADMM introduces auxiliary variables to separate the linear weight operations from the non-linear activation functions[^1]. 

This converts the training process into a sequence of smaller, independent minimization sub-problems that can often be solved to global optimality in closed form[^1].

### Formulation for Standard Neural Networks

For a standard feed-forward network, the architecture is split by defining new variables for each layer $l$:

*   **Pre-activations ($z$):** Defined as $z_l = W_l a_{l-1}$[^1].
*   **Activations ($a$):** Defined as $a_l = h_l(z_l)$, where $h_l$ is the non-linear activation function[^1].

Instead of strictly enforcing these equalities, the optimization problem is relaxed by adding quadratic penalty functions to the objective loss[^1]. These penalties are controlled by hyperparameters (often denoted as $\beta$ and $\rho$)[^1]. To ensure the network converges to the correct global output, we introduce Lagrange multipliers[^1].

The training loop then alternates directions, holding all other variables constant while updating one specific block at a time[^1]:

1.  **Weight Updates:** Becomes a straightforward linear least-squares problem[^1].
2.  **Activation Updates:** Also reduces to a simple least-squares minimization[^1].
3.  **Pre-activation Updates:** Decomposes into independent, one-dimensional non-convex problems that are easily solved via lookup tables or if-then logic (such as for ReLU functions)[^1].

Note that the order of this steps could be swap as we will see in later sections, for simplicity for now we leave it like this. 

### Formulation for Spiking Neural Networks

Applying ADMM to SNNs introduces the complexity of the time dimension[^2]. The optimization variables must account for discrete time steps $t$:

*   **Membrane Potentials ($z_{l,t}$):** The internal state of the neurons[^2].
*   **Spikes ($a_{l,t}$):** The binary outputs of the neurons[^2].

The constraints for SNNs are far more complex because they must model the Leaky Integrate-and-Fire (LIF) neuronal dynamics[^2]. This includes an exponential decay factor (leakage) applied to previous time steps and a thresholding mechanism determining if a spike is emitted[^2]. 

Because the SNN activation function is a discontinuous Heaviside step function, ADMM utilizes a dedicated subroutine during the pre-activation ($z$) update[^2]. Rather than calculating a gradient, the optimizer directly evaluates the local Lagrangian cost under both conditions (spiking versus resting) and assigns the state that yields the lowest objective energy[^2].

## Advantages of the ADMM Framework

By abandoning gradient descent in favor of distributed sub-problems, the ADMM approach provides distinct theoretical advantages:

*   **Linear Scaling:** The minimization sub-steps inherently support data parallelism[^1]. ADMM exhibits strong linear scaling, allowing training to be efficiently distributed across thousands of compute cores[^1].
*   **No Vanishing Gradients:** Because weights are updated based on localized layer targets rather than downstream gradients, information is not annihilated by shallow layers[^1].
*   **Gradient-Free SNN Optimization:** The framework treats the non-differentiable spiking mechanics exactly, entirely avoiding the inaccuracies introduced by surrogate gradient approximations[^2].


## References
[^1]: Taylor, G., Burmeister, R., Xu, Z., Singh, B., Patel, A., & Goldstein, T. (2016). *Training Neural Networks Without Gradients: A Scalable ADMM Approach*. Proceedings of the 33rd International Conference on Machine Learning.
[^2]: Perin, G., Bidini, C., Mazzieri, R., & Rossi, M. (2025). *ADMM-Based Training for Spiking Neural Networks*. Preprint submitted to IEEE MLSP 2025.