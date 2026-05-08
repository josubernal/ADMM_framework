"""
This module contains the activation functions designed specifically for ADMM.

In the ADMM framework, activations are treated as constraints or non-linear penalties
rather than standard forward-pass operations. Therefore, each activation function
must provide its own analytical or proximal $z$-update step.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .dataclasses import ADMMLayerConfig, ADMMState


class ADMMActivationBase(nn.Module, ABC):
    """Abstract Base Class for ADMM Activation Functions."""

    def __init__(self):
        super().__init__()
        self.config: ADMMLayerConfig = ADMMLayerConfig()

    def _setup(self, config: ADMMLayerConfig) -> None:
        """Receives and stores ADMM hyperparameters from the parent layer.

        Args:
            config (ADMMLayerConfig): The layer-specific configuration container.
        """
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass applying the activation non-linearity."""
        pass

    @abstractmethod
    def activation_z_update(self, state: ADMMState) -> torch.Tensor:
        """Computes the proximal update for the pre-activation tensor ($z$)."""
        pass


####################################################################################################
# ACTIVATION FUNCTIONS
####################################################################################################


class ADMM_Identity(ADMMActivationBase):
    """Place-holder identity activation function for linear outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def activation_z_update(self, state: ADMMState) -> torch.Tensor:
        return state.forward


class ADMM_ReLU(ADMMActivationBase):
    """ADMM implementation of the Rectified Linear Unit (ReLU)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def activation_z_update(self, state: ADMMState) -> torch.Tensor:
        r"""Calculates the proximal update for ReLU.

        Formula evaluated:
        $z_{l} = \max\left(0, \frac{\beta_l a_l + \rho_l \text{forward}(a_l)}{\beta_l + \rho_l}\right)$

        Args:
            state (ADMMState): The current ADMM state containing spatial variables.

        Returns:
            torch.Tensor: The updated $z$ tensor.
        """
        z = (self.config.beta * state.a + self.config.rho * state.forward) / (
            self.config.beta + self.config.rho
        )
        return torch.where(z > 0, z, state.forward)


class ADMM_Heaviside(ADMMActivationBase):
    r"""ADMM implementation of a Heaviside (Step) Function for Spiking Networks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.config.thetas).to(x.dtype)

    def check_entries(
        self, state: ADMMState, is_vectorized: bool = False
    ) -> torch.Tensor:
        r"""Universal boolean masking for ADMM discrete energy evaluation.

        Enforces the non-convex Heaviside domain constraints by explicitly evaluating
        the discrete Lagrangian energy difference between the spiking and resting states.

        Mathematical Formulation:

        $\delta_1 = \beta (1 - 2a)$

        $\delta_2 = \rho ((z - y_{temporal})^2 - (\theta - y_{temporal})^2)$

        $\delta_3 = \rho ((z_{res} - \delta z + \theta a)^2 - ( z_{res} - \delta \theta + \theta a)^2)$

        Condition logic:

        * $z = \theta$ if $z > \theta$ and $\delta_1 + \delta_2 + \delta_3 > 0$
        * $z = \theta + \epsilon$ if $z \le \theta$ and $\delta_2 + \delta_3 - \delta_1 > 0$

        Args:
            state (ADMMState): The current ADMM state containing auxiliary variables.
            is_vectorized (bool, optional): Flag indicating if this is a full sequence update
                (applies the indicator function to zero out $\delta_3$ at $t=T$). Defaults to False.

        Returns:
            torch.Tensor: The projected and bounded $z$ tensor.
        """
        z = state.z
        temporal_forward = state.forward
        a = state.a

        delta1 = a.mul(-2.0).add_(1.0).mul_(self.config.beta)

        total_delta = z.sub(temporal_forward).pow_(2)
        tmp = temporal_forward.sub(self.config.thetas).pow_(2)
        total_delta.sub_(tmp).mul_(self.config.rho)
        total_delta.add_(delta1)  # total_delta is now delta1 + delta2

        if state.z_minus_forward is not None:
            res_term = state.z_minus_forward.add(a, alpha=self.config.thetas)

            d3 = res_term.sub(self.config.deltas * z).pow_(2)
            d3_sub = res_term.sub(self.config.deltas * self.config.thetas).pow_(2)
            d3.sub_(d3_sub).mul_(self.config.rho)

            if is_vectorized:
                d3[-1].zero_()

            total_delta.add_(d3)

        mask1 = (z > self.config.thetas).logical_and_(total_delta > 0)

        total_delta.add_(delta1, alpha=-2.0)
        mask2 = (z <= self.config.thetas).logical_and_(total_delta > 0)

        z.masked_fill_(mask1, self.config.thetas)
        z.masked_fill_(mask2, self.config.thetas + 1e-5)
        return z

    def activation_z_unrolled(self, state: ADMMState) -> torch.Tensor:
        r"""Unrolled version of the update, handling specific time-step logic.

        Formula evaluated:

        * For $t < T$: $z_t = \frac{y_{temporal} + \delta(z_{t+1} - y_{t+1} + \theta a_t)}{1 + \delta^2}$
        * For $t = T$: $z_t = y_{temporal}$

        Args:
            state (ADMMState): The current ADMM state tracking variables for the active timestep.

        Returns:
            torch.Tensor: The updated $z$ tensor for the current timestep.
        """
        temporal_forward = state.forward
        z_minus_forward = state.z_minus_forward
        a_t = state.a

        if z_minus_forward is not None:
            z_unconstrained = z_minus_forward.add(a_t, alpha=self.config.thetas)
            z_unconstrained.mul_(self.config.deltas).add_(temporal_forward)
            z_unconstrained.div_(1.0 + self.config.deltas**2)
        else:
            z_unconstrained = temporal_forward.clone()

        check_state = ADMMState(
            forward=temporal_forward,
            a=a_t,
            z=z_unconstrained,
            z_minus_forward=z_minus_forward,
        )
        return self.check_entries(check_state, is_vectorized=False)

    def activation_z_update(self, state: ADMMState) -> torch.Tensor:
        r"""Executes a Pure Vectorized (Jacobi) block of the $z$-update.

        Simultaneously computes the unconstrained proximal step for the entire
        temporal sequence, followed by parallel discrete constraint projection.

        Args:
            state (ADMMState): The current ADMM state containing full sequence variables.

        Returns:
            torch.Tensor: The updated $z$ tensor evaluated across all timesteps.
        """
        forward = state.forward
        z = state.z
        a = state.a

        # 1. Compute temporal forward pass (t)
        temporal_forward = forward.clone()
        temporal_forward[1:].add_(z[:-1], alpha=self.config.deltas)
        temporal_forward[1:].add_(a[:-1], alpha=-self.config.thetas)

        # 2. Compute residual of the next timestep (t+1)
        z_minus_forward_next = torch.zeros_like(z)
        z_minus_forward_next[:-1].copy_(z[1:]).sub_(forward[1:])

        # 3. Calculate unconstrained optimal z
        numerator = temporal_forward.mul(self.config.rho)
        denominator = torch.full_like(numerator, self.config.rho)

        num_slice = numerator[:-1]
        temporal_term = z[1:].sub(forward[1:]).add_(a[:-1], alpha=self.config.thetas)
        num_slice.add_(temporal_term, alpha=self.config.deltas * self.config.rho)

        denominator[:-1].add_((self.config.deltas**2) * self.config.rho)
        z_unconstrained = numerator.div_(denominator)

        # 4. Evaluate Discrete Constraints
        check_state = ADMMState(
            forward=temporal_forward,
            a=a,
            z=z_unconstrained,
            z_minus_forward=z_minus_forward_next,
        )
        return self.check_entries(check_state, is_vectorized=True)
