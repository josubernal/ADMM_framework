"""
This module contains the activation functions designed specifically for ADMM.

In the ADMM framework, activations are treated as constraints or non-linear penalties
rather than standard forward-pass operations. Therefore, each activation function
must provide its own analytical or proximal $z$-update step.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .dataclasses import ADMM_LayerConfig, ADMM_State


class ADMM_ActivationBase(nn.Module, ABC):
    """Abstract Base Class for ADMM Activation Functions."""

    def __init__(self):
        super().__init__()
        self.config: ADMM_LayerConfig = ADMM_LayerConfig()

    def _setup(self, config: ADMM_LayerConfig) -> None:
        """Receives and stores ADMM hyperparameters from the parent layer.

        Args:
            config (ADMM_LayerConfig): The layer-specific configuration container.
        """
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass applying the activation non-linearity."""
        pass

    @abstractmethod
    def activation_z_update(self, state: ADMM_State) -> torch.Tensor:
        """Computes the proximal update for the pre-activation tensor ($z$)."""
        pass


####################################################################################################
# ACTIVATION FUNCTIONS
####################################################################################################


class ADMM_Identity(ADMM_ActivationBase):
    """Place-holder identity activation function for linear outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def activation_z_update(self, state: ADMM_State) -> torch.Tensor:
        return state.forward


class ADMM_ReLU(ADMM_ActivationBase):
    """ADMM implementation of the Rectified Linear Unit (ReLU)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def activation_z_update(self, state: ADMM_State) -> torch.Tensor:
        r"""Calculates the proximal update for ReLU.

        Formula evaluated:
        $$
        z_l \leftarrow 
        \begin{cases}
        \frac{\beta_l a_l + \rho_l F_l(a_{l-1})}{\beta_l + \rho_l} & \text{if } z_l > 0 \\\\
        F_l(a_{l-1}) & \text{if } z_l \leq 0
        \end{cases}
        $$

        Args:
            state (ADMM_State): The current ADMM state containing spatial variables.

        Returns:
            torch.Tensor: The updated $z$ tensor.
        """
        z = (self.config.beta * state.a + self.config.rho * state.forward) / (
            self.config.beta + self.config.rho
        )
        return torch.where(z > 0, z, state.forward)


class ADMM_Heaviside(ADMM_ActivationBase):
    r"""ADMM implementation of a Heaviside (Step) Function for Spiking Networks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.config.thetas).to(x.dtype)

    def check_entries(
        self, state: ADMM_State, is_vectorized: bool = False
    ) -> torch.Tensor:
        r"""Universal boolean masking for ADMM discrete energy evaluation.

        Enforces the non-convex Heaviside domain constraints by explicitly evaluating
        the discrete Lagrangian energy difference between the spiking and resting states.

        Mathematical Formulation:

        * $\Delta_1 = \beta(1 - 2a_{l,t})$
        * $\Delta_2 = \rho \left( (z_{l,t}^* - \mathcal{F}_{l,t})^2 - (\theta - \mathcal{F}_{l,t})^2 \right)$
        * $\Delta_3 = \rho \left( (z_{l,t+1} - F_l(a_{l-1,t+1}) - \delta z_{l,t}^* + \theta a_{l,t})^2 - (z_{l,t+1} - F_l(a_{l-1,t+1}) - \delta \theta + \theta a_{l,t})^2 \right) \mathbb{1}_{\{t<T\}}$

        Condition logic:

        $$ z_{l,t} \leftarrow \begin{cases} \theta & \text{if } z_{l,t}^* > \theta \text{ and } (\Delta_1 + \Delta_2 +\Delta_3) > 0 \cr \theta + \epsilon & \text{if } z_{l,t}^* \leq \theta \text{ and } (\Delta_2 +\Delta_3 - \Delta_1) > 0 \cr z_{l,t}^* & \text{otherwise} \end{cases} $$

        Args:
            state (ADMM_State): The current ADMM state containing auxiliary variables.
            is_vectorized (bool, optional): Flag indicating if this is a full sequence update
                (applies the indicator function to zero out $\delta_3$ at $t=T$). Defaults to False.

        Returns:
            torch.Tensor: The projected and bounded $z$ tensor.
        """
        z_star = state.z
        temporal_forward = state.forward
        a = state.a

        delta1 = a.mul(-2.0).add_(1.0).mul_(self.config.beta)

        total_delta = z_star.sub(temporal_forward).pow_(2)
        tmp = temporal_forward.sub(self.config.thetas).pow_(2)
        total_delta.sub_(tmp).mul_(self.config.rho)  # total_delta is now delta2
        total_delta.add_(delta1)  # total_delta is now delta1 + delta2

        if state.z_minus_forward is not None:
            res_term = state.z_minus_forward.add(a, alpha=self.config.thetas)

            d3 = res_term.sub(self.config.deltas * z_star).pow_(2)
            d3_sub = res_term.sub(self.config.deltas * self.config.thetas).pow_(2)
            d3.sub_(d3_sub).mul_(self.config.rho)

            if is_vectorized:
                d3[-1].zero_()

            total_delta.add_(d3)  # total_delta is now delta1 + delta2 + delta3

        mask1 = (z_star > self.config.thetas).logical_and_(total_delta > 0)

        total_delta.add_(
            delta1, alpha=-2.0
        )  # total_delta is now delta2 + delta3 - delta1
        mask2 = (z_star <= self.config.thetas).logical_and_(total_delta > 0)

        z_star.masked_fill_(mask1, self.config.thetas)
        z_star.masked_fill_(mask2, self.config.thetas + 1e-5)
        return z_star

    def activation_z_unrolled(self, state: ADMM_State) -> torch.Tensor:
        r"""Unrolled version of the $z$ update.

        Formula evaluated:

        $$z_{l,t}^* = \frac{\mathcal{F}_{l,t} + \delta\big(z_{l,t+1} - F_l(a_{l-1,t+1}) + \theta a_{l,t}\big)\mathbb{1}_{\{t<T\}}}{1 + \delta^2 \mathbb{1}_{\{t<T\}}}$$

        You can find the corresponding vectorized version in the [activation_z_update][admm.activations.ADMM_Heaviside.activation_z_update] method.

        Args:
            state (ADMMState): The current ADMM state tracking variables for the active timestep.

        Returns:
            torch.Tensor: The updated $z$ tensor for the current timestep.
        """
        temporal_forward = state.forward
        z_minus_forward = state.z_minus_forward
        a_t = state.a

        if z_minus_forward is not None:
            z_star = z_minus_forward.add(a_t, alpha=self.config.thetas)
            z_star.mul_(self.config.deltas).add_(temporal_forward)
            z_star.div_(1.0 + self.config.deltas**2)
        else:
            z_star = temporal_forward.clone()

        check_state = ADMM_State(
            forward=temporal_forward,
            a=a_t,
            z=z_star,
            z_minus_forward=z_minus_forward,
        )
        return self.check_entries(check_state, is_vectorized=False)

    def activation_z_update(self, state: ADMM_State) -> torch.Tensor:
        r"""Vectorized version of the $z$ update.

        Formula evaluated:

        $$z_l^* = \frac{\mathcal{F}_l + \delta S^\top \big(z_l - F_l(a_{l-1}) + \theta a_l\big)}{\mathcal{I} + \delta^2 S^\top S}$$

        You can find the corresponding unrolled version in the [activation_z_unrolled][admm.activations.ADMM_Heaviside.activation_z_unrolled] method.

        Args:
            state (ADMMState): The current ADMM state containing full sequence variables.

        Returns:
            torch.Tensor: The updated $z$ tensor evaluated across all timesteps.
        """
        forward = state.forward
        z = state.z
        a = state.a

        temporal_forward = forward.clone()
        temporal_forward[1:].add_(z[:-1], alpha=self.config.deltas)
        temporal_forward[1:].add_(a[:-1], alpha=-self.config.thetas)

        z_minus_forward_next = torch.zeros_like(z)
        z_minus_forward_next[:-1].copy_(z[1:]).sub_(forward[1:])

        z_star = z_minus_forward_next.add(a, alpha=self.config.thetas)
        z_star.mul_(self.config.deltas).add_(temporal_forward)

        denominator = torch.ones_like(z_star)
        denominator[:-1].add_(self.config.deltas**2)
        z_star.div_(denominator)

        z_star[-1] = temporal_forward[-1]  # Applying the boundary condition at t=T

        check_state = ADMM_State(
            forward=temporal_forward,
            a=a,
            z=z_star,
            z_minus_forward=z_minus_forward_next,
        )
        return self.check_entries(check_state, is_vectorized=True)
