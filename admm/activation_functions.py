r"""
This module contains the activation functions designed specifically for ADMM.

In the standard Backpropagation framework, activations simply apply a non-linear
transformation during the forward pass. However, in the ADMM framework, activations
are treated as non-linear penalties. Therefore, each activation
function must provide its own analytical $z$-update step to resolve the local sub-problem.
"""

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn

from .dataclasses import ADMM_LayerConfig, ADMM_LayerState, TemporalCache
from .functional.utils import broadcast_to_match


class ADMM_ActivationBase(nn.Module, ABC):
    """Abstract Base Class for ADMM Activation Functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass applying the activation non-linearity.

        Args:
            x (torch.Tensor): The input pre-activation tensor.

        Returns:
            torch.Tensor: The activated tensor.
        """
        pass

    @abstractmethod
    def update_z(
        self, state: ADMM_LayerState, forward: torch.Tensor, config: ADMM_LayerConfig
    ) -> None:
        """Computes the vectorized update for the pre-activation tensor ($z$).

        Args:
            state (ADMM_LayerState): The state object containing the variables to update.
            forward (torch.Tensor): The precomputed spatial transformation.
            config (ADMM_LayerConfig): The local configuration containing penalty parameters.
        """
        pass

    def update_z_decoupled(
        self,
        state: ADMM_LayerState,
        forward: torch.Tensor,
        time_steps: List[int],
        config: ADMM_LayerConfig,
    ) -> None:
        """Applies the $z$ update in a decoupled manner.

        Orchestrates the sequence loop internally by utilizing a mock cache, bypassing
        the strict causal dependency of fully unrolled networks.

        Args:
            state (ADMM_LayerState): The layer's state object.
            forward (torch.Tensor): The precomputed spatial transformation.
            time_steps (list): The sequence of timesteps to process.
            config (ADMM_LayerConfig): The local layer configuration.
        """
        # The activation handles the mock cache creation and loop orchestration!
        mock_cache = SimpleNamespace(forward_pass=forward)
        for t in time_steps:
            self.update_z_unrolled(t, mock_cache, state, config)
        del mock_cache


####################################################################################################
# ACTIVATION FUNCTIONS
####################################################################################################


class ADMM_Identity(ADMM_ActivationBase):
    """Place-holder identity activation function for linear outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def update_z(
        self, state: ADMM_LayerState, forward: torch.Tensor, config: ADMM_LayerConfig
    ) -> None:
        state.z.copy_(forward)


class ADMM_ReLU(ADMM_ActivationBase):
    """ADMM implementation of the Rectified Linear Unit (ReLU)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    def update_z(
        self, state: ADMM_LayerState, forward: torch.Tensor, config: ADMM_LayerConfig
    ) -> None:
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
            state (ADMM_LayerState): The current ADMM state containing spatial variables.
            forward (torch.Tensor): The precomputed spatial transformation.
            config (ADMM_LayerConfig): The local configuration containing penalty parameters.
        """
        z = config.beta * state.a + config.rho * forward
        target_forward = forward
        if config.use_lagrange and state.lambda_lagrange is not None:
            lam = broadcast_to_match(state.lambda_lagrange, forward)
            z.sub_(lam)
            target_forward = forward - lam.div(config.rho)

        z.div_(config.beta + config.rho)
        state.z.copy_(torch.where(z > 0, z, target_forward))


class ADMM_Heaviside(ADMM_ActivationBase):
    r"""ADMM implementation of a Heaviside (Step) Function for Spiking Networks."""

    def __init__(self, thetas: float = 1.0):
        """Initializes the Heaviside activation.

        Args:
            thetas (float, optional): The spiking firing threshold. Defaults to 1.0.
        """
        super().__init__()
        self.thetas = thetas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.thetas).to(x.dtype)

    def check_entries(
        self,
        z_star: torch.Tensor,
        a: torch.Tensor,
        forward: torch.Tensor,
        z_minus_forward: Optional[torch.Tensor],
        config: ADMM_LayerConfig,
        is_vectorized: bool = False,
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
            z_star (torch.Tensor): The ideal unconstrained pre-activation update.
            a (torch.Tensor): The activation tensor.
            forward (torch.Tensor): The spatial forward pass tensor.
            z_minus_forward (torch.Tensor): The precomputed difference for the next timestep.
            config (ADMM_LayerConfig): The local layer configuration parameters.
            is_vectorized (bool, optional): Flag indicating if this is a full sequence update
                (applies the indicator function to zero out $\Delta_3$ at $t=T$). Defaults to False.

        Returns:
            torch.Tensor: The projected and bounded $z$ tensor.
        """
        delta1 = a.mul(-2.0).add_(1.0).mul_(config.beta)

        total_delta = z_star.sub(forward).pow_(2)
        tmp = forward.sub(config.thetas).pow_(2)
        total_delta.sub_(tmp).mul_(config.rho)  # total_delta is now delta2
        total_delta.add_(delta1)  # total_delta is now delta1 + delta2

        if z_minus_forward is not None:
            res_term = z_minus_forward.add(a, alpha=config.thetas)

            d3 = res_term.sub(config.deltas * z_star).pow_(2)
            d3_sub = res_term.sub(config.deltas * config.thetas).pow_(2)
            d3.sub_(d3_sub).mul_(config.rho)

            if is_vectorized:
                d3[-1].zero_()

            total_delta.add_(d3)  # total_delta is now delta1 + delta2 + delta3

        mask1 = (z_star > config.thetas).logical_and_(total_delta > 0)

        total_delta.add_(
            delta1, alpha=-2.0
        )  # total_delta is now delta2 + delta3 - delta1
        mask2 = (z_star <= config.thetas).logical_and_(total_delta > 0)

        z_star.masked_fill_(mask1, config.thetas)
        z_star.masked_fill_(mask2, config.thetas + 1e-5)
        return z_star

    def update_z_unrolled(
        self,
        t: int,
        cache: TemporalCache,
        state: ADMM_LayerState,
        config: ADMM_LayerConfig,
    ) -> None:
        r"""Unrolled version of the $z$ update.

        Formula evaluated:

        $$z_{l,t}^* = \frac{\mathcal{F}_{l,t} + \delta\big(z_{l,t+1} - F_l(a_{l-1,t+1}) + \theta a_{l,t}\big)\mathbb{1}_{\{t<T\}}}{1 + \delta^2 \mathbb{1}_{\{t<T\}}}$$

        You can find the corresponding vectorized version in the [activation_z_update][admm.activation_functions.ADMM_Heaviside.update_z] method.

        Args:
            t (int): Current timestep.
            cache (TemporalCache): Precomputed matrices.
            state (ADMM_LayerState): The layer state.
            config (ADMM_LayerConfig): Layer configuration.
        """
        T = state.z.size(0)
        temporal_forward = cache.forward_pass[t].clone()
        if t > 0:
            temporal_forward.add_(state.z[t - 1], alpha=config.deltas)
            temporal_forward.add_(state.a[t - 1], alpha=-config.thetas)

        if t == T - 1 and config.use_lagrange and state.lambda_lagrange is not None:
            lam = broadcast_to_match(state.lambda_lagrange, temporal_forward)
            temporal_forward.sub_(lam, alpha=1.0 / config.rho)

        if t < T - 1:
            z_minus_forward = state.z[t + 1].sub(cache.forward_pass[t + 1])
            if t == T - 2 and config.use_lagrange and state.lambda_lagrange is not None:
                lam = broadcast_to_match(state.lambda_lagrange, z_minus_forward)
                z_minus_forward.add_(lam, alpha=1.0 / config.rho)
        else:
            z_minus_forward = None
        a_t = state.a[t]

        if z_minus_forward is not None:  # Case t<T-1
            z_star = z_minus_forward.add(a_t, alpha=config.thetas)
            z_star.mul_(config.deltas).add_(temporal_forward)
            z_star.div_(1.0 + config.deltas**2)
        else:
            z_star = temporal_forward.clone()

        state.z[t].copy_(
            self.check_entries(
                z_star=z_star,
                a=a_t,
                forward=temporal_forward,
                z_minus_forward=z_minus_forward,
                config=config,
                is_vectorized=False,
            )
        )

    def update_z(
        self, state: ADMM_LayerState, forward: torch.Tensor, config: ADMM_LayerConfig
    ) -> None:
        r"""Vectorized version of the $z$ update.

        Formula evaluated:

        $$z_l^* = \frac{\mathcal{F}_l + \delta S^\top \big(z_l - F_l(a_{l-1}) + \theta a_l\big)}{\mathcal{I} + \delta^2 S^\top S}$$

        You can find the corresponding unrolled version in the [activation_z_unrolled][admm.activation_functions.ADMM_Heaviside.update_z_unrolled] method.

        Args:
            state (ADMM_LayerState): The layer state containing full sequence variables.
            forward (torch.Tensor): Precomputed forward pass.
            config (ADMM_LayerConfig): Layer configuration.
        """
        z = state.z
        a = state.a

        temporal_forward = forward.clone()
        temporal_forward[1:].add_(z[:-1], alpha=config.deltas)
        temporal_forward[1:].add_(a[:-1], alpha=-config.thetas)

        z_minus_forward_next = torch.zeros_like(z)
        z_minus_forward_next[:-1].copy_(z[1:]).sub_(forward[1:])

        if config.use_lagrange and state.lambda_lagrange is not None:
            lam_rho = broadcast_to_match(state.lambda_lagrange, state.z[-1]).div(
                config.rho
            )
            temporal_forward[-1].sub_(lam_rho)
            z_minus_forward_next[-2].add_(lam_rho)

        z_star = z_minus_forward_next.add(a, alpha=config.thetas)
        z_star.mul_(config.deltas).add_(temporal_forward)

        denominator = torch.ones_like(z_star)
        denominator[:-1].add_(config.deltas**2)
        z_star.div_(denominator)

        z_star[-1] = temporal_forward[-1]  # Applying the boundary condition at t=T

        state.z.copy_(
            self.check_entries(
                z_star=z_star,
                a=a,
                forward=temporal_forward,
                config=config,
                z_minus_forward=z_minus_forward_next,
                is_vectorized=True,
            )
        )
