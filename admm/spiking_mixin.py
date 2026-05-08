r"""This module provides temporal modeling and unrolled solvers for Spiking
Neural Networks (SNNs). It is designed to be used as a Mixin, overriding
standard spatial hooks with temporal dependencies (leakage, reset).
"""

from types import SimpleNamespace
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dataclasses import ADMMState, TemporalCache
from .solvers import solve_spiking_system, solve_woodbury_system
from .temporal_helpers import (
    compute_temporal_dependencies,
    fold_time,
    get_spiking_a_adjoint,
    get_spiking_a_denominator,
    get_spiking_v,
    unfold_time,
)

############################################################################################################
# Spiking Mixin
############################################################################################################


class ADMM_Spiking:
    """Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs)."""

    def _fold_time(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[tuple]]:
        """Delegates to temporal_helpers.fold_time."""
        return fold_time(x=x)

    def _unfold_time(
        self, x_flat: torch.Tensor, tb_shape: Optional[tuple]
    ) -> torch.Tensor:
        """Delegates to temporal_helpers.unfold_time."""
        return unfold_time(x_flat=x_flat, tb_shape=tb_shape)

    def _compute_temporal_dependencies(
        self, include_reset: bool = True
    ) -> torch.Tensor:
        """Delegates to temporal_helpers.compute_temporal_dependencies."""
        return compute_temporal_dependencies(
            z=self.z,
            a=self.a,
            deltas=self.config.deltas,
            thetas=self.config.thetas,
            use_reset=self.config.use_reset,
            include_reset=include_reset,
        )

    def _get_v(self, include_reset: bool = True) -> torch.Tensor:
        """Delegates to temporal_helpers.get_spiking_v."""
        return get_spiking_v(
            z=self.z,
            bias=self._format_bias(),
            temporal_dependencies=self._compute_temporal_dependencies(include_reset),
            rho=self.config.rho,
            lambda_lagrange=self.lambda_lagrange,
            broadcast_func=self._broadcast_to_match,
        )

    def _get_a_denominator(
        self,
        beta_current: float,
        rho_current: float,
        thetas_current: float,
        a_shape: tuple,
        unrolled: bool = False,
    ) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]:
        """Delegates to temporal_helpers.get_spiking_a_denominator."""
        temporal_penalty = rho_current * (thetas_current**2)
        W = self._get_expanded_weights(a_shape)
        out_features, in_features = W.shape

        if (
            getattr(self, "W", None) is not None
            and self.W.dim() == 2
            and in_features > out_features
        ):
            main_dict = {
                "W": W,
                "beta": beta_current + temporal_penalty,
                "rho": self.config.rho,
            }
            last_dict = {"W": W, "beta": beta_current, "rho": self.config.rho}
            return main_dict, last_dict, in_features

        WtW, in_features = self._get_WtW(a_shape=a_shape)

        return get_spiking_a_denominator(
            WtW=WtW,
            in_features=in_features,
            beta_current=beta_current,
            rho_next=self.config.rho,
            temporal_penalty=temporal_penalty,
            unrolled=unrolled,
        )

    def _get_a_adjoint(self, next_layer: nn.Module) -> torch.Tensor:
        """Delegates to temporal_helpers.get_spiking_a_adjoint."""
        return get_spiking_a_adjoint(layer=self, next_layer=next_layer)

    def update_lambda(self, a_prev: torch.Tensor) -> None:
        r"""Updates the Lagrange multiplier ($\lambda$) for the layer constraint.

        Formula evaluated:
        $\lambda_T^{k+1} = \lambda_T^k + \rho z_T - \rho \delta z_{T-1} - \rho \text{forward}(a_{prev})$

        Args:
            a_prev (torch.Tensor): The activations from the previous layer or inputs.
        """
        if not self.config.use_lagrange or self.lambda_lagrange is None:
            return

        z_T = self.z[-1]
        z_T_minus_1 = self.z[-2]
        forward = self.spatial_forward(a_prev[-1].unsqueeze(0)).squeeze(0)
        self.lambda_lagrange.add_(z_T, alpha=self.config.rho)
        self.lambda_lagrange.add_(
            z_T_minus_1, alpha=-self.config.rho * self.config.deltas
        )
        self.lambda_lagrange.add_(forward, alpha=-self.config.rho)

    def update_z_last(
        self, a_prev: torch.Tensor, labels: torch.Tensor, loss_f: Any = None
    ) -> None:
        """Delegates the vectorized sequential boundary update to the active loss function."""
        forward = self.spatial_forward(a_prev)
        temporal_forward = self.vectorized_forward(a_prev)
        shape = self.z[-1]
        labels = self._broadcast_to_match(labels, shape)
        if self.config.use_lagrange and self.lambda_lagrange is not None:
            lam = self._broadcast_to_match(self.lambda_lagrange, shape)
        else:
            lam = torch.zeros_like(shape)
        self.z.copy_(
            loss_f.update_z_last_spiking(
                forward=forward,
                temporal_forward=temporal_forward,
                labels=labels,
                lambda_lagrange=lam,
                z=self.z,
                rho=self.config.rho,
                deltas=self.config.deltas,
            )
        )

    def update_z_last_unrolled(
        self,
        a_prev: torch.Tensor,
        labels: torch.Tensor,
        time_steps: list,
        jacobi: bool = False,
        loss_f: Any = None,
    ) -> None:
        """Delegates the unrolled sequence boundary update to the active loss function."""
        labels = self._broadcast_to_match(labels, self.z[-1])
        if self.config.use_lagrange and self.lambda_lagrange is not None:
            lam = self._broadcast_to_match(self.lambda_lagrange, self.z[-1])
        else:
            lam = torch.zeros_like(self.z[-1])
        forward = self.spatial_forward(a_prev)
        self.z.copy_(
            loss_f.update_z_last_unrolled_spiking(
                forward=forward,
                labels=labels,
                lambda_lagrange=lam,
                z=self.z,
                rho=self.config.rho,
                deltas=self.config.deltas,
                time_steps=time_steps,
                jacobi=jacobi,
            )
        )

    def _create_cache(
        self, next_layer: nn.Module, a_prev: torch.Tensor
    ) -> TemporalCache:
        """Delegates cache construction to the TemporalCache factory.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            TemporalCache: A typed data class containing the precomputed matrices.
        """
        return TemporalCache.build(self, next_layer, a_prev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard sequential pass for initialization or inference.

        Simulates the spiking mechanics step-by-step over the time dimension.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the full temporal simulation.
        """
        y = self.spatial_forward(x)
        z = torch.zeros_like(y)
        z_prev = torch.zeros_like(y[0])
        a_prev = torch.zeros_like(y[0])
        T = y.size(0)
        for t in range(T):
            reset = (
                self.config.thetas * a_prev
                if (t > 0 and self.config.use_reset)
                else 0.0
            )
            z_t = y[t] + self.config.deltas * z_prev - reset
            z[t] = z_t
            z_prev = z_t
            a_prev = self.h(z_t)
        return z

    def vectorized_forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Vectorized ADMM pass for optimization and constraint evaluation.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            torch.Tensor: The evaluated constraints including temporal dependencies.
        """
        z = self.spatial_forward(a_prev) + self._compute_temporal_dependencies()
        return z

    def update_a(self, next_layer: nn.Module, a_prev: torch.Tensor) -> None:
        """Manages the vectorized activation ($a$) update for the sequence.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
        """
        # self.config.beta * self.h(self.z) + adjoint + temporal penalty
        numerator = self._get_a_adjoint(next_layer=next_layer)
        numerator.add_(self.h(self.z), alpha=self.config.beta)

        # Temporal penalty  =  -rho*thetas(z_t+1 -forward_t+1 -delta*z)
        num_slice = numerator[:-1]
        forward_pass = self.spatial_forward(a_prev)
        num_slice.sub_(forward_pass[1:], alpha=-self.config.thetas * self.config.rho)
        del forward_pass
        num_slice.add_(self.z[1:], alpha=-self.config.thetas * self.config.rho)
        num_slice.add_(
            self.z[:-1], alpha=self.config.deltas * self.config.thetas * self.config.rho
        )

        denominator_main, denominator_last, in_features = next_layer._get_a_denominator(
            a_shape=self.a.shape,
            beta_current=self.config.beta,
            rho_current=self.config.rho,
            thetas_current=self.config.thetas,
            unrolled=False,
        )

        new_a = next_layer._solve_activation_system(
            numerator=numerator,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            a_shape=self.a.shape,
            in_features=in_features,
        )

        new_a = torch.clamp(new_a, min=0.0, max=1.0)
        self.a.copy_(new_a)

    def _solve_activation_system(
        self,
        numerator: torch.Tensor,
        denominator_main: torch.Tensor,
        denominator_last: torch.Tensor,
        a_shape: tuple,
        in_features: int,
    ) -> torch.Tensor:
        """Routes the activation update to the temporal spiking linear algebra solver."""
        return solve_spiking_system(
            A_main=denominator_main,
            A_last=denominator_last,
            B=numerator,
            a_shape=a_shape,
            in_features=in_features,
            T=numerator.size(0),
        )

    def update_bias(self, a_prev: torch.Tensor) -> None:
        r"""Averages the residual errors across spatial and temporal dimensions.

        Formula evaluated: $b = \text{mean}(z - A(a_{prev}) - \text{temporal\_dependencies})$

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
        """
        in_mean = self.spatial_forward(a_prev, use_bias=False)
        in_mean.add_(self._compute_temporal_dependencies())
        in_mean.neg_().add_(self.z)

        if self.lambda_lagrange is not None:
            lambda_lagrange = self._broadcast_to_match(self.lambda_lagrange, self.z[-1])
            in_mean[-1].add_(lambda_lagrange, alpha=1.0 / self.config.rho)

        new_bias = torch.mean(in_mean, dim=self._get_bias_reduction_dims())
        self.b.copy_(new_bias)

    def update_az_interleaved(
        self,
        next_layer: nn.Module,
        a_prev: torch.Tensor,
        time_steps: list,
        update_z_first: bool,
    ) -> None:
        """Orchestrates the interleaved updates of $a$ and $z$ over time using caching.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            time_steps (list): The list of sequence time steps to update.
            update_z_first (bool): Flag determining order of local evaluation.
        """
        cache = self._create_cache(next_layer, a_prev)

        for t in time_steps:
            if update_z_first:
                self.update_z_unrolled(t, cache)
                self.update_a_unrolled(t, cache, next_layer)
            else:
                self.update_a_unrolled(t, cache, next_layer)
                self.update_z_unrolled(t, cache)

    def update_a_unrolled(
        self, t: int, cache: TemporalCache, next_layer: nn.Module
    ) -> None:
        """Manages the unrolled activation ($a$) update for a specific timestep.

        Utilizes precomputed terms from `TemporalCache`.

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices.
            next_layer (nn.Module): The subsequent layer in the network.
        """
        h_t = self.config.beta * self.h(self.z[t])
        temporal_penalty_numerator_t = 0.0
        if t < self.z.size(0) - 1:
            temporal_penalty_numerator_t = (
                -self.config.thetas
                * self.config.rho
                * (
                    self.z[t + 1]
                    - self.config.deltas * self.z[t]
                    - cache.forward_pass[t + 1]
                )
            )
        numerator = cache.adjoint[t] + h_t + temporal_penalty_numerator_t

        is_last = t == self.z.size(0) - 1
        denominator = cache.denominator_last if is_last else cache.denominator_main
        new_a_t = next_layer._solve_activation_system_unrolled(
            numerator=numerator, denominator=denominator
        )

        self.a[t].copy_(torch.clamp(new_a_t, min=0.0, max=1.0))

    def _solve_activation_system_unrolled(
        self, numerator: torch.Tensor, denominator: Union[torch.Tensor, dict]
    ) -> torch.Tensor:
        """Executes the unrolled step using dense matrix multiplication.

        Reshaping is aligned with `solve_linear_system` in `solvers.py`.
        """
        if isinstance(denominator, dict):
            return solve_woodbury_system(
                W=denominator["W"],
                B=numerator,
                beta=denominator["beta"],
                rho=denominator["rho"],
                a_shape=numerator.shape,
            )
        original_shape = numerator.shape
        in_features = denominator.size(1)
        numerator_flat = numerator.reshape(-1, in_features)
        a_t_flat = torch.matmul(numerator_flat, denominator)
        return a_t_flat.view(original_shape)

    def update_z_unrolled(
        self, t: int, cache: TemporalCache, z_to_use: Optional[torch.Tensor] = None
    ) -> None:
        r"""Performs the unrolled pre-activation ($z$) update for hidden layers.

        Defines the proximal update based on two main terms:
         1. Temporal Forward pass: ($\text{Forward pass} + \delta z_{prev} - \theta a_{prev}$)
         2. $z_{next} - \text{Forward Pass}$

        Applies the proximal operator to these terms via the activation function's
        specialized unrolled solver.

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices.
            z_to_use (torch.Tensor, optional): State override for Jacobi steps. Defaults to None.
        """
        z_to_use = z_to_use if z_to_use is not None else self.z
        T = self.z.size(0)
        temporal_forward = cache.forward_pass[t].clone()
        if t > 0:
            temporal_forward.add_(z_to_use[t - 1], alpha=self.config.deltas)
            temporal_forward.add_(self.a[t - 1], alpha=-self.config.thetas)

        if t == T - 1 and self.config.use_lagrange and self.lambda_lagrange is not None:
            lam = self._broadcast_to_match(self.lambda_lagrange, temporal_forward)
            temporal_forward.sub_(lam, alpha=1.0 / self.config.rho)

        if t < T - 1:
            z_minus_forward = z_to_use[t + 1].sub(cache.forward_pass[t + 1])
        else:
            z_minus_forward = None
        current_state = ADMMState(
            forward=temporal_forward, z_minus_forward=z_minus_forward, a=self.a[t]
        )
        new_z_t = self.h.activation_z_unrolled(current_state)
        self.z[t].copy_(new_z_t)

    def update_z_decoupled(self, a_prev: torch.Tensor, time_steps: list) -> None:
        """Decoupled causal sweep for the $z$ update.

        Reuses the unrolled logic across all timesteps without redefining the math by
        creating a lightweight mock cache.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            time_steps (list): The list of sequence time steps to update.
        """
        forward_pass = self.spatial_forward(a_prev)
        mock_cache = SimpleNamespace(forward_pass=forward_pass)
        for t in time_steps:
            self.update_z_unrolled(t, mock_cache)
        del forward_pass
        del mock_cache
