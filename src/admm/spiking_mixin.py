"""This module provides temporal modeling and unrolled solvers for Spiking
Neural Networks (SNNs). It is designed to be used as a Mixin, overriding
standard spatial hooks with temporal dependencies (leakage, reset).
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataclasses import ADMM_LayerConfig, ADMM_LayerState, TemporalCache
from .functional.a_update_solvers import (
    solve_fft_system_spiking,
    solve_fft_system_unrolled,
    solve_standard_system_spiking,
    solve_standard_system_unrolled,
    solve_woodbury_system_spiking,
    solve_woodbury_system_unrolled,
)
from .functional.utils import (
    broadcast_to_match,
    compute_temporal_dependencies,
)

############################################################################################################
# Spiking Mixin
############################################################################################################


class ADMM_Spiking:
    """Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs)."""

    def forward(
        self, a_prev: torch.Tensor, z_tminus1: torch.Tensor, a_tminus1: torch.tensor
    ) -> torch.Tensor:
        r"""Standard sequential pass for initialization or inference.

        * For Static Networks: Simply returns the spatial transformation ($z_l = F_l(a_{l-1})$).
        * For Spiking Networks (SNNs): Simulates the mechanics step-by-step.

        Note:
            To refer to its non-spiking counterpart, see [`forward`][src.admm.base_layer.ADMM_Layer.forward].

            **Performance Note (Compute vs. Memory):**
            This method is **compute-bound**. It strictly enforces causality via an $O(T)$
            sequential loop, resulting in slow execution but minimal memory footprint.
            Use this strictly for inference.

        Args:
            a_prev (torch.Tensor): The input activations from the previous layer at the current timestep.
            z_tminus1 (torch.Tensor): The pre-activations (membrane potential) from the previous timestep.
            a_tminus1 (torch.Tensor): The activations (spikes) from the previous timestep.

        Returns:
            torch.Tensor: The output tensor after the full temporal simulation.
        """
        y_t = self.spatial_forward(a_prev)
        if z_tminus1 is None:
            z_tminus1 = torch.zeros_like(y_t)
            a_tminus1 = torch.zeros_like(y_t)

        reset = self.config.thetas * a_tminus1 if (self.config.use_reset) else 0.0
        return y_t + self.config.deltas * z_tminus1 - reset

    def get_v(self, state: ADMM_LayerState) -> torch.Tensor:
        r"""Returns the target tensor $v$ for spiking layers.

        Formula evaluated:
        $v_l = z_l - b_l - T_l$

        If layer $l$ has a Lagrangian multiplier:
        $v_l = z_l - b_l - T_l + \frac{\lambda_l}{\rho}\mathbb{1}_{\{t=T\}}$

        Note:
            You can find the corresponding non-spiking version [`get_v`][src.admm.affine_layer.ADMM_AffineLayer.get_v] in the
            [`Affine Layer`][src.admm.affine_layer] module.
        Args:
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the current layer.

        Returns:
            torch.Tensor: The computed target tensor $v$.
        """
        v = state.z.clone()
        bias_formatted = self._format_bias(state)
        if isinstance(bias_formatted, torch.Tensor):
            v.sub_(bias_formatted)
        v.sub_(compute_temporal_dependencies(state, self.config))
        if self.config.use_lagrange and state.lambda_lagrange is not None:
            lambda_lagrange = broadcast_to_match(state.lambda_lagrange, v)
            v.add_(lambda_lagrange, alpha=1.0 / self.config.rho)
        return v

    def _compute_bias_covariance(
        self, state: ADMM_LayerState, a_prev: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes the mean spatial covariance for the bias update step.

        Formula evaluated:
        $z_l - \mathcal{A}_l(a_{l-1}) - T_l$

        If layer $l$ has a Lagrangian multiplier:
        $z_l - \mathcal{A}_l(a_{l-1}) - T_l + \frac{\lambda}{\rho}_\{{t=T}\}$

        Note:
            You can find the corresponding non-spiking version [`_compute_bias_covariance`][src.admm.affine_layer.ADMM_AffineLayer._compute_bias_covariance] in the
            [`admm.affine_layer`][src.admm.affine_layer] module.

        Args:
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the current layer.
            a_prev (torch.Tensor): The activations from the previous layer.

        Returns:
            torch.Tensor: The intermediate covariance tensor used to calculate the bias sum.
        """
        in_mean = self.spatial_forward(a_prev, use_bias=False)
        in_mean.add_(compute_temporal_dependencies(state, self.config))
        in_mean.neg_().add_(
            state.z
        )  # We perform .neg_() trick so we can save memory allocation

        if self.config.use_lagrange and state.lambda_lagrange is not None:
            lam_spatial = broadcast_to_match(state.lambda_lagrange, state.z)
            in_mean[-1].add_(lam_spatial[-1], alpha=1.0 / self.config.rho)
        return in_mean

    def _get_a_numerator_no_adjoint(
        self,
        state: ADMM_LayerState,
        next_config: ADMM_LayerConfig,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculates the spiking numerator block for the vectorized activation ($a$) update.

        Extends the static formula with the temporal reset penalty term for $t < T$:
        $N = \beta_l h_{l,\theta}(z_{l}) - \rho_l \theta S^T\big(z_l - \delta S z_l-F_l(a_{l-1}) \big)$

         Args:
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] containing the current pre-activations ($z$).
            next_config (ADMM_LayerConfig): [The configuration object][src.admm.dataclasses.ADMM_LayerConfig] of the subsequent layer.
            a_prev (torch.Tensor): The activations from the previous layer.

        Returns:
            torch.Tensor: The computed partial numerator tensor.
        """
        numerator = self.h(state.z).clone().mul_(self.config.beta)

        # Temporal penalty
        # ρ_l·θ·S^T(F_l(a_{l-1})) - ρ_l·θ·S^T(z_l) + ρ_l·θ·δ·S^T(S(z_l))
        # S^T (pulls from t+1 to t). It is only valid for t < T.
        num_slice = numerator[:-1]
        forward_pass = self.spatial_forward(a_prev)
        # S^T ([1:] means t+1 to t)
        num_slice.add_(forward_pass[1:], alpha=self.config.thetas * next_config.rho)
        del forward_pass
        # S^T ([1:] means t+1 to t)
        num_slice.add_(state.z[1:], alpha=-self.config.thetas * next_config.rho)
        # The advance (S^T) and delay (S) operators cancel out (S^T * S = I).
        # We simply add the current timestep (t, via [:-1]).
        num_slice.add_(
            state.z[:-1],
            alpha=self.config.deltas * self.config.thetas * next_config.rho,
        )
        return numerator

    def _get_fft_a_denominator(
        self, config_prev: ADMM_LayerConfig, a_shape: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the denominator matrices for the activation ($a$) update using FFTs.

        Formula evaluated in the frequency domain:
        $D = \beta_l I + \rho_{l+1} \mathcal{F}(W_{l+1})^* \mathcal{F}(W_{l+1}) + \rho_l \theta^2 S^TS$

        Note:
            We separate this function, because it is needed for the unrolled cache, see [`TemporalCache`][src.admm.dataclasses.TemporalCache]

        Args:
            config_prev (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] of the previous layer.
            a_shape (tuple): The physical geometry shape of the activation tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The main denominator matrix (for $t < T$).
                - The final denominator matrix (for $t = T$).
        """
        # This function is generally called using next_layer._get_fft_a_denominator.
        # Then the layer indices are shifted.
        temporal_penalty = config_prev.rho * (config_prev.thetas**2)

        height, width = a_shape[-2:]
        _, _, kernel_h, kernel_w = self.W.shape

        pad_h = height - kernel_h
        pad_w = width - kernel_w
        padded_kernel = F.pad(self.W, (0, pad_w, 0, pad_h))

        kernel_fft = torch.fft.fft2(padded_kernel)
        kernel_fft = kernel_fft.permute(2, 3, 0, 1)  # Shape: [H, W, C_out, C_in]

        WtW_fft = torch.matmul(kernel_fft.conj().transpose(-2, -1), kernel_fft)
        denominator_last = WtW_fft * self.config.rho
        denominator_last.diagonal(dim1=-2, dim2=-1).add_(config_prev.beta)
        # Last time step does not have temporal reset (S^TS)
        denominator_main = denominator_last.clone()
        denominator_main.diagonal(dim1=-2, dim2=-1).add_(temporal_penalty)

        return denominator_main, denominator_last

    def _get_a_denominator(
        self,
        config_prev: ADMM_LayerConfig,
        W_expanded: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]:
        r"""Computes the spiking denominator matrix for the activation ($a$) update.

        Formula evaluated:
        $D = \beta_l I + \rho_{l+1} \mathcal{A}_{l+1}^* \circ \mathcal{A}_{l+1} + \rho_l \theta^2 S^TS}$

        Note:
            We separate this function, because it is needed for the unrolled cache, see [`TemporalCache`][src.admm.dataclasses.TemporalCache]

        Args:
            config_prev (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] of the previous layer.
            W_expanded (torch.Tensor): The expanded weight matrix of the next layer.

        Returns:
            Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]: A tuple containing:
                - The main denominator matrix (for $t < T$).
                - The final denominator matrix (for $t = T$).
                - The tracked input features count.
        """
        # This function is generally called using nextlayer._get_a_denominator.
        # Then the layer indices are shifted.
        temporal_penalty = config_prev.rho * (config_prev.thetas**2)
        in_features = W_expanded.size(1)
        WtW = torch.matmul(W_expanded.t(), W_expanded)
        denominator_last = WtW * self.config.rho

        denominator_last.diagonal().add_(config_prev.beta)
        # Last time step does not have temporal reset (S^TS)
        denominator_main = denominator_last.clone()
        denominator_main.diagonal().add_(temporal_penalty)
        return denominator_main, denominator_last, in_features

    def _solve_fft_system(
        self,
        numerator: torch.Tensor,
        next_layer: nn.Module,
        a_shape: tuple,
    ) -> torch.Tensor:
        r"""Solves the activation system via the Fast Fourier Transform (FFT) for spiking layers.

        Note:
            You can find the corresponding non-spiking version [`_solve_fft_system`][src.admm.affine_layer.ADMM_AffineLayer._solve_fft_system] in the
            [`Affine Layer`][src.admm.affine_layer] module.

        Args:
            numerator (torch.Tensor): The assembled numerator target tensor.
            next_layer (nn.Module): The subsequent convolutional layer in the network.
            a_shape (tuple): The desired output shape of the activation tensor.

        Returns:
            torch.Tensor: The solved activations evaluated in the frequency domain, clamped to $[0, 1]$.
        """
        denominator_main, denominator_last = next_layer._get_fft_a_denominator(
            config_prev=self.config, a_shape=a_shape
        )

        return solve_fft_system_spiking(
            numerator=numerator,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
        )

    def _solve_woodbury_system(
        self,
        next_layer: nn.Module,
        W_expanded: torch.Tensor,
        numerator: torch.Tensor,
        a_shape: tuple,
    ) -> torch.Tensor:
        r"""Solves the activation system via the Woodbury Matrix Identity, calling [`solve_woodbury_system_static`][src.admm.functional.a_update_solvers.solve_woodbury_system_static].

        Preferred when the number of output features ($M$) is much smaller than the
        number of input features ($N$), meaning $M \ll N$. Inverting the small
        $M \times M$ matrix is significantly more efficient than solving the dense
        $N \times N$ system.

        Note:
            You can find the corresponding non-spiking version [`_solve_woodbury_system`][src.admm.affine_layer.ADMM_AffineLayer._solve_woodbury_system] in the
            [`Affine Layer`][src.admm.affine_layer] module.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            W_expanded (torch.Tensor): The expanded weight matrix of the next layer.
            numerator (torch.Tensor): The assembled numerator target tensor.
            a_shape (tuple): The desired output shape of the activation tensor.

        Returns:
            torch.Tensor: The solved spatial activations.
        """
        temporal_penalty = self.config.thetas * self.config.rho
        return solve_woodbury_system_spiking(
            W_expanded=W_expanded,
            numerator=numerator,
            beta=self.config.beta,
            rho=next_layer.config.rho,
            temporal_penalty=temporal_penalty,
            a_shape=a_shape,
            T=numerator.size(0),
        )

    def _solve_standard_system(
        self, next_layer, W_expanded, numerator, a_shape
    ) -> torch.Tensor:
        """Solves the standard dense system for spiking activations.

        Note:
            You can find the corresponding non-spiking version [`_solve_standard_system`][src.admm.affine_layer.ADMM_AffineLayer._solve_standard_system] in the
            [`Affine Layer`][src.admm.affine_layer] module.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            W_expanded (torch.Tensor): The expanded weight matrix of the next layer.
            numerator (torch.Tensor): The assembled numerator target tensor.
            a_shape (tuple): The state object of the current layer.

        Returns:
            torch.Tensor: The solved spiking activations.
        """
        denominator_main, denominator_last, in_features = next_layer._get_a_denominator(
            config_prev=self.config, W_expanded=W_expanded
        )
        return solve_standard_system_spiking(
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            numerator=numerator,
            a_shape=a_shape,
            in_features=in_features,
            T=numerator.size(0),
        )

    def update_a_unrolled(
        self,
        t: int,
        cache: TemporalCache,
        next_layer: nn.Module,
        state: ADMM_LayerState,
    ) -> None:
        r"""Manages the unrolled (Gauss-Seidel) activation ($a$) update for spiking layers.

        Assembles the per-timestep numerator, factoring in the temporal reset penalty
        if $t < T$, and dynamically routes to the optimal linear solver (FFT, Woodbury,
        or Standard) based on the layer configurations. Output is clamped to $[0, 1]$.

        Args:
            t (int): The current timestep index being evaluated.
            cache (TemporalCache): Cached states and matrices for unrolled optimization.
            next_layer (nn.Module): The subsequent layer in the network.
            state (ADMM_LayerState): The state object of the current layer.
        """
        is_last = t == state.z.size(0) - 1

        # β_l·h_l(z_{l,t})
        h_t = self.config.beta * self.h(state.z[t])

        # - ρ_l·θ·(z_{l,t+1} - δ·z_{l,t} - F_l(a_{l-1,t}))
        temporal_reset_numerator_t = 0.0
        if not is_last:
            temporal_reset_numerator_t = (
                -self.config.thetas
                * self.config.rho
                * (
                    state.z[t + 1]
                    - self.config.deltas * state.z[t]
                    - cache.forward_pass[t + 1]
                )
            )

        numerator = cache.adjoint[t] + h_t + temporal_reset_numerator_t
        # Step 2: Solve  (β_l I + ρ_{l+1} A_{l+1}*A_{l+1}) a_l = numerator
        # --- FFT path ---
        denominator = cache.denominator_last if is_last else cache.denominator_main
        if cache.route == "fft":
            new_a_t = solve_fft_system_unrolled(
                numerator=numerator, denominator=denominator
            )

        # --- Woodbury path ---
        elif cache.route == "woodbury":
            beta_effective = (
                self.config.beta
                if is_last
                else (self.config.beta + self.config.rho * (self.config.thetas**2))
            )
            new_a_t = solve_woodbury_system_unrolled(
                numerator=numerator,
                W_expanded=cache.W_expanded,
                inv_denominator=denominator,
                beta_effective=beta_effective,
                rho=next_layer.config.rho,
                a_shape=state.a[t].shape,
            )

        # --- Standard dense path ---
        elif cache.route == "standard":
            new_a_t = solve_standard_system_unrolled(
                numerator=numerator, denominator=denominator
            )

        else:
            raise ValueError(f"Unknown cache route: {cache.route}")

        state.a[t].copy_(torch.clamp(new_a_t, min=0.0, max=1.0))

    def update_lambda(self, state: ADMM_LayerState, a_prev: torch.Tensor) -> None:
        r"""Updates the Lagrange multiplier ($\lambda$) based on the current layer constraints for spiking neurons.

        Formula evaluated:
        $\lambda_l \leftarrow \lambda_l + \rho_l \big(z_{l,T} - \mathcal{F}_{l,T}\big)$

        Note:
            You can find the corresponding non-spiking version [update_lambda][src.admm.base_layer.ADMM_Layer.update_lambda] in the [Base Layer][src.admm.base_layer] module.

        Args:
            state (ADMM_LayerState): The state object of the current layer.
            a_prev (torch.Tensor): The activations from the previous layer.
        """
        if not self.config.use_lagrange or state.lambda_lagrange is None:
            return

        z_T = state.z[-1]
        z_T_minus_1 = state.z[-2]
        forward = self.spatial_forward(a_prev[-1].unsqueeze(0)).squeeze(0)
        state.lambda_lagrange.add_(z_T, alpha=self.config.rho)
        state.lambda_lagrange.add_(
            z_T_minus_1, alpha=-self.config.rho * self.config.deltas
        )
        state.lambda_lagrange.add_(forward, alpha=-self.config.rho)
