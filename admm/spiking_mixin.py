r"""This module provides temporal modeling and unrolled solvers for Spiking
Neural Networks (SNNs). It is designed to be used as a Mixin, overriding
standard spatial hooks with temporal dependencies (leakage, reset).
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from .dataclasses import ADMM_LayerState, TemporalCache
from .functional.fft_convolution import (
    _solve_activation_system as fft_solve_activation_system,
)
from .functional.fft_convolution import (
    _solve_activation_system_unrolled as fft_solve_activation_system_unrolled,
)
from .functional.fft_convolution import (
    get_a_denominator as fft_get_a_denominator,
)
from .functional.utils import broadcast_to_match, compute_temporal_dependencies
from .solvers import solve_spiking_system, solve_woodbury_system_spiking

############################################################################################################
# Spiking Mixin
############################################################################################################


class ADMM_Spiking:
    """Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs)."""

    def _solve_activation_system(
        self,
        numerator: torch.Tensor,
        denominator_main: torch.Tensor,
        denominator_last: torch.Tensor,
        a_shape: tuple,
        in_features: int,
    ) -> torch.Tensor:
        """Routes the vectorized activation update to the temporal spiking linear algebra solver.

        Solves the boundary-aware system in two passes: one for the main timesteps
        ($t < T$, which carry the extra $\\rho \\theta^2$ temporal penalty on the diagonal)
        and one for the final timestep ($t = T$, where no future reset exists).

        Refer to its non-spiking counterpart, [_solve_activation_system][admm.affine.ADMM_AffineLayer._solve_activation_system],
        in the [Affine][admm.affine] module.

        Args:
            numerator (torch.Tensor): The precomputed numerator tensor $u$ of shape `[T, Batch, ...]`.
            denominator_main (torch.Tensor): The LHS matrix $D_{main} = \\beta I + \\rho W^T W + \\rho \\theta^2 I$ for $t < T$.
            denominator_last (torch.Tensor): The LHS matrix $D_{last} = \\beta I + \\rho W^T W$ for $t = T$.
            a_shape (tuple): The physical shape of the full activation tensor.
            in_features (int): Number of input features.

        Returns:
            torch.Tensor: The solved activations across all timesteps, clamped to $[0, 1]$.
        """
        return solve_spiking_system(
            A_main=denominator_main,
            A_last=denominator_last,
            B=numerator,
            a_shape=a_shape,
            in_features=in_features,
            T=numerator.size(0),
        )

    def _solve_woodbury_system(
        self,
        numerator: torch.Tensor,
        W_expanded: torch.Tensor,
        a_shape: tuple,
        beta_effective: float,
    ) -> torch.Tensor:
        r"""Solves the activation system via the Woodbury Matrix Identity for spiking layers.

        Identical in structure to the static Woodbury solver, but accepts a caller-supplied
        `beta_effective` that absorbs the temporal penalty $\\rho \\theta^2$ for $t < T$:

        * $\\beta_{eff} = \\beta + \\rho \\theta^2$ for main timesteps ($t < T$)
        * $\\beta_{eff} = \\beta$ for the final timestep ($t = T$)

        Formula evaluated (Woodbury simplified):
        $a = \\frac{1}{\\beta_{eff}} u - \\frac{\\rho}{\\beta_{eff}} W^T (\\beta_{eff} I_M + \\rho W W^T)^{-1} W u$

        Refer to its non-spiking counterpart, [_solve_woodbury_system][admm.affine.ADMM_AffineLayer._solve_woodbury_system],
        in the [Affine][admm.affine] module.

        Args:
            numerator (torch.Tensor): The precomputed numerator tensor $u$.
            W_expanded (torch.Tensor): The expanded weight matrix $W$ of shape `[out_features, in_features]`.
            a_shape (tuple): The desired output shape of the activation tensor.
            beta_effective (float): The effective $\\beta$ value, optionally including the temporal penalty.

        Returns:
            torch.Tensor: The solved activations $a$, reshaped to `a_shape`.
        """
        return solve_woodbury_system_spiking(
            W=W_expanded,
            B=numerator,
            beta=beta_effective,
            rho=self.config.rho,
            a_shape=a_shape,
        )

    def _solve_activation_system_unrolled(
        self, numerator: torch.Tensor, denominator: Union[torch.Tensor, dict]
    ) -> torch.Tensor:
        """Executes the unrolled step using dense matrix multiplication.

        Reshaping is aligned with `solve_linear_system` in `solvers.py`.

        Args:
            numerator (torch.Tensor): The numerator for a single timestep $t$.
            denominator (torch.Tensor): The pre-inverted matrix $D^{-1}$, already transposed
                to match the row-vector convention used by the dense solver.

        Returns:
            torch.Tensor: The solved activation for timestep $t$.
        """
        original_shape = numerator.shape
        in_features = denominator.size(1)
        numerator_flat = numerator.reshape(-1, in_features)
        a_t_flat = torch.matmul(numerator_flat, denominator)
        return a_t_flat.view(original_shape)

    def _compute_bias_covariance(self, state: ADMM_LayerState, a_prev: torch.Tensor):
        in_mean = self.spatial_forward(a_prev, use_bias=False)
        in_mean.add_(compute_temporal_dependencies(state, self.config))
        in_mean.neg_().add_(state.z)

        if self.config.use_lagrange and state.lambda_lagrange is not None:
            lam_spatial = broadcast_to_match(state.lambda_lagrange, state.z)
            in_mean[-1].add_(lam_spatial[-1], alpha=1.0 / self.config.rho)
        return in_mean

    def forward(
        self, a_prev: torch.Tensor, z_tminus1: torch.Tensor, a_tminus1: torch.tensor
    ) -> torch.Tensor:
        r"""Standard sequential pass for initialization or inference.

        * For Static Networks: Simply returns the spatial transformation ($z_l = F_l(a_{l-1})$).
        * For Spiking Networks (SNNs): Simulates the mechanics step-by-step.

        To refer to its non-spiking counterpart, see [forward][admm.core.ADMM_Layer.forward]. For the vectorized optimization pass, see [vectorized_forward][admm.spiking_mixin.ADMM_Spiking.vectorized_forward].

        > **Performance Note (Compute vs. Memory):**
        > This method is **compute-bound**. It strictly enforces causality via an $O(T)$
        > sequential loop, resulting in slow execution but minimal memory footprint.
        > Use this strictly for inference.

        Args:
            a_prev (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the full temporal simulation.
        """
        y_t = self.spatial_forward(a_prev)
        if z_tminus1 is None:
            z_tminus1 = torch.zeros_like(y_t)
            a_tminus1 = torch.zeros_like(y_t)

        reset = self.config.thetas * a_tminus1 if (self.config.use_reset) else 0.0
        return y_t + self.config.deltas * z_tminus1 - reset

    def get_v(self, state: ADMM_LayerState, include_reset: bool = True) -> torch.Tensor:
        r"""Returns the target tensor $v$ for spiking layers.

         Formula evaluated:

        * $v_l = z_l - b_l -T_l$
        * $v_l = z_l - b_l -T_l + \frac{\lambda_l}{\rho}$ If layer l has a Lagrangian multiplier

        You can find the corresponding non-spiking version [get_v][admm.affine.ADMM_AffineLayer.get_v] in the [Affine][admm.affine] module.
        This function is being called by the [get_v][admm.spiking_mixin.ADMM_Spiking.get_v] method in the [Spiking Mixin][admm.spiking_mixin] module, which overrides the non-spiking version to incorporate temporal dependencies.

        Args:
            z (torch.Tensor): The membrane potential tensor.
            bias (torch.Tensor or int): The reshaped bias tensor (or 0 if unused).
            temporal_dependencies (torch.Tensor): The precomputed leakage/reset tensor.
            rho (float): The spatial affine penalty parameter.
            lambda_lagrange (torch.Tensor, optional): The dual variable tensor. Defaults to None.
            broadcast_func (callable, optional): Helper function to align dimensions. Defaults to None.


        Returns:
            torch.Tensor: The computed target tensor $v$.
        """
        v = state.z.clone()
        bias = self._format_bias(state)
        if isinstance(bias, torch.Tensor):
            v.sub_(bias)
        v.sub_(compute_temporal_dependencies(state, self.config, include_reset))
        if state.lambda_lagrange is not None:
            lam_sp = broadcast_to_match(state.lambda_lagrange, state.z[-1])
            v[-1].add_(lam_sp, alpha=1.0 / self.config.rho)
        return v

    def get_a_numerator_no_adjoint(
        self,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculates the spiking numerator block for the vectorized activation ($a$) update.

        Extends the static formula with the temporal reset penalty term for $t < T$:

        $u_t = \beta h_{\theta}(z_t) + \rho \theta \big(F_l(a_{l-1,t+1}) - z_{t+1} + \delta z_t\big) \cdot \mathbb{1}_{t<T}$

        Args:
            state (ADMM_LayerState): The state of the current layer.
            a_prev (torch.Tensor): The previous layer's activations of shape `[T, Batch, ...]`.

        Returns:
            torch.Tensor: The partial numerator tensor (adjoint term not yet added).
        """
        numerator = self.h(state.z).clone().mul_(self.config.beta)

        # Temporal penalty  =  -rho*thetas(z_t+1 -forward_t+1 -delta*z_t)
        num_slice = numerator[:-1]
        forward_pass = self.spatial_forward(a_prev)
        num_slice.add_(forward_pass[1:], alpha=self.config.thetas * self.config.rho)
        del forward_pass
        num_slice.add_(state.z[1:], alpha=-self.config.thetas * self.config.rho)
        num_slice.add_(
            state.z[:-1],
            alpha=self.config.deltas * self.config.thetas * self.config.rho,
        )
        return numerator

    def get_a_denominator(
        self,
        beta_current: float,
        rho_current: float,
        thetas_current: float,
        a_shape: tuple,
        unrolled: bool = False,
    ) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]:
        r"""Computes the spiking denominator matrix for the activation ($a$) update.

        Formula evaluated: $D = \beta_l I + \rho_{l+1} \mathcal{A}_{l+1}^* \circ \mathcal{A}_{l+1} + \rho \theta^2 I\mathbb{1}_{t<T}$

        You can find the corresponding non-spiking version [get_a_denominator][admm.affine.ADMM_AffineLayer.get_a_denominator] in the [Affine][admm.affine] module.
        This function is being called by the [get_a_denominator][admm.spiking_mixin.ADMM_Spiking.get_a_denominator] method in the [Spiking Mixin][admm.spiking_mixin] module, which overrides the non-spiking version to incorporate temporal dependencies.

        Args:
            WtW (torch.Tensor): The computed $W^T W$ covariance matrix of the next layer.
            in_features (int): The number of input features.
            beta_current (float): The $\beta$ penalty parameter of the current layer.
            rho_next (float): The $\rho$ penalty parameter of the next layer.
            temporal_penalty (float): Precomputed penalty constraint ($\rho * \theta^2$).
            unrolled (bool, optional): If True, computes and transposes the inverse directly. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: The main denominator matrix (for $t < T$).
                - torch.Tensor: The final denominator matrix (for $t = T$).
                - int: The tracked input features count.
        """
        temporal_penalty = rho_current * (thetas_current**2)
        W = self._get_expanded_weights(a_shape)
        out_features, in_features = W.shape

        in_features = W.size(1)
        WtW = torch.matmul(W.t(), W)
        denominator_last = WtW * self.config.rho

        if denominator_last.dim() > 2:
            denominator_last.diagonal(dim1=-2, dim2=-1).add_(beta_current)
        else:
            denominator_last.diagonal().add_(beta_current)

        denominator_main = denominator_last.clone()
        if denominator_main.dim() > 2:
            denominator_main.diagonal(dim1=-2, dim2=-1).add_(temporal_penalty)
        else:
            denominator_main.diagonal().add_(temporal_penalty)

        if unrolled:
            denominator_main = torch.linalg.inv(denominator_main).transpose(-2, -1)
            denominator_last = torch.linalg.inv(denominator_last).transpose(-2, -1)

        return denominator_main, denominator_last, in_features

    def update_a(
        self,
        next_layer: nn.Module,
        next_state: ADMM_LayerState,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
    ) -> None:
        r"""Manages the vectorized (Jacobi) activation ($a$) update for spiking layers.

        Assembles the full temporal numerator and routes to the appropriate solver.
        Because $a_t$ does not depend on $a_{t-1}$, the entire sequence $a_{1:T}$ can
        be solved simultaneously in a single vectorized pass (Jacobi scheme).

        The three solver paths mirror the static `update_a`, with spiking-specific
        adjustments to the denominator ($+\\rho \\theta^2$ for $t < T$):

        * **FFT path**: Both layers are convolutional and `use_fft` is enabled. The temporal
          boundary is handled by passing two separate FFT denominators: `denominator_main`
          for $t < T$ and `denominator_last` for $t = T$.
        * **Woodbury path**: Output dimension $M \\ll$ input dimension $N$. The temporal
          penalty is absorbed into `beta_effective` before calling the solver.
        * **Standard path**: Default dense solver. Two distinct Gram matrices are built and
          dispatched to the spiking linear system solver.

        Refer to its non-spiking counterpart, [update_a][admm.affine.ADMM_AffineLayer.update_a],
        in the [Affine][admm.affine] module, and its unrolled (Gauss-Seidel) counterpart,
        [update_a_unrolled][admm.spiking_mixin.ADMM_Spiking.update_a_unrolled].

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            next_state (ADMM_LayerState): The state of the subsequent layer.
            state (ADMM_LayerState): The state of the current layer.
            a_prev (torch.Tensor): The previous layer's activations of shape `[T, Batch, ...]`.
        """
        # ------------------------------------------------------------------
        # Step 1: Build the numerator  u = β·h(z_l) + ρ·A*_{l+1}(v_{l+1})
        #         The spiking override of get_a_numerator_no_adjoint also adds
        #         the temporal reset penalty: -ρθ(z_{t+1} - F_{t+1} - δz_t)
        # ------------------------------------------------------------------
        numerator = self.get_a_numerator_no_adjoint(state=state, a_prev=a_prev)

        inside_adjoint = next_layer.get_v(next_state, include_reset=False)
        adjoint = next_layer.adjoint_operator(
            inside_adjoint, original_input_shape=state.a.shape
        )
        numerator.add_(adjoint, alpha=self.config.rho)

        # ------------------------------------------------------------------
        # Step 2: Expand weights (handles Linear, Conv, pooled variants)
        # ------------------------------------------------------------------
        W_expanded = next_layer.pool_op.expand_weights(
            W=next_layer.W, a_shape=state.a.shape
        )

        # Temporal penalty scalar  ρθ², added to the diagonal for t < T
        temporal_penalty = self.config.rho * (self.config.thetas**2)

        # ------------------------------------------------------------------
        # Step 3: Solve  (β I + ρ A*A [+ ρθ²I for t<T]) a = u
        # ------------------------------------------------------------------

        # --- FFT path (convolutional layers only) ---
        # The FFT denominator is built in the frequency domain; the two boundary
        # matrices (main / last) are passed separately to the FFT solver so it can
        # handle the t<T vs t=T split internally.
        # --- FFT path (convolutional layers only) ---
        # The FFT denominator is built in the frequency domain; the two boundary
        # matrices (main / last) are passed separately to the FFT solver so it can
        # handle the t<T vs t=T split internally.
        if (
            self.config.use_fft
            and getattr(self, "convolution", False)
            and getattr(next_layer, "convolution", False)
        ):
            # This returns the base FFT matrix: D = ρ W^H W + β I
            denominator_last, _, in_features = fft_get_a_denominator(
                next_layer,
                config_prev=self.config,
                W_expanded=W_expanded,
                a_shape=state.a.shape,
            )

            # For t < T, mathematically add the temporal penalty to the diagonals
            denominator_main = denominator_last.clone()
            denominator_main.diagonal(dim1=-2, dim2=-1).add_(temporal_penalty)

            new_a = fft_solve_activation_system(
                next_layer,
                numerator=numerator,
                denominator_main=denominator_main,
                denominator_last=denominator_last,
                a_shape=state.a.shape,
                in_features=in_features,
            )

        # --- Woodbury path (wide input layers: M << N) ---
        # The temporal penalty is absorbed into beta_effective per boundary condition,
        # then two separate Woodbury solves are concatenated along the time dimension.
        elif next_layer._should_use_woodbury(state.a.shape):
            beta_main = self.config.beta + temporal_penalty  # for t < T
            beta_last = self.config.beta  # for t = T

            new_a_main = self._solve_woodbury_system(
                numerator=numerator[:-1],
                W_expanded=W_expanded,
                a_shape=(state.a.shape[0] - 1, *state.a.shape[1:]),
                beta_effective=beta_main,
            )
            new_a_last = self._solve_woodbury_system(
                numerator=numerator[-1:],
                W_expanded=W_expanded,
                a_shape=(1, *state.a.shape[1:]),
                beta_effective=beta_last,
            )
            new_a = torch.cat([new_a_main, new_a_last], dim=0)

        # --- Standard dense path ---
        # Builds D_main and D_last explicitly via get_a_denominator and dispatches
        # to the spiking solver, which handles the two boundaries in a single call.
        else:
            denominator_main, denominator_last, in_features = (
                next_layer.get_a_denominator(
                    beta_current=self.config.beta,
                    rho_current=self.config.rho,
                    thetas_current=self.config.thetas,
                    a_shape=state.a.shape,
                )
            )
            new_a = next_layer._solve_activation_system(
                numerator=numerator,
                denominator_main=denominator_main,
                denominator_last=denominator_last,
                a_shape=state.a.shape,
                in_features=in_features,
            )

        state.a.copy_(torch.clamp(new_a, min=0.0, max=1.0))

    def update_a_unrolled(
        self,
        t: int,
        cache: TemporalCache,
        next_layer: nn.Module,
        state: ADMM_LayerState,
    ) -> None:
        r"""Manages the unrolled (Gauss-Seidel) activation ($a$) update for spiking layers.

        Processes a **single timestep** $t$ using the most recently computed states,
        enabling causal information to propagate immediately within the same ADMM iteration.

        The three solver paths are the same as the vectorized `update_a`, but operate on
        a single slice $a_t$ instead of the full sequence:

        * **FFT path**: Calls the pre-inverted FFT unrolled solver, which performs a
          batched matrix-vector multiply in the frequency domain per timestep.
        * **Woodbury path**: Absorbs the temporal penalty into `beta_effective` for $t < T$
          and solves the per-step $M \\times M$ system.
        * **Standard path**: Looks up the precomputed (and pre-inverted) denominator from
          the `TemporalCache` and performs a single dense matrix-vector multiply.

        Refer to its vectorized counterpart, [update_a][admm.spiking_mixin.ADMM_Spiking.update_a].

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices (adjoint, forward pass, denominators).
            next_layer (nn.Module): The subsequent layer in the network.
            state (ADMM_LayerState): The state of the current layer.
        """
        is_last = t == state.z.size(0) - 1

        # ------------------------------------------------------------------
        # Step 1: Build the per-step numerator
        #         u_t = β·h(z_t) + adjoint_t + temporal_reset_penalty_t
        # ------------------------------------------------------------------
        h_t = self.config.beta * self.h(state.z[t])

        # Temporal reset penalty: -ρθ(z_{t+1} - δz_t - F_{t+1})  [active only for t < T]
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

        # ------------------------------------------------------------------
        # Step 2: Solve  (β_eff I + ρ A*A) a_t = u_t  via the right solver
        # ------------------------------------------------------------------

        # --- FFT path (convolutional layers only) ---
        # Uses the pre-inverted per-frequency-bin denominator stored in the cache.
        # The correct boundary matrix (main vs. last) was already selected by the
        # cache-building step, so we pass the appropriate one directly.
        if (
            self.config.use_fft
            and getattr(self, "convolution", False)
            and getattr(next_layer, "convolution", False)
        ):
            denominator_t = (
                cache.denominator_last if is_last else cache.denominator_main
            )

            new_a_t = fft_solve_activation_system_unrolled(
                next_layer, numerator=numerator, denominator=denominator_t
            )

        # --- Woodbury path (wide input layers: M << N) ---
        # Absorbs the temporal penalty into beta_effective for t < T.
        elif next_layer._should_use_woodbury(state.a[t].shape):
            temporal_penalty = self.config.rho * (self.config.thetas**2)
            beta_effective = (
                self.config.beta if is_last else (self.config.beta + temporal_penalty)
            )

            W_expanded = next_layer._get_expanded_weights(state.a[t].shape)
            new_a_t = self._solve_woodbury_system(
                numerator=numerator,
                W_expanded=W_expanded,
                a_shape=state.a[t].shape,
                beta_effective=beta_effective,
            )

        # --- Standard dense path ---
        # Retrieves the pre-inverted denominator for this boundary condition from
        # the cache and performs a single dense matrix-vector multiply.
        else:
            denominator_t = (
                cache.denominator_last if is_last else cache.denominator_main
            )
            new_a_t = next_layer._solve_activation_system_unrolled(
                numerator=numerator, denominator=denominator_t
            )

        state.a[t].copy_(torch.clamp(new_a_t, min=0.0, max=1.0))

    def update_lambda(self, state: ADMM_LayerState, a_prev: torch.Tensor) -> None:
        r"""Updates the Lagrange multiplier ($\lambda$) based on the current layer constraints for spiking neurons.

        Formula evaluated:
        $\lambda_l \leftarrow \lambda_l + \rho_l \big(z_{l,T} - \mathcal{F}_{l,T}\big)$

        You can find the corresponding non-spiking version [update_lambda][admm.core.ADMM_Layer.update_lambda] in the [Core][admm.core] module.

        Args:
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
