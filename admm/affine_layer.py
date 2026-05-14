r"""
This module manages layers with trainable parameters (Weights and Biases)
such as Linear and Conv2d layers. It contains the optimization logic
to update these parameters using standard spatial mathematics.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base_layer import ADMM_Layer
from .dataclasses import ADMM_LayerConfig, ADMM_LayerCovariance, ADMM_LayerState
from .functional.fft_convolution import (
    _solve_activation_system as fft_solve_activation_system,
)
from .functional.fft_convolution import (
    get_a_denominator as fft_get_a_denominator,
)
from .functional.utils import broadcast_to_match, should_use_woodbury
from .initializers import get_initializer
from .pooling import ADMM_Flatten
from .solvers import (
    solve_least_squares_weights,
    solve_linear_system,
    solve_woodbury_system_static,
)

############################################################################################################
# Affine Layer Manager
############################################################################################################


class ADMM_AffineLayer(ADMM_Layer):
    """Manages affine layers with trainable parameters like Linear and Conv2d.

    Handles the initialization of weights and biases and contains the optimization
    solvers to update them using ADMM principles.
    """

    def __init__(
        self,
        h: nn.Module = None,
        pool_op: nn.Module = None,
        rho: Optional[float] = None,
        beta: Optional[float] = None,
        deltas: Optional[float] = None,
        thetas: Optional[float] = None,
        use_reset: Optional[bool] = None,
        config: Any = None,
    ):
        """Initializes the trainable affine layer container.

        Args:
            h (nn.Module, optional): The activation function. Defaults to None.
            pool_op (nn.Module, optional): The pooling/flattening operator. Defaults to ADMM_Flatten.
            rho (float, optional): Affine penalty parameter. Defaults to None.
            beta (float, optional): Activation penalty parameter. Defaults to None.
            deltas (float, optional): Temporal leakage parameter. Defaults to None.
            thetas (float, optional): Spiking threshold parameter. Defaults to None.
            use_reset (bool, optional): Whether to apply spike resets. Defaults to None.
            config (ADMM_LayerConfig, optional): Layer configuration object. Defaults to None.
        """
        super().__init__(
            rho=rho,
            beta=beta,
            deltas=deltas,
            thetas=thetas,
            use_reset=use_reset,
            h=h,
            config=config,
        )
        self.W: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.pool_op = pool_op if pool_op is not None else ADMM_Flatten()

    def _init_weights_and_bias(self, weight_shape: tuple, bias_shape: tuple) -> None:
        """Initializes weights and biases based on the global strategy."""
        init_strategy = (
            self.global_config.init
            if getattr(self, "global_config", None) is not None
            else "s-uniform"
        )

        initializer = get_initializer(init_strategy)

        self.W = initializer.init_weights(weight_shape, device=self.device)
        self.b = (
            initializer.init_bias(bias_shape, device=self.device)
            if self.config.use_bias
            else 0
        )

    def _format_bias(
        self,
        state: ADMM_LayerState,
    ) -> Union[torch.Tensor, int]:
        """Reshapes the bias vector to match the dimensionality of the layer's output.

        Returns:
            Union[torch.Tensor, int]: The reshaped bias tensor, or 0 if bias is disabled.
        """
        if not self.config.use_bias:
            return 0
        target_shape = [1] * state.z.dim()
        target_shape[self.channel_dim] = self.b.size(0)
        return self.b.view(*target_shape)

    def _solve_activation_system(
        self,
        numerator: torch.Tensor,
        denominator_main: torch.Tensor,
        denominator_last: torch.Tensor,
        a_shape: tuple,
        in_features: int,
    ) -> torch.Tensor:
        """Universal Solver for the $a$ update routing to numerical handlers.

        For static (non-spiking) layers, `denominator_main` and `denominator_last` are
        identical, so we solve a single linear system using the last-step denominator.

        Refer to its spiking counterpart, [_solve_activation_system][admm.spiking_mixin.ADMM_Spiking._solve_activation_system],
        in the [Spiking Mixin][admm.spiking_mixin] module, which routes to the temporal solver.

        Args:
            numerator (torch.Tensor): The precomputed numerator tensor.
            denominator_main (torch.Tensor): The primary LHS system matrix (unused here; kept for API consistency).
            denominator_last (torch.Tensor): The LHS matrix used to solve the system.
            a_shape (tuple): The physical shape of the activation tensor.
            in_features (int): Number of input features.

        Returns:
            torch.Tensor: The exact updated activations.
        """
        return solve_linear_system(denominator_last, numerator, a_shape, in_features)

    def _solve_woodbury_system(
        self,
        numerator: torch.Tensor,
        W_expanded: torch.Tensor,
        a_shape: tuple,
    ) -> torch.Tensor:
        r"""Solves the activation system via the Woodbury Matrix Identity.

        Preferred when the number of output features $M$ is much smaller than the
        number of input features $N$ (i.e. $M \ll N$), so inverting the small
        $M \times M$ matrix is far cheaper than the $N \times N$ dense system.

        Formula evaluated (Woodbury simplified):
        $a = \frac{1}{\beta} u - \frac{\rho}{\beta} W^T (\beta I_M + \rho W W^T)^{-1} W u$

        Refer to its spiking counterpart, [_solve_woodbury_system][admm.spiking_mixin.ADMM_Spiking._solve_woodbury_system],
        in the [Spiking Mixin][admm.spiking_mixin] module, which absorbs the temporal penalty into $\beta$.

        Args:
            numerator (torch.Tensor): The precomputed numerator tensor $u$.
            W_expanded (torch.Tensor): The expanded weight matrix $W$ of shape `[out_features, in_features]`.
            a_shape (tuple): The desired output shape of the activation tensor.

        Returns:
            torch.Tensor: The solved activations $a$, reshaped to `a_shape`.
        """
        return solve_woodbury_system_static(
            W=W_expanded,
            B=numerator,
            beta=self.config.beta,
            rho=self.config.rho,
            a_shape=a_shape,
        )

    def _compute_bias_covariance(self, state: ADMM_LayerState, a_prev: torch.Tensor):
        in_mean = self.spatial_forward(a_prev, use_bias=False)

        in_mean.neg_().add_(state.z)
        if self.config.use_lagrange and state.lambda_lagrange is not None:
            lam_spatial = broadcast_to_match(state.lambda_lagrange, state.z)

            in_mean.add_(lam_spatial, alpha=1.0 / self.config.rho)
        return in_mean

    def get_v(
        self, state: ADMM_LayerState, include_reset: bool = False
    ) -> torch.Tensor:
        r"""Returns the target tensor $v$ for non spiking layers.

        Formula evaluated:

        * $v_l = z_l - b_l$
        * $v_l = z_l - b_l + \frac{\lambda_l}{\rho}$ If layer l has a Lagrangian multiplier

        You can find the corresponding spiking version [get_spiking_v][admm.temporal_helpers.get_spiking_v] in the [Temporal Helpers][admm.temporal_helpers] module.

        Returns:
            torch.Tensor: The computed target tensor $v$.
        """
        v = state.z.clone()
        bias_formatted = self._format_bias(state)
        if isinstance(bias_formatted, torch.Tensor):
            v.sub_(bias_formatted)
        if self.config.use_lagrange and state.lambda_lagrange is not None:
            lambda_lagrange = broadcast_to_match(state.lambda_lagrange, v)
            v.add_(lambda_lagrange, alpha=1.0 / self.config.rho)
        return v

    def compute_batch_covariances(
        self, state: ADMM_LayerState, a_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the numerator and denominator for the weight update.

        Calculates the numerator ($Y^T P$) and denominator ($P^T P$) required for
        the least-squares weight update step.

        It is overwritten in the convolutional mixin to avoid OOM errors, [compute_batch_covariances][admm.convolutional_mixin.ADMM_Convolution.compute_batch_covariances].

        Args:
            Y (torch.Tensor): The base target tensor.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The numerator and denominator matrices.
        """
        v = self.get_v(state)
        P = self._compute_P(a_prev)
        Y = v.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])

        numerator = Y.t() @ P
        denominator = P.t() @ P

        bias_sum = None
        bias_count = 0

        if self.config.use_bias:
            in_mean = self._compute_bias_covariance(state=state, a_prev=a_prev)
            dims = self._get_bias_reduction_dims()
            bias_sum = torch.sum(in_mean, dim=dims)

            num_elements = 1
            for d in dims:
                num_elements *= in_mean.shape[d]
            bias_count = num_elements

        return numerator, denominator, bias_sum, bias_count

    def update_weights(
        self, covariances: ADMM_LayerCovariance, cache_pinv: bool = False
    ) -> None:
        """Manages the update for the layer's weights by solving a regularized least-squares problem.

        Uses [compute_covariances][admm.affine.ADMM_AffineLayer.compute_covariances] to get the necessary matrices and then applies the [solver][admm.solvers].

        Args:
            a_prev (torch.Tensor): The activations from the previous layer.
            cache_pinv (bool, optional): If True, reuses the previously computed
                pseudoinverse to accelerate updates. Defaults to False.
        """

        new_W, temp_pinv = solve_least_squares_weights(
            covariances.numerator,
            covariances.denominator,
            use_cholesky=self.config.use_cholesky,
            cached_pinv=covariances.pinv if cache_pinv else None,
            use_cg=True,
        )
        self.W.copy_(new_W.detach().reshape(self.W.shape))
        return temp_pinv.detach() if cache_pinv and temp_pinv is not None else None

    def update_bias(self, covariances: ADMM_LayerCovariance) -> None:
        r"""Averages the residual errors to update the bias vector.

        Formula evaluated: $b_l = \text{mean}(z_l - A(a_{l-1}))$

        You can find the corresponding spiking version [update_bias][admm.spiking_mixin.ADMM_Spiking.update_bias] in the [Spiking Mixin][admm.spiking_mixin] module.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
        """
        if not self.config.use_bias or covariances.bias_count == 0:
            return

        global_bias_mean = covariances.bias_sum / covariances.bias_count
        self.b.copy_(global_bias_mean)

    def get_a_numerator_no_adjoint(
        self, state: ADMM_LayerState, a_prev: torch.Tensor
    ) -> torch.Tensor:
        r"""Calculates the non spiking numerator block for the spatial activation ($a$) update.

        Formula evaluated: $N = \beta_l h_l(z_l) +\rho_l \mathcal{A}_{l+1}^*\big(v_{l+1}\big)$

        Args:
            beta_current (float): The $\beta$ penalty parameter of the current layer.
            a_shape (tuple): The physical geometry shape of the activation tensor.
            h_z (torch.Tensor): The precomputed $h(z)$ term for the current activations.
        """
        numerator = self.h(state.z).clone().mul_(self.config.beta)
        return numerator

    def get_a_denominator(
        self, config_prev: ADMM_LayerConfig, W_expanded: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]:
        r"""Computes the non spiking denominator matrix for the activation ($a$) update.
        Reroutes to specialized solvers if necessary.

        Formula evaluated: $D = \beta_l I + \rho_{l+1} \mathcal{A}_{l+1}^* \circ \mathcal{A}_{l+1}$

        You can find the corresponding spiking version [get_spiking_a_denominator][admm.temporal_helpers.get_spiking_a_denominator] in the [Temporal Helpers][admm.temporal_helpers] module.

        Args:
            beta_current (float): The penalty parameter $\beta$ for the update.
            a_shape (tuple): The physical geometry shape of the activation tensor.

        Returns:
            tuple:
                - Union[torch.Tensor, dict]: The computed denominator matrix (or Woodbury params).
                - Union[torch.Tensor, dict]: The last-step denominator matrix (or Woodbury params).
                - int: The number of input features.
        """

        in_features = W_expanded.size(1)
        denominator = torch.matmul(W_expanded.t(), W_expanded)
        denominator.mul_(self.config.rho)
        denominator.diagonal().add_(config_prev.beta)
        return denominator, denominator, in_features

    def update_a(
        self,
        next_layer: nn.Module,
        next_state: ADMM_LayerState,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
    ) -> None:
        r"""Manages the activation ($a$) update for standard spatial layers.

        Assembles the numerator $u = \beta h(z_l) + \rho \mathcal{A}_{l+1}^*(v_{l+1})$
        and then routes to the appropriate linear solver based on the layer configuration:

        * **FFT path**: Used when both the current and next layers are convolutional and
          `use_fft` is enabled. Solves the system entirely in the frequency domain, bypassing
          the $O(N^3)$ spatial bottleneck.
        * **Woodbury path**: Used when the output dimension $M$ is much smaller than the input
          dimension $N$ ($M \ll N$). Inverts a tiny $M \times M$ system instead of the $N \times N$ one.
        * **Standard path**: Default dense solver. Builds the Gram matrix $D = \beta I + \rho W^T W$
          and solves the system directly.

        Refer to its spiking counterpart, [update_a][admm.spiking_mixin.ADMM_Spiking.update_a],
        in the [Spiking Mixin][admm.spiking_mixin] module.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            next_state (ADMM_LayerState): The state of the subsequent layer.
            state (ADMM_LayerState): The state of the current layer.
            a_prev (torch.Tensor): The previous layer's activations.
        """
        # ------------------------------------------------------------------
        # Step 1: Build the numerator  u = β·h(z_l) + ρ·A*_{l+1}(v_{l+1})
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

        # ------------------------------------------------------------------
        # Step 3: Solve  (β I + ρ A*A) a = u  via the appropriate solver
        # ------------------------------------------------------------------

        # --- FFT path (convolutional layers only) ---
        # Leverages the Convolution Theorem to diagonalize the spatial system
        # in the frequency domain, solving C_in independent systems per frequency bin.
        if self.config.use_fft and self.convolution and next_layer.convolution:
            denominator_main, denominator_last, in_features = fft_get_a_denominator(
                next_layer,
                config_prev=self.config,
                W_expanded=W_expanded,
                a_shape=state.a.shape,
            )

            new_a = fft_solve_activation_system(
                next_layer,
                numerator=numerator,
                denominator_main=denominator_main,
                denominator_last=denominator_last,
                a_shape=state.a.shape,
                in_features=in_features,
            )
        # --- Woodbury path (wide input layers: M << N) ---
        # Inverts an M×M matrix instead of N×N, dropping complexity from O(N³) to O(M³).
        elif should_use_woodbury(W=next_layer.W, W_expanded=W_expanded):
            new_a = self._solve_woodbury_system(
                numerator=numerator,
                W_expanded=W_expanded,
                a_shape=state.a.shape,
            )

        # --- Standard dense path ---
        # Builds D = β I + ρ W^T W explicitly and solves via linalg.solve.
        else:
            denominator_main, denominator_last, in_features = (
                next_layer.get_a_denominator(
                    config_prev=self.config, W_expanded=W_expanded
                )
            )
            new_a = next_layer._solve_activation_system(
                numerator=numerator,
                denominator_main=denominator_main,
                denominator_last=denominator_last,
                a_shape=state.a.shape,
                in_features=in_features,
            )

        state.a.copy_(new_a)
