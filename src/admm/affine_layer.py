r"""
This module provides the core infrastructure for layers with trainable affine
parameters (e.g., Linear, Conv2d). It encapsulates the exact mathematical solvers
required to optimize these parameters ($W$ and $b$) and their corresponding activations ($a$) dynamically
routing computations through standard dense, Woodbury, or Fast Fourier Transform (FFT) paths.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_layer import ADMM_Layer
from .dataclasses import ADMM_LayerCovariance, ADMM_LayerState
from .functional.a_update_solvers import (
    solve_fft_system_static,
    solve_standard_system_static,
    solve_woodbury_system_static,
)
from .functional.utils import broadcast_to_match, should_use_woodbury
from .functional.weights_solvers import solve_weights
from .initializers import get_initializer
from .pooling import ADMM_Flatten

############################################################################################################
# Affine Layer Manager
############################################################################################################


class ADMM_AffineLayer(ADMM_Layer):
    """Manages affine layers with trainable parameters like Linear and Conv2d.

    Handles the initialization of weights and biases and contains the core
    optimization solvers to update them using Alternating Direction Method
    of Multipliers (ADMM) principles. In addition, handles the activation updates.
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
            h (Optional[nn.Module]): The activation function. Defaults to [ADMM_Identity()][src.admm.activation_functions.ADMM_Identity].
            pool_op (Optional[nn.Module]): The pooling/flattening operator. Defaults to [ADMM_Flatten()][src.admm.pooling.ADMM_Flatten].
            rho (Optional[float]): Affine penalty parameter. Defaults to the one set by the [layer configuration][src.admm.dataclasses.ADMM_LayerConfig].
            beta (Optional[float]): Activation penalty parameter. Defaults to the one set by the [layer configuration][src.admm.dataclasses.ADMM_LayerConfig].
            deltas (Optional[float]): Temporal leakage parameter. Defaults to the one set by the [layer configuration][src.admm.dataclasses.ADMM_LayerConfig].
            thetas (Optional[float]): Spiking threshold parameter. Defaults to the one set by the [layer configuration][src.admm.dataclasses.ADMM_LayerConfig].
            use_reset (Optional[bool]): Whether to apply spike resets. Defaults to the what is set by the [layer configuration][src.admm.dataclasses.ADMM_LayerConfig].
            config (Optional[ADMM_LayerConfig]): Local [layer configuration object][src.admm.dataclasses.ADMM_LayerConfig]. Defaults to standard configuration.
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
        """Initializes weights and biases based on the layer's [global strategy][src.admm.dataclasses.ADMM_Config].

        Args:
            weight_shape (tuple): Original spatial/channel shape of the weights.
            bias_shape (tuple): Original shape of the bias.
        """
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

        Args:
            state (ADMM_LayerState):[State object][src.admm.dataclasses.ADMM_LayerState] of the layer containing activation tensors.

        Returns:
            Union[torch.Tensor, int]: The reshaped bias tensor, or 0 if bias is disabled.
        """
        if not self.config.use_bias:
            return 0
        target_shape = [1] * state.z.dim()
        target_shape[self.channel_dim] = self.b.size(0)
        return self.b.view(*target_shape)

    def get_v(self, state: ADMM_LayerState) -> torch.Tensor:
        r"""Returns the target tensor $v$ for non-spiking layers.

        Formula evaluated:

        $$ v_l = z_l - b_l $$

        If layer $l$ has a Lagrangian multiplier:

        $$ v_l = z_l - b_l + \frac{\lambda_l}{\rho}$$

        Note:
            You can find the corresponding spiking version [`get_v`][src.admm.spiking_mixin.ADMM_Spiking.get_v] in the
            [`admm.spiking_mixin`][src.admm.spiking_mixin] module.

        Args:
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the current layer.

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

    def _compute_bias_covariance(
        self, state: ADMM_LayerState, a_prev: torch.Tensor
    ) -> torch.Tensor:
        r"""Computes the mean spatial covariance for the bias update step.

        Formula evaluated:

        $$z_l - \mathcal{A}_l(a_{l-1})$$

        If layer $l$ has a Lagrangian multiplier:

        $$ z_l - \mathcal{A}_l(a_{l-1}) + \frac{\lambda}{\rho} $$

        Note:
            We separate this function, because it is overriden by it's spiking version [`_compute_bias_covariance`][src.admm.spiking_mixin.ADMM_Spiking._compute_bias_covariance] in the
            [`admm.spiking_mixin`][src.admm.spiking_mixin] module.

        Args:
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the current layer.
            a_prev (torch.Tensor): The activations from the previous layer.

        Returns:
            torch.Tensor: The intermediate covariance tensor used to calculate the bias sum.
        """
        in_mean = self.spatial_forward(a_prev, use_bias=False)

        in_mean.neg_().add_(
            state.z
        )  # We perform .neg_() trick so we can save memory allocation
        if self.config.use_lagrange and state.lambda_lagrange is not None:
            lam_spatial = broadcast_to_match(state.lambda_lagrange, state.z)
            in_mean.add_(lam_spatial, alpha=1.0 / self.config.rho)
        return in_mean

    def compute_batch_covariances(
        self, state: ADMM_LayerState, a_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        r"""Computes the numerator and denominator matrices for the weight update.

        Calculates the numerator ($Y^T P$) and denominator ($P^T P$) required for
        the least-squares weight update step.

        Note:
            Calls [`get_v`][src.admm.affine_layer.ADMM_AffineLayer.get_v], [`_compute_bias_covariance`][src.admm.affine_layer.ADMM_AffineLayer._compute_bias_covariance], and
            `_compute_P` and `_get_bias_reduction_dims` defined by each layer in the [Layers module][src.admm.layers]
        Args:
            state (ADMM_LayerState): State object of the current layer.
            a_prev (torch.Tensor): The activations from the previous layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]: A tuple containing:
                - The numerator matrix.
                - The denominator matrix.
                - The sum of the bias covariances (or None if bias is disabled).
                - The count of elements reduced for the bias calculation.
        """
        v = self.get_v(state)
        P = self._compute_P(a_prev)
        # Flatten spatial/temporal dimensions into independent observations
        # Moves the channel dimension to the end, then collapses Batch x Time x H x W into rows. (2D matrix)
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
    ) -> Optional[torch.Tensor]:
        """Manages the update for the layer's weights by solving a least-squares problem.

        Uses the provided covariances to solve the dense system and caches the pseudoinverse
        of the first layer to accelerate future updates.

        Note:
            Handles both static and spiking updates.

        Args:
            covariances (ADMM_LayerCovariance): [The precomputed covariances][src.admm.dataclasses.ADMM_LayerCovariance] containing
                the numerator, denominator, and optionally a cached pseudoinverse.
            cache_pinv (bool, optional): If True, computes and returns the pseudoinverse
                to accelerate future updates. Defaults to False.

        Returns:
            Optional[torch.Tensor]: The detached pseudoinverse if `cache_pinv` is True,
            otherwise None.
        """

        new_W, temp_pinv = solve_weights(
            covariances.numerator,
            covariances.denominator,
            config=self.global_config,
            cached_pinv=covariances.pinv if cache_pinv else None,
        )
        self.W.copy_(new_W.detach().reshape(self.W.shape))
        return temp_pinv.detach() if cache_pinv and temp_pinv is not None else None

    def update_bias(self, covariances: ADMM_LayerCovariance) -> None:
        r"""Averages the residual errors to update the bias vector.

        Formula evaluated:
        $b_l = \text{mean(bias covariances)}$

        Note:
            Handles both static and spiking updates.

        Args:
            covariances (ADMM_LayerCovariance): [The precomputed covariances][src.admm.dataclasses.ADMM_LayerCovariance] containing
                the bias sum and bias count.
        """
        if not self.config.use_bias or covariances.bias_count == 0:
            return

        global_bias_mean = covariances.bias_sum / covariances.bias_count
        self.b.copy_(global_bias_mean)

    def _get_a_numerator_no_adjoint(
        self,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculates the non-spiking numerator block for the spatial activation ($a$) update.

        Formula evaluated:
        $N = \beta_l h_l(z_l)$

        Note:
            We separate this function, because it is overriden by it's spiking version [`_get_a_numerator_no_adjoint`][src.admm.spiking_mixin.ADMM_Spiking._get_a_numerator_no_adjoint] in the
            [`admm.spiking_mixin`][src.admm.spiking_mixin] module.


        Args:
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] containing the current pre-activations ($z$).
            a_prev (torch.Tensor): The activations from the previous layer. [(Needed for the spiking override)][src.admm.spiking_mixin.ADMM_Spiking._get_a_numerator_no_adjoint]

        Returns:
            torch.Tensor: The computed partial numerator tensor.
        """
        numerator = self.h(state.z).clone().mul_(self.config.beta)
        return numerator

    def _solve_standard_system(
        self,
        next_layer: nn.Module,
        W_expanded: torch.Tensor,
        numerator: torch.Tensor,
        a_shape: tuple,
    ) -> torch.Tensor:
        r"""Solves the standard dense system for spatial activations.
        Computes the denominator and calls the solver [`solve_standard_system_static`][src.admm.functional.a_update_solvers.solve_standard_system_static]

        Denominator formula:
        $\beta_lI + \rho_{l+1}\mathcal{A}_{l+1}^* \circ \mathcal{A}_{l+1}$

        Note:
            We separate this function, because it is overriden by it's spiking version [`_solve_standard_system`][src.admm.spiking_mixin.ADMM_Spiking._solve_standard_system] in the
            [`admm.spiking_mixin`][src.admm.spiking_mixin] module.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            W_expanded (torch.Tensor): The expanded weight matrix of the next layer.
            numerator (torch.Tensor): The assembled numerator target tensor.
            a_shape (tuple): The desired output shape of the activation tensor.

        Returns:
            torch.Tensor: The solved spatial activations.
        """
        in_features = W_expanded.size(1)
        denominator = torch.matmul(W_expanded.t(), W_expanded)
        denominator.mul_(next_layer.config.rho)
        denominator.diagonal().add_(self.config.beta)
        new_a = solve_standard_system_static(
            denominator=denominator,
            numerator=numerator,
            a_shape=a_shape,
            in_features=in_features,
        )
        return new_a

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
            We separate this function, because it is overriden by it's spiking version [`_solve_woodbury_system`][src.admm.spiking_mixin.ADMM_Spiking._solve_woodbury_system] in the
            [`admm.spiking_mixin`][src.admm.spiking_mixin] module.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            W_expanded (torch.Tensor): The expanded weight matrix of the next layer.
            numerator (torch.Tensor): The assembled numerator target tensor.
            a_shape (tuple): The desired output shape of the activation tensor.

        Returns:
            torch.Tensor: The solved spatial activations.
        """
        return solve_woodbury_system_static(
            W_expanded=W_expanded,
            numerator=numerator,
            beta=self.config.beta,
            rho=next_layer.config.rho,
            a_shape=a_shape,
        )

    def _solve_fft_system(
        self,
        next_layer: nn.Module,
        numerator: torch.Tensor,
        a_shape: tuple,
    ) -> torch.Tensor:
        r"""Solves the activation system via the Fast Fourier Transform (FFT) for static layers.
         Computes the denominator and calls the solver [`solve_fft_system_static`][src.admm.functional.a_update_solvers.solve_fft_system_static]

        Denominator formula evaluated in the frequency domain:
        $\beta_l I + \rho_{l+1} \mathcal{F}(W_{l+1})^* \mathcal{F}(W_{l+1})$

        Note:
             We separate this function, because it is overriden by it's spiking version [`_solve_fft_system`][src.admm.spiking_mixin.ADMM_Spiking._solve_fft_system] in the
             [`admm.spiking_mixin`][src.admm.spiking_mixin] module.

        Args:
             next_layer (nn.Module): The subsequent layer in the network.
             numerator (torch.Tensor): The assembled numerator target tensor.
             a_shape (tuple): The desired output shape of the activation tensor.

        Returns:
             torch.Tensor: The solved spatial activations.
        """

        height, width = a_shape[-2:]
        _, _, kernel_h, kernel_w = next_layer.W.shape

        pad_h = height - kernel_h
        pad_w = width - kernel_w
        padded_kernel = F.pad(next_layer.W, (0, pad_w, 0, pad_h))

        kernel_fft = torch.fft.fft2(padded_kernel)
        kernel_fft = kernel_fft.permute(2, 3, 0, 1)  # Shape: [H, W, C_out, C_in]

        denominator = torch.matmul(kernel_fft.conj().transpose(-2, -1), kernel_fft)
        denominator.mul_(next_layer.config.rho)
        denominator.diagonal(dim1=-2, dim2=-1).add_(self.config.beta)

        return solve_fft_system_static(
            numerator=numerator,
            denominator=denominator,
        )

    def update_a(
        self,
        next_layer: nn.Module,
        next_state: ADMM_LayerState,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
    ) -> None:
        r"""Manages the activation ($a$) update for standard spatial layers.

        Assembles the numerator $n = \beta_l h_l(z_l) + \rho_{l+1} \mathcal{A}_{l+1}^*(v_{l+1})$
        and dynamically routes to the optimal linear solver based on the layer configurations:

        * **FFT path**: Used when both the current and next layers are convolutional and
          `use_fft` is enabled. Solves the system entirely in the frequency domain,
          bypassing the $O(N^3)$ spatial bottleneck.
        * **Woodbury path**: Used when the output dimension $M$ is much smaller than the input
          dimension $N$ ($M \ll N$). Inverts a tiny $M \times M$ system instead of the dense $N \times N$ one.
        * **Standard path**: Default dense solver.

        Note:
            Handles both static and spiking updates. This is archieved by overriding the necessary internal functions.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            next_state (ADMM_LayerState): The [state object][src.admm.dataclasses.ADMM_LayerState] of the subsequent layer.
            state (ADMM_LayerState): The [state object][src.admm.dataclasses.ADMM_LayerState] of the current layer.
            a_prev (torch.Tensor): The previous layer's activations.
        """

        # Step 1: numerator = β_l·h_l(z_l) + ρ_{l+1}·A*_{l+1}(v_{l+1})
        numerator = self._get_a_numerator_no_adjoint(state=state, a_prev=a_prev)

        inside_adjoint = next_layer.get_v(next_state)
        adjoint = next_layer.adjoint_operator(
            inside_adjoint, original_input_shape=state.a.shape
        )
        numerator.add_(adjoint, alpha=next_layer.config.rho)

        # Step 2: Expand weights (handles Linear, Conv, pooled variants)
        W_expanded = next_layer.pool_op.expand_weights(
            W=next_layer.W, a_shape=state.a.shape
        )

        # Step 3: Solve  (β_l I + ρ_{l+1} A_{l+1}*A_{l+1}) a_l = numerator
        # FFT path
        if (
            self.config.use_fft
            and getattr(self, "convolution", False)
            and getattr(next_layer, "convolution", False)
        ):
            new_a = self._solve_fft_system(
                next_layer=next_layer,
                numerator=numerator,
                a_shape=state.a.shape,
            )

        # Woodbury path
        elif should_use_woodbury(W=next_layer.W, W_expanded=W_expanded):
            new_a = self._solve_woodbury_system(
                next_layer=next_layer,
                W_expanded=W_expanded,
                numerator=numerator,
                a_shape=state.a.shape,
            )

        # Standard path
        else:
            new_a = self._solve_standard_system(
                next_layer=next_layer,
                W_expanded=W_expanded,
                numerator=numerator,
                a_shape=state.a.shape,
            )

        state.a.copy_(new_a)
