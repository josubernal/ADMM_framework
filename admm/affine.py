r"""
This module manages layers with trainable parameters (Weights and Biases)
such as Linear and Conv2d layers. It contains the optimization logic
to update these parameters using standard spatial mathematics.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .core import ADMM_Layer
from .dataclasses import ADMMState
from .initializers import get_initializer
from .pooling import ADMM_Flatten
from .solvers import solve_least_squares_weights, solve_linear_system

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
            config (ADMMLayerConfig, optional): Layer configuration object. Defaults to None.
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
        self.pinv: Optional[torch.Tensor] = None
        self.W: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.pool_op = pool_op if pool_op is not None else ADMM_Flatten()

    def _init_weights_and_bias(self, weight_shape: tuple, bias_shape: tuple) -> None:
        """Initializes weights and biases based on the global strategy."""
        init_strategy = (
            self.global_config.init
            if getattr(self, "global_config", None) is not None
            else "pytorch"
        )

        initializer = get_initializer(init_strategy)

        self.W = initializer.init_weights(weight_shape, device=self.device)
        self.b = (
            initializer.init_bias(bias_shape, device=self.device)
            if self.config.use_bias
            else 0
        )

    def _format_bias(self) -> Union[torch.Tensor, int]:
        """Reshapes the bias vector to match the dimensionality of the layer's output.

        Returns:
            Union[torch.Tensor, int]: The reshaped bias tensor, or 0 if bias is disabled.
        """
        if not self.config.use_bias:
            return 0
        target_shape = [1] * self.z.dim()
        target_shape[self.channel_dim] = self.b.size(0)
        return self.b.view(*target_shape)

    def _compute_covariances(
        self, Y: torch.Tensor, a_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the numerator and denominator for the weight update.

        Calculates the numerator ($Y^T P$) and denominator ($P^T P$) required for
        the least-squares weight update step.

        Args:
            Y (torch.Tensor): The base target tensor.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The numerator and denominator matrices.
        """
        P = self._compute_P(a_prev)
        Y_flat = Y.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])

        numerator = Y_flat.t() @ P
        denominator = P.t() @ P

        return numerator, denominator

    def _get_v(self) -> torch.Tensor:
        r"""Returns the target tensor $v$ for parameter updates.

        Formula evaluated:

        * Standard: $v = z - b$
        * Lagrangian: $v = z - b + \frac{\lambda}{\rho}$

        Returns:
            torch.Tensor: The computed target tensor $v$.
        """
        v = self.z.clone()
        bias_formatted = self._format_bias()
        if isinstance(bias_formatted, torch.Tensor):
            v.sub_(bias_formatted)
        if self.lambda_lagrange is not None:
            lambda_lagrange = self._broadcast_to_match(self.lambda_lagrange, v)
            v.add_(lambda_lagrange, alpha=1.0 / self.rho)
        return v

    def _get_expanded_weights(self, a_shape: tuple) -> torch.Tensor:
        """Delegates weight expansion to the pooling operator to match spatial dims.

        Args:
            a_shape (tuple): The physical geometry of the activations.

        Returns:
            torch.Tensor: The expanded or flattened weight matrix.
        """
        if self.pool_op and len(a_shape) > 3:
            return self.pool_op.expand_weights(self.W, original_a_shape=a_shape)

        return self.W.view(self.W.size(0), -1)

    def _get_a_numerator(
        self, beta_current: float, a_shape: tuple, h_z: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the linear numerator block for the spatial activation ($a$) update."""
        inside_adjoint = self._get_v()
        adjoint = self.adjoint_operator(inside_adjoint, original_input_shape=a_shape)
        numerator = h_z.clone().mul_(beta_current)
        numerator.add_(adjoint, alpha=self.rho)
        return numerator

    def _get_WtW(self, a_shape: tuple) -> Tuple[torch.Tensor, int]:
        """Template method to compute $W^T W$ matrix covariance."""
        W = self._get_expanded_weights(a_shape=a_shape)
        in_features = W.size(1)
        WtW = torch.matmul(W.t(), W)
        return WtW, in_features

    def _get_a_denominator(
        self, beta_current: float, a_shape: tuple
    ) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]:
        r"""Computes the denominator matrix for the activation ($a$) update step.

        Formula evaluated: $A = \beta I + \rho W^T W$

        Args:
            beta_current (float): The penalty parameter $\beta$ for the update.
            a_shape (tuple): The physical geometry shape of the activation tensor.

        Returns:
            tuple:
                - Union[torch.Tensor, dict]: The computed denominator matrix (or Woodbury params).
                - Union[torch.Tensor, dict]: The last-step denominator matrix (or Woodbury params).
                - int: The number of input features.
        """
        W = self._get_expanded_weights(a_shape=a_shape)
        out_features, in_features = W.shape
        if (
            getattr(self, "W", None) is not None
            and self.W.dim() == 2
            and in_features > out_features
        ):
            return self._get_woodbury_params(beta_current, a_shape)

        denominator, in_features = self._get_WtW(a_shape)
        denominator.mul_(self.rho)
        denominator.diagonal().add_(beta_current)
        return denominator, denominator, in_features

    def _get_woodbury_params(
        self, beta_current: float, a_shape: tuple
    ) -> Tuple[dict, dict, int]:
        """Generates dictionary parameters strictly formatted for the Woodbury Identity solver."""
        W = self._get_expanded_weights(a_shape=a_shape)
        _, in_features = W.shape
        dic = {"W": W, "beta": beta_current, "rho": self.rho}
        return dic, dic, in_features

    def update_weights(self, a_prev: torch.Tensor, cache_pinv: bool = False) -> None:
        """Manages the update for the layer's weights by solving a regularized least-squares problem.

        Args:
            a_prev (torch.Tensor): The activations from the previous layer.
            cache_pinv (bool, optional): If True, reuses the previously computed
                pseudoinverse to accelerate updates. Defaults to False.
        """
        v = self._get_v()
        numerator, denominator = self._compute_covariances(v, a_prev)
        new_W, temp_pinv = solve_least_squares_weights(
            numerator,
            denominator,
            use_cholesky=self.config.use_cholesky,
            cached_pinv=self.pinv if cache_pinv else None,
            use_cg=True,
        )
        self.W.copy_(new_W.detach().reshape(self.W.shape))
        if cache_pinv and temp_pinv is not None:
            self.pinv = temp_pinv.detach()
        else:
            self.pinv = None

    def update_bias(self, a_prev: torch.Tensor) -> None:
        r"""Averages the residual errors to update the bias vector.

        Formula evaluated: $b_l = \text{mean}(z_l - A(a_{l-1}))$

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
        """
        if not self.config.use_bias:
            return
        in_mean = self.spatial_forward(a_prev, use_bias=False)
        in_mean.neg_().add_(self.z)

        if self.lambda_lagrange is not None:
            lam_spatial = self._broadcast_to_match(self.lambda_lagrange, self.z)
            in_mean.add_(lam_spatial, alpha=1.0 / self.rho)

        new_bias = torch.mean(in_mean, dim=self._get_bias_reduction_dims())
        self.b.copy_(new_bias)

    def update_z(self, a_prev: torch.Tensor) -> None:
        """Applies the $z$ update using the activation function's proximal operator.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
        """
        forward = self.spatial_forward(a_prev)
        if getattr(self, "use_lagrange", False) and self.lambda_lagrange is not None:
            lam = self._broadcast_to_match(self.lambda_lagrange, forward)
            forward.sub_(lam, alpha=1.0 / self.rho)

        current_state = ADMMState(forward=forward, a=self.a, z=self.z)
        new_z = self.h.activation_z_update(current_state)
        self.z.data.copy_(new_z)

    def update_a(self, next_layer: nn.Module, a_prev: torch.Tensor) -> None:
        """Manages the activation ($a$) update for standard spatial layers.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
        """
        numerator = next_layer._get_a_numerator(
            beta_current=self.beta, a_shape=self.a.shape, h_z=self.h(self.z)
        )
        denominator_main, denominator_last, in_features = next_layer._get_a_denominator(
            beta_current=self.beta, a_shape=self.a.shape
        )
        new_a = next_layer._solve_activation_system(
            numerator=numerator,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            a_shape=self.a.shape,
            in_features=in_features,
        )

        self.a.copy_(new_a)

    def _solve_activation_system(
        self,
        numerator: torch.Tensor,
        denominator_main: torch.Tensor,
        denominator_last: torch.Tensor,
        a_shape: tuple,
        in_features: int,
    ) -> torch.Tensor:
        """Universal Solver for the $a$ update routing to numerical handlers.

        Args:
            numerator (torch.Tensor): The precomputed numerator tensor.
            denominator_main (torch.Tensor): The primary LHS system matrix.
            denominator_last (torch.Tensor): The secondary LHS system matrix.
            a_shape (tuple): The physical shape of the activation tensor.
            in_features (int): Number of input features tracking.

        Returns:
            torch.Tensor: The exact updated activations.
        """
        return solve_linear_system(denominator_last, numerator, a_shape, in_features)
