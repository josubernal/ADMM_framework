r"""This module defines the absolute base for all ADMM layers.
It manages the fundamental auxiliary variables ($a$ and $z$), the dual variable ($\lambda$),
handles the configuration setup, and dictates the generic forward passes.

It is strictly non-parametric (no weights/biases) and non-temporal.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from .activations import ADMM_Identity
from .dataclasses import ADMM_Config, ADMM_LayerConfig

####################################################################################################
# Base Layer Interface
####################################################################################################


class ADMM_Layer(nn.Module):
    r"""The core Base Layer Interface for all ADMM modules.

    Manages the fundamental auxiliary variables ($a$, $z$, $\lambda$) and securely
    handles the localized configuration setup.
    """

    def __init__(
        self,
        rho: Optional[float] = None,
        beta: Optional[float] = None,
        deltas: Optional[float] = None,
        thetas: Optional[float] = None,
        use_reset: Optional[bool] = None,
        h: nn.Module = None,
        config: Optional[ADMM_LayerConfig] = None,
    ):
        """Initializes the base layer and its constraints.

        Args:
            rho (float, optional): Affine penalty parameter. Defaults to None.
            beta (float, optional): Activation penalty parameter. Defaults to None.
            deltas (float, optional): Temporal leakage parameter. Defaults to None.
            thetas (float, optional): Spiking threshold parameter. Defaults to None.
            use_reset (bool, optional): Whether to apply spike resets. Defaults to None.
            h (nn.Module, optional): The activation function module. Defaults to ADMM_Identity.
            config (ADMM_LayerConfig, optional): Layer configuration object. Defaults to None.
        """
        super().__init__()
        self.device = None

        self.lambda_lagrange = None
        self.z = None
        self.a = None
        self.h = h if h is not None else ADMM_Identity()
        self.config = config if config is not None else ADMM_LayerConfig()
        self.config.rho = rho if rho is not None else self.config.rho
        self.config.beta = beta if beta is not None else self.config.beta
        self.config.deltas = deltas if deltas is not None else self.config.deltas
        self.config.thetas = thetas if thetas is not None else self.config.thetas
        self.config.use_reset = (
            use_reset if use_reset is not None else self.config.use_reset
        )

    def _setup(self, global_config: Optional[ADMM_Config] = None) -> None:
        """Cascades configuration setup to child modules."""
        self.global_config = global_config

        if hasattr(self, "h") and hasattr(self.h, "_setup"):
            self.h._setup(self.config)

    def _init_lambda_lagrange(self) -> None:
        """Initializes the Lagrange multiplier tensor with the specified shape."""
        if self.config.use_lagrange:
            shape = self.z.shape
            self.lambda_lagrange = torch.zeros(shape, device=self.device)

    def _broadcast_to_match(
        self, tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Helper method to safely align tensor dimensions for element-wise operations.

        Args:
            tensor (torch.Tensor): The tensor to be broadcasted.
            target_tensor (torch.Tensor): The reference tensor whose dimensions are matched.

        Returns:
            torch.Tensor: The broadcasted tensor matching the target's dimensionality.
        """
        if tensor.dim() < target_tensor.dim():
            missing_dims = target_tensor.dim() - tensor.dim()
            return tensor.view(*tensor.shape, *([1] * missing_dims))
        return tensor

    def forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        r"""Standard sequential pass for initialization or inference.

        * For Static Networks: Simply returns the spatial transformation ($z_l = F_l(a_{l-1})$).
        * For Spiking Networks (SNNs): Simulates the mechanics step-by-step.

        To refer to its spiking counterpart, see [forward][admm.spiking_mixin.ADMM_Spiking.forward]. For the vectorized optimization pass, see [vectorized_forward][admm.core.ADMM_Layer.vectorized_forward].

        Args:
            a_prev (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the spatial pass.
        """
        y = self.spatial_forward(a_prev)
        return y

    def vectorized_forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Vectorized ADMM pass for optimization and constraint evaluation.

        To refer to its spiking counterpart, see [vectorized_forward][admm.spiking_mixin.ADMM_Spiking.vectorized_forward]. For the unrolled version, see [forward][admm.core.ADMM_Layer.forward].

        Args:
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            torch.Tensor: The evaluated constraints including temporal dependencies.
        """
        y = self.spatial_forward(a_prev)
        return y

    def update_z_last(
        self, a_prev: torch.Tensor, labels: torch.Tensor, loss_f: Any = None
    ) -> None:
        """Applies the $z$ update using the [loss function operators][admm.loss_functions].

        You can find the corresponding unrolled and spiking versions [update_z_last_unrolled][admm.spiking_mixin.ADMM_Spiking.update_z_last_unrolled] and [update_z_last][admm.spiking_mixin.ADMM_Spiking.update_z_last] in the [Spiking Mixin][admm.spiking_mixin] module.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth target labels.
            loss_f (ADMM_Loss, optional): The objective function managing the update. Defaults to None.
        """
        forward = self.vectorized_forward(a_prev)
        labels = self._broadcast_to_match(labels, forward)
        if self.config.use_lagrange and self.lambda_lagrange is not None:
            lam = self._broadcast_to_match(self.lambda_lagrange, forward)
        else:
            lam = torch.zeros_like(forward)
        self.z.copy_(loss_f.update_z_last_core(forward, self.config.rho, labels, lam))

    def update_lambda(self, a_prev: torch.Tensor) -> None:
        r"""Updates the Lagrange multiplier ($\lambda$) based on the current layer constraints.

        Formula evaluated:
        $\lambda_l \leftarrow \lambda_l + \rho_l \big(z_l - F_l(a_{l-1})\big)$

        You can find the corresponding spiking version [update_lambda][admm.spiking_mixin.ADMM_Spiking.update_lambda] in the [Spiking Mixin][admm.spiking_mixin] module.

        Args:
            a_prev (torch.Tensor): The activations from the previous layer.
        """
        if not self.config.use_lagrange or self.lambda_lagrange is None:
            return
        forward = self.spatial_forward(a_prev)
        self.lambda_lagrange.add_(self.z, alpha=self.config.rho)
        self.lambda_lagrange.add_(forward, alpha=-self.config.rho)
