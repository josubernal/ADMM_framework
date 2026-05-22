r"""This module defines the absolute base for all ADMM layers.

It manages the dual variable ($\lambda$) updates, handles the global and local
configuration setup, and dictates the generic forward pass routing.

This class is designed to be strictly non-parametric (no weights or biases)
and non-temporal. All mathematical spatial transformations must be implemented
by subclasses.
"""

from typing import Optional

import torch
import torch.nn as nn

from .activation_functions import ADMM_Identity
from .dataclasses import ADMM_Config, ADMM_LayerConfig, ADMM_LayerState

####################################################################################################
# Base Layer Interface
####################################################################################################


class ADMM_Layer(nn.Module):
    r"""The core Base Layer Interface for all ADMM modules.

    Acts as the blueprint for all trainable and non-trainable ADMM layers.
    It manages the fundamental Lagrangian multiplier ($\lambda$) updates and
    securely handles the localized configuration setup.
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
            rho (float, optional): Affine penalty parameter ($\rho$). Defaults to the one set by the [layer configuration][src.admm.dataclasses.ADMM_LayerConfig].
            beta (float, optional): Activation penalty parameter ($\beta$). Defaults to None.
            deltas (float, optional): Temporal leakage parameter for spiking networks. Defaults to None.
            thetas (float, optional): Spiking threshold parameter. Defaults to None.
            use_reset (bool, optional): Whether to apply spike resets. Defaults to None.
            h (nn.Module, optional): The [activation function][src.admm.activation_functions] module. Defaults to [ADMM_Identity][src.admm.activation_functions.ADMM_Identity].
            config (ADMM_LayerConfig, optional): Local [layer configuration object][src.admm.dataclasses.ADMM_LayerConfig]. Defaults to standard configuration.
        """
        super().__init__()
        self.device: Optional[torch.device] = None
        self.global_config: Optional[ADMM_Config] = None
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
        """Cascades the [global configuration][src.admm.dataclasses.ADMM_Config] setup to this layer and its child modules.

        Args:
            global_config (ADMM_Config, optional): The network-wide configuration orchestrator.
        """
        self.global_config = global_config

        if hasattr(self, "h") and hasattr(self.h, "_setup"):
            self.h._setup(self.config)

    def forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        r"""Standard sequential pass for initialization or inference.

        * For Static Networks: Simply returns the spatial transformation ($z_l = F_l(a_{l-1})$).
        * For Spiking Networks (SNNs): Simulates the LIF mechanics.

        To refer to its spiking counterpart, see [forward][src.admm.spiking_mixin.ADMM_Spiking.forward].

        Args:
            a_prev (torch.Tensor): The input tensor from the previous layer.

        Returns:
            torch.Tensor: The output tensor after the spatial pass.
        """
        y = self.spatial_forward(a_prev)
        return y

    def update_lambda(self, state: ADMM_LayerState, a_prev: torch.Tensor) -> None:
        r"""Updates the Lagrange multiplier ($\lambda$) based on the current layer constraints.

        Formula evaluated:
        $\lambda_l \leftarrow \lambda_l + \rho_l \big(z_l - F_l(a_{l-1})\big)$

        You can find the corresponding spiking version [update_lambda][src.admm.spiking_mixin.ADMM_Spiking.update_lambda] in the [Spiking Mixin][src.admm.spiking_mixin] module.

        Args:
            state (ADMM_LayerState): The [specific state][src.admm.dataclasses.ADMM_LayerState] object for the current layer containing the current dual variable.
            a_prev (torch.Tensor): The activations from the previous layer to compute the forward constraint.
        """
        if not self.config.use_lagrange or state.lambda_lagrange is None:
            return
        forward = self.spatial_forward(a_prev)
        state.lambda_lagrange.add_(state.z, alpha=self.config.rho)
        state.lambda_lagrange.add_(forward, alpha=-self.config.rho)
