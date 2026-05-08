"""This module contains the core orchestrator for the network and ADMM optimizer.
ADMM breaks the network down into layer-wise sub-problems and updates in a loop:

* Weight Updates
* Activation & Pre-activation Updates
* Dual Variable / Lagrange Multiplier Updates

The manager handles both static and spiking networks, automatically adjusting the optimization loop based on the network type and selected training method.
"""

import random
import warnings
from typing import Optional, Union

import torch
import torch.nn as nn

from .dataclasses import ADMMConfig
from .initializers import get_initializer
from .loss_functions import ADMM_SSE


class ADMM(nn.Module):
    """Universal Manager for ADMM networks.

    Handles both Static and Spiking ADMM networks automatically, orchestrating
    the layer-wise alternating optimization loops. It manages the global
    hyperparameters and delegates the mathematical proximal updates to the
    individual layers.

    Args:
        layers (nn.ModuleList): The sequential list of ADMM layers comprising the network.
        device (torch.device, optional): The computing device. If None, auto-detects CUDA.
            Defaults to None.
        loss_f (ADMM_Loss, optional): The [ADMM-compatible objective function][admm.loss_functions].
            Defaults to [ADMM_SSE()][admm.loss_functions.ADMM_SSE].
        T (int, optional): The number of time steps for spiking networks. If None, assumes static network.
            Defaults to None.
        config (ADMMConfig, optional): [Global configuration][admm.dataclasses.ADMMConfig] for ADMM training. Defaults to standard settings.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        device=None,
        loss_f=None,
        T=None,
        config: Optional[ADMMConfig] = None,
    ):
        super().__init__()

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.loss_f = loss_f if loss_f is not None else ADMM_SSE()
        self.layers = layers
        self.L = len(self.layers)
        self.initialized = False

        self.T = T
        self.is_spiking = self.T is not None

        self.config = config if config is not None else ADMMConfig()

        if not self.is_spiking and self.config.train_method != "vectorized":
            warnings.warn(
                "Non-spiking networks can only be trained using the 'vectorized' method. "
                "Automatically switching train_method to 'vectorized'.",
                UserWarning,
            )
            self.config.train_method = "vectorized"

        self._configure_layers()

    def _configure_layers(self):
        """Initializes each layer."""
        for layer in self.layers:
            layer._setup(global_config=self.config)

    def _get_batchsize(self, inputs: torch.Tensor):
        """Extracts batch size dynamically based on the network type.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            int: The batch size (dim 1 for Spiking, dim 0 for Static).
        """
        return inputs.shape[1] if self.is_spiking else inputs.shape[0]

    def _get_time_steps(self):
        """Helper that returns time-steps depending on the selected training method.

        Returns:
            list or None: A list of time steps if applicable, otherwise None.
        """
        time_steps = None
        if self.is_spiking:
            if self.config.train_method.endswith("random"):
                time_steps = random.sample(range(self.T), self.T)
            elif self.config.train_method.endswith("sequential"):
                time_steps = list(range(self.T))
            elif self.config.train_method.endswith("backwards"):
                time_steps = list(range(self.T - 1, -1, -1))
        return time_steps

    def _get_layers(self):
        """Helper that returns layer order depending on the selected method.

        Returns:
            list or None: A list of time steps if applicable, otherwise None.
        """
        if self.config.layer_order == "backwards":
            layer_indices = list(range(self.L - 1, -1, -1))
        elif self.config.layer_order == "random-last":
            layer_indices = random.sample(range(self.L - 1), self.L - 1)
            layer_indices.append(self.L - 1)
        elif self.config.layer_order == "random":
            layer_indices = list(range(self.L))
            random.shuffle(layer_indices)
        elif self.config.layer_order == "sequential":
            layer_indices = list(range(self.L))
        else:
            raise ValueError(
                f"Invalid layer order. Selected method {self.config.layer_order} does not exist, please read the documentation."
            )
        return layer_indices

    def _init_states(self, inputs: torch.Tensor):
        """Warm-starts the ADMM auxiliary variables 'z' and 'a'.

        Utilizes the selected initialization strategy and initializes the Lagrange multiplier.

        Args:
            inputs (torch.Tensor): The initial input tensor.
        """
        self.initialized = True
        initializer = get_initializer(self.config.init)

        initializer.init_states(self.layers, inputs, self.device)

        for layer in self.layers:
            if getattr(layer, "use_lagrange", False):
                shape = layer.z[-1].shape if self.is_spiking else layer.z.shape
                layer.lambda_lagrange = torch.zeros(shape, device=inputs.device)

    def forward_model(
        self, inputs: torch.Tensor, return_firing_rates: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[float]]]:
        """Standard Feed-Forward pass.

        Args:
            inputs (torch.Tensor): The input tensor to the network.
            return_firing_rates (bool, optional): Whether to return firing rates. Defaults to False.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, list[float]]]:
                If `return_firing_rates` is False, returns only the final network output (`torch.Tensor`).
                If True, returns a tuple containing:

                - **torch.Tensor**: The final network output.
                - **list[float]**: The activation firing rates per layer.
        """
        x = inputs.to(self.device)
        final_z = None
        firing_rates = []

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                z_pred = layer.forward(x)
                x = layer.h(z_pred)

                if return_firing_rates and self.is_spiking:
                    if i < len(self.layers) - 1:
                        # Sum of all spikes divided by the total number of elements
                        layer_firing_rate = (
                            x.sum().item() / x.numel()
                            if self.is_spiking
                            else float("NaN")
                        )
                        firing_rates.append(layer_firing_rate)

                final_z = z_pred

        final_out = final_z[-1] if self.is_spiking else final_z
        batch_size = inputs.size(1) if self.is_spiking else inputs.size(0)

        if final_out.dim() > 2:
            final_out = final_out.view(batch_size, -1)
        if return_firing_rates:
            return final_out, firing_rates
        return final_out

    def _optimize_w_and_b(
        self, layer: nn.Module, a_prev: torch.Tensor, cache_pinv: bool = False
    ):
        """Unified interface for updating all trainable parameters (W, b).

        Args:
            layer (nn.Module): The layer to update.
            a_prev (torch.Tensor): The previous layer's activations.
            cache_pinv (bool, optional): Whether to cache the pseudoinverse. Defaults to False.
        """
        layer.update_weights(a_prev, cache_pinv=cache_pinv)
        if getattr(layer.config, "use_bias", False):
            layer.update_bias(a_prev)

    def _optimize_a_and_z(
        self,
        layer: nn.Module,
        next_layer: nn.Module,
        a_prev: torch.Tensor,
        time_steps=None,
    ):
        """Unified interface for updating activations and pre-activations variables (a, z).

        Args:
            layer (nn.Module): The current layer being optimized.
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        if self.config.train_method.startswith("unrolled") and getattr(
            layer, "spiking", False
        ):
            layer.update_az_interleaved(
                next_layer, a_prev, time_steps, self.config.update_z_first
            )
        elif self.config.train_method.startswith("decoupled") and getattr(
            layer, "spiking", False
        ):
            if self.config.update_z_first:
                layer.update_z_decoupled(a_prev, time_steps)
                layer.update_a(next_layer, a_prev)
            else:
                layer.update_a(next_layer, a_prev)
                layer.update_z_decoupled(a_prev, time_steps)
        else:
            if self.config.update_z_first:
                layer.update_z(a_prev)
                layer.update_a(next_layer, a_prev)
            else:
                layer.update_a(next_layer, a_prev)
                layer.update_z(a_prev)

    def _optimize_z_last(
        self,
        layer: nn.Module,
        a_prev: torch.Tensor,
        labels: torch.Tensor,
        time_steps=None,
    ):
        """Default static state optimization for the final layer's pre-activations.

        Args:
            layer (nn.Module): The final layer of the network.
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        if self.config.train_method != "vectorized":
            layer.update_z_last_unrolled(a_prev, labels, time_steps, loss_f=self.loss_f)
        else:
            layer.update_z_last(a_prev, labels, loss_f=self.loss_f)

    @torch.no_grad()
    def fit(self, inputs: torch.Tensor, labels: torch.Tensor, warming: bool = False):
        """Orchestrates the fitting loop for the ADMM optimization process.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The target labels tensor.
            warming (bool, optional): If True, bypasses the lambda update. Defaults to False.
        """
        with torch.no_grad():
            if not self.initialized:
                self._init_states(inputs)

            time_steps = self._get_time_steps()
            layer_indices = self._get_layers()
            for l in layer_indices:
                layer = self.layers[l]
                a_prev = inputs if l == 0 else self.layers[l - 1].a
                if l < self.L - 1:
                    next_layer = self.layers[l + 1]
                    cache_pinv = l == 0
                    self._optimize_w_and_b(layer, a_prev, cache_pinv=cache_pinv)
                    self._optimize_a_and_z(layer, next_layer, a_prev, time_steps)
                    if not warming:
                        layer.update_lambda(a_prev)
                    del a_prev
                elif l == self.L - 1:
                    self._optimize_w_and_b(layer, a_prev)
                    self._optimize_z_last(layer, a_prev, labels, time_steps)
                    if not warming:
                        layer.update_lambda(a_prev)
