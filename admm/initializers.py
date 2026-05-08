r"""This module manages the initialization of the auxiliary states ($z$ and $a$) for ADMM layers.

Unlike standard gradient descent, ADMM exactly solves for and overwrites the network weights
during the very first optimization step. Consequently, the initial values of the weights
and biases do not directly dictate the training trajectory.

Instead, standard weight initialization strategies (e.g., Kaiming, Xavier) are leveraged
strictly to perform a mathematically sound initial forward pass. This warm-up pass seeds
the starting values for the pre-activations ($z$) and activations ($a$). Because the quality
of these initial states critically impacts the convergence speed and numerical stability
of the ADMM solver, proper initialization is essential.

Isolating this process from the main training loop keeps the core ADMM orchestrator clean
and provides a modular interface for experimenting with novel initialization strategies.
"""

import math
from abc import ABC

import torch
import torch.nn as nn


class ADMM_Initializer(ABC):
    """Abstract Base class for ADMM Initialization Strategies."""

    def init_weights(self, weight_shape: tuple, device: torch.device) -> torch.Tensor:
        r"""Initializes the weights for a layer to zero.

        Args:
            weight_shape (tuple): The dimensions of the weight tensor.

        Returns:
            torch.Tensor: The initialized weight parameter without gradient tracking.
        """
        return torch.zeros(*weight_shape, device=device)

    def init_bias(self, bias_shape: tuple, device: torch.device) -> torch.Tensor:
        r"""Initializes the bias for a layer to zero.

        Args:
            bias_shape (tuple): The dimensions of the bias tensor.

        Returns:
            torch.Tensor: The initialized bias parameter (zeros) without gradient tracking.
        """
        return torch.zeros(*bias_shape, device=device)

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ):
        r"""Initializes the auxiliary variables $z$ and $a$.
        It performs a forward pass through the network and then populates $z$ and $a$ with the output.

        Args:
            layers (nn.ModuleList): The list of layers in the network.
            inputs (torch.Tensor): The initial input tensor to the network.
            device (torch.device): The device on which the tensors should be allocated.
        """
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers[:-1]:
                z_pred = layer.forward(x)
                layer.z = z_pred.clone()
                layer.a = z_pred.clone()
                x = layer.a
            z_pred = layers[-1].forward(x)
            layers[-1].z = z_pred.clone()


class WeightsZerosInitializer(ADMM_Initializer):
    r"""Initializes weights and bias to zero. Initializes the auxiliary variables $z$ and $a$ performing a forward pass through the network."""

    def init_weights(
        self, weight_shape: tuple, device: torch.device = None
    ) -> torch.Tensor:
        return torch.zeros(*weight_shape, device=device)


class WeightsXavierInitializer(ADMM_Initializer):
    r"""Initializes weights with Xavier normal initialization. Initializes the auxiliary variables $z$ and $a$ performing a forward pass through the network."""

    def init_weights(
        self, weight_shape: tuple, device: torch.device = None
    ) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        nn.init.xavier_normal_(w)
        return w


class WeightsRandomInitializer(ADMM_Initializer):
    r"""Initializes weights with standard normal distribution. Initializes the auxiliary variables $z$ and $a$ performing a forward pass through the network.."""

    def init_weights(
        self, weight_shape: tuple, device: torch.device = None
    ) -> torch.Tensor:
        return torch.randn(*weight_shape, device=device)


class WeightsPytorchDefaultInitializer(ADMM_Initializer):
    r"""
    Exact replication of PyTorch's default initialization
    for nn.Linear and nn.Conv2d. Initializes weights to kaiming uniform ($a=\sqrt{5}$). Initializes the auxiliary variables $z$ and $a$ performing a forward pass through the network.
    """

    def init_weights(
        self, weight_shape: tuple, device: torch.device = None
    ) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        # This is the exact source-code configuration used by PyTorch
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        return w


class ZUniform(ADMM_Initializer):
    r"""
    Pure random Initialization for the state variable $z$. $a$ is initialized passing $z$ through the activation function.
    """

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers[:-1]:
                z_pred = layer.forward(x)
                layer.z = torch.randn_like(z_pred)
                a_pred = layer.h(layer.z) if layer.h is not None else layer.z
                layer.a = a_pred.clone()

                x = layer.a
            z_pred = layers[-1].forward(x)
            layer[-1].z = torch.randn_like(z_pred)


class StatesUniform(ADMM_Initializer):
    r"""
    Pure random Initialization for the states variables $z$ and $a$.
    """

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers[:-1]:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred)
                layer.z = torch.rand_like(z_pred)
                layer.a = torch.rand_like(a_pred)
                x = a_pred
            z_pred = layers[-1].forward(x)
            layers[-1].z = torch.rand_like(z_pred)


class RelaxedSpikeInitializer(ADMM_Initializer):
    r"""
    Pre-activations ($z$) are given Gaussian noise to simulate membrane potential variance.
    Activations ($a$) are forced into a dense Bernoulli distribution ($p=0.5$).
    """

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers[:-1]:
                # 1. Do a forward pass just to get the exact tensor geometries
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred

                layer.z = torch.randn_like(z_pred)
                layer.a = torch.randint(
                    0, 2, size=a_pred.shape, dtype=a_pred.dtype, device=device
                )

                x = a_pred
            z_pred = layers[-1].forward(x)
            layers[-1].z = torch.randn_like(z_pred)


def get_initializer(init_type: str) -> ADMM_Initializer:
    r"""Factory function to retrieve the correct initializer strategy.

    Args:
        init_type (str): The string identifier for the initialization strategy.
            Possible values include:

            * "zeros": WeightsZerosInitializer()
            * "xavier": WeightsXavierInitializer()
            * "wrandom": WeightsRandomInitializer()
            * "pytorch": WeightsPytorchDefaultInitializer()
            * "z-uniform": ZUniform()
            * "s-uniform": StatesUniform()
            * "relaxed":  RelaxedSpikeInitializer()

    Returns:
        ADMM_Initializer: An instance of the requested initialization strategy.

    Raises:
        ValueError: If the provided init_type is not found in the defined strategies.
    """
    strategies = {
        "zeros": WeightsZerosInitializer(),
        "xavier": WeightsXavierInitializer(),
        "wrandom": WeightsRandomInitializer(),
        "pytorch": WeightsPytorchDefaultInitializer(),
        "z-uniform": ZUniform(),
        "s-uniform": StatesUniform(),
        "relaxed": RelaxedSpikeInitializer(),
    }
    if init_type not in strategies:
        raise ValueError(
            f"Initialization method '{init_type}' not defined. Options: {list(strategies.keys())}"
        )
    return strategies[init_type]
