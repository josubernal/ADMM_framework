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
from typing import List, Tuple

import torch
import torch.nn as nn

from .dataclasses import ADMM_LayerState


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

    def _get_warmup_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Centralized helper to perform a warm-up forward pass and capture z and a shapes/values.

        Properly handles both Static (batch-wise) and Spiking (time-unrolled) temporal causality.
        """
        x = inputs.to(device)
        is_spiking = getattr(layers[0], "spiking", False)
        L = len(layers)

        z_inits = []
        a_inits = []

        if is_spiking:
            T = x.size(0)
            z_seqs = [[] for _ in range(L)]
            a_seqs = [[] for _ in range(L)]
            z_prevs = [None] * L
            a_prevs = [None] * L

            with torch.no_grad():
                for t in range(T):
                    a_t = x[t]
                    for i, layer in enumerate(layers):
                        z_t = layer.forward(a_t, z_prevs[i], a_prevs[i])
                        z_prevs[i] = z_t

                        a_t = layer.h(z_t)
                        a_prevs[i] = a_t

                        z_seqs[i].append(z_t)
                        a_seqs[i].append(a_t)

            for i in range(L):
                z_inits.append(torch.stack(z_seqs[i], dim=0))
                if i < L - 1:
                    a_inits.append(torch.stack(a_seqs[i], dim=0))
                else:
                    a_inits.append(None)
        else:
            with torch.no_grad():
                a_prev = x
                for i, layer in enumerate(layers):
                    z_t = layer.forward(a_prev)
                    a_t = layer.h(z_t)

                    z_inits.append(z_t)
                    if i < L - 1:
                        a_inits.append(a_t)
                    else:
                        a_inits.append(None)

                    a_prev = a_t

        return z_inits, a_inits

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ) -> list[ADMM_LayerState]:
        r"""Initializes the auxiliary variables $z$ and $a$.
        It performs a forward pass through the network and then populates $z$ and $a$ with the output.

        Args:
            layers (nn.ModuleList): The list of layers in the network.
            inputs (torch.Tensor): The initial input tensor to the network.
            device (torch.device): The device on which the tensors should be allocated.
        """
        z_inits, a_inits = self._get_warmup_states(layers, inputs, device)
        states = []
        is_spiking = getattr(layers[0], "spiking", False)

        for i, layer in enumerate(layers):
            z_init = z_inits[i].clone()
            a_init = a_inits[i].clone() if a_inits[i] is not None else None

            lam_shape = z_init[-1] if is_spiking else z_init
            lam_init = (
                torch.zeros_like(lam_shape) if layer.config.use_lagrange else None
            )

            states.append(ADMM_LayerState(z=z_init, a=a_init, lambda_lagrange=lam_init))

        return states


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
    ) -> list[ADMM_LayerState]:
        z_inits, a_inits = self._get_warmup_states(layers, inputs, device)
        states = []
        is_spiking = getattr(layers[0], "spiking", False)

        for i, layer in enumerate(layers):
            z_init = torch.randn_like(z_inits[i])
            a_init = layer.h(z_init) if i < len(layers) - 1 else None

            lam_shape = z_init[-1] if is_spiking else z_init
            lam_init = (
                torch.zeros_like(lam_shape) if layer.config.use_lagrange else None
            )

            states.append(ADMM_LayerState(z=z_init, a=a_init, lambda_lagrange=lam_init))

        return states


class StatesUniform(ADMM_Initializer):
    r"""
    Pure random Initialization for the states variables $z$ and $a$.
    """

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ) -> list[ADMM_LayerState]:
        z_inits, a_inits = self._get_warmup_states(layers, inputs, device)
        states = []
        is_spiking = getattr(layers[0], "spiking", False)

        for i, layer in enumerate(layers):
            z_init = torch.rand_like(z_inits[i])
            a_init = torch.rand_like(a_inits[i]) if a_inits[i] is not None else None

            lam_shape = z_init[-1] if is_spiking else z_init
            lam_init = (
                torch.zeros_like(lam_shape) if layer.config.use_lagrange else None
            )

            states.append(ADMM_LayerState(z=z_init, a=a_init, lambda_lagrange=lam_init))

        return states


class RelaxedSpikeInitializer(ADMM_Initializer):
    r"""
    Pre-activations ($z$) are given Gaussian noise to simulate membrane potential variance.
    Activations ($a$) are forced into a dense Bernoulli distribution ($p=0.5$).
    """

    def init_states(
        self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device
    ) -> list[ADMM_LayerState]:
        z_inits, a_inits = self._get_warmup_states(layers, inputs, device)
        states = []
        is_spiking = getattr(layers[0], "spiking", False)

        for i, layer in enumerate(layers):
            z_init = torch.randn_like(z_inits[i])
            a_init = (
                torch.randint(
                    0, 2, size=a_inits[i].shape, dtype=a_inits[i].dtype, device=device
                )
                if a_inits[i] is not None
                else None
            )

            lam_shape = z_init[-1] if is_spiking else z_init
            lam_init = (
                torch.zeros_like(lam_shape) if layer.config.use_lagrange else None
            )

            states.append(ADMM_LayerState(z=z_init, a=a_init, lambda_lagrange=lam_init))

        return states


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
