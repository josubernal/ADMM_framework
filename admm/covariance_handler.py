r"""
This module contains the covariance tracking mechanisms designed for ADMM.

Unlike the standard Backpropagation framework where gradients are accumulated
batch-by-batch to update weights via gradient descent, the ADMM framework solves
for layer parameters globally. This module provides the infrastructure to accumulate
the required sequence statistics (numerators and denominators) across all batches,
enabling exact least-squares updates for weights and biases at the end of the epoch.
"""

from typing import Optional

import torch

from .dataclasses import ADMM_LayerCovariance


class ADMM_CovarianceHandler:
    """Tracks global covariance matrices for parameter optimization across all batches.

    This handler accumulates the numerator and denominator terms required to solve
    the least-squares updates for layer weights and biases across the entire dataset.
    """

    def __init__(self, num_layers: int):
        """Initializes the covariance handler.

        Args:
            num_layers (int): The total number of layers in the network.
        """
        self.num_layers = num_layers
        self.states = [ADMM_LayerCovariance() for _ in range(num_layers)]

    def get_covariances(self, layer_idx: int) -> ADMM_LayerCovariance:
        """Retrieves the global accumulated state for a specific layer.

        Args:
            layer_idx (int): The target layer index.

        Returns:
            ADMM_LayerCovariance: The accumulated covariance dataclass.
        """
        return self.states[layer_idx]

    def set_pinv(self, layer_idx: int, pinv: Optional[torch.Tensor]) -> None:
        """Caches the pseudo-inverse for future epochs to avoid redundant inversions.

        Args:
            layer_idx (int): The target layer index.
            pinv (Optional[torch.Tensor]): The computed pseudo-inverse matrix.
        """
        self.states[layer_idx].pinv = pinv

    def reset_accumulators(self) -> None:
        """Clears the global accumulators at the start of Phase 1 of the [fitting loop][admm.manager.fit]."""
        for layer in self.states:
            layer.numerator = 0.0
            layer.denominator = 0.0
            layer.bias_sum = 0.0
            layer.bias_count = 0

    def accumulate(
        self,
        layer_idx: int,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        bias_count: Optional[int] = 0,
        bias_sum: Optional[torch.Tensor] = None,
    ) -> None:
        """Adds a batch's matrices to the global running total for a specific layer.

        Args:
            layer_idx (int): The index of the layer being updated.
            numerator (torch.Tensor): The accumulated numerator matrix (e.g., $S^T Z$).
            denominator (torch.Tensor): The accumulated denominator matrix (e.g., $S^T S$).
            bias_count (Optional[int], optional): The number of sequence elements contributing to the bias.
            bias_sum (Optional[torch.Tensor], optional): The accumulated spatial sum for bias calculation.
        """
        state = self.states[layer_idx]

        if isinstance(state.numerator, float):
            state.numerator = numerator.clone()
            state.denominator = denominator.clone()
        else:
            state.numerator.add_(numerator)
            state.denominator.add_(denominator)

        if bias_sum is not None:
            if isinstance(state.bias_sum, float):
                state.bias_sum = bias_sum.clone()
                state.bias_count = bias_count
            else:
                state.bias_sum.add_(bias_sum)
                state.bias_count += bias_count
