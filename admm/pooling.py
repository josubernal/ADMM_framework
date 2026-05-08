"""
This module contains pooling and flattening operations designed specifically
for Alternating Direction Method of Multipliers (ADMM) based neural network training.

ADMM optimization requires mathematically explicit "adjoint" (transpose) linear
operators to map auxiliary variables and errors back to their original spatial dimensions.
The classes defined here provide both the forward transformation (pooling/flattening)
and its corresponding mathematical adjoint.

IMPORTANT:
    All pooling methods flatten the output to a 2D shape `(Batch x Features)`
    for compatibility with linear layers by default.
"""

from abc import ABC, abstractmethod

import torch


class ADMMPoolingBase(ABC):
    """Abstract Base Class for ADMM Pooling and Flattening operators.

    Enforces that all pooling methods implement a forward pass, an explicit
    adjoint operator, and a weight expansion helper.
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward pooling or flattening operation.

        Args:
            x (torch.Tensor): The input tensor to be pooled/flattened.

        Returns:
            torch.Tensor: The pooled or flattened output tensor.
        """
        pass

    @abstractmethod
    def adjoint(
        self, output: torch.Tensor, original_input_shape=None, tb_shape=None
    ) -> torch.Tensor:
        """Applies the adjoint (transpose) operation to map variables back to the input space.

        Args:
            output (torch.Tensor): The tensor to be mapped back.
            original_input_shape (tuple, optional): The full target shape (e.g., [B, C, H, W]
                or [T, B, C, H, W]). Used to restore the exact spatial grid. Defaults to None.
            tb_shape (tuple, optional): A tuple (Time, Batch) used specifically to split a
                collapsed first dimension back into separate Time and Batch dimensions.
                Defaults to None.

        Returns:
            torch.Tensor: The mapped tensor in its original spatial dimensions.
        """
        pass

    @abstractmethod
    def expand_weights(self, W: torch.Tensor, original_a_shape=None) -> torch.Tensor:
        """Expands the weights of the next layer to match the spatial dimensions of a.

        This is necessary for computing the linear system in ADMM when spatial pooling
        is involved.

        Args:
            W (torch.Tensor): The weight matrix of the next layer (shape [Out, In]).
            original_a_shape (tuple, optional): The original shape of a before pooling
                (e.g., [B, C, H, W]) used to determine how to expand W. Defaults to None.

        Returns:
            torch.Tensor: The expanded weight tensor.
        """
        pass


####################################################################################################
# POOLING OPERATORS
####################################################################################################


class ADMM_Flatten(ADMMPoolingBase):
    """Explicit Flattening Operator for ADMM.

    How it works:

    * **Forward**: Collapses spatial dimensions (e.g., `[B, C, H, W]`) into a flat vector `[B, N]`.
    * **Adjoint**: Reconstructs the original spatial shape from the flat vector. Because flattening
        doesn't change any values (just their arrangement), the adjoint is simply a reshape.
    """

    def __call__(self, x):
        return x.reshape(x.size(0), -1)

    def adjoint(self, output, original_input_shape=None, tb_shape=None):
        if original_input_shape is not None:
            return output.reshape(original_input_shape)
        if tb_shape is not None:
            return output.reshape(tb_shape[0], tb_shape[1], *output.shape[1:])
        return output

    def expand_weights(self, W: torch.Tensor, original_a_shape: tuple):
        return W.view(W.size(0), -1)


class ADMM_GAP(ADMMPoolingBase):
    """Global Average Pooling (GAP) Operator for ADMM.

    How it works:

    * **Forward**: Averages an entire spatial grid `(H x W)` down to a single `1x1` pixel per channel,
        then flattens it.
    * **Adjoint**: To act as the mathematical transpose of an average, the adjoint takes the
        incoming `1x1` error, broadcasts (copies) it back across the full `H x W` spatial grid,
        and divides by the total number of pixels `(H * W)` to maintain energy equivalence.
    """

    def __call__(self, x):
        if x.dim() > 2:
            return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(
                x.size(0), -1
            )
        return x.reshape(x.size(0), -1)

    def adjoint(self, output, original_input_shape=None, tb_shape=None):
        if original_input_shape is not None and len(original_input_shape) > 3:
            H, W = original_input_shape[-2:]
            grad_spatial = output.view(output.size(0), -1, 1, 1).expand(
                -1, -1, H, W
            ) / (H * W)

            if tb_shape is not None:
                return grad_spatial.reshape(
                    tb_shape[0], tb_shape[1], *grad_spatial.shape[1:]
                )
            return grad_spatial.reshape(original_input_shape)

        # Fallback for flat shapes
        if tb_shape is not None:
            return output.reshape(tb_shape[0], tb_shape[1], *output.shape[1:])
        return output

    def expand_weights(self, W: torch.Tensor, original_a_shape: tuple):
        h, w = original_a_shape[-2:]
        in_c = original_a_shape[-3]
        out_f = W.shape[0]

        pool_h, pool_w = (1, 1)

        W_spatial = W.view(out_f, in_c, pool_h, pool_w)
        W_expanded = torch.nn.functional.interpolate(
            W_spatial, size=(h, w), mode="nearest"
        )
        scale = (pool_h * pool_w) / (h * w)

        return (W_expanded * scale).view(out_f, -1)


class ADMM_SpatialPool(ADMMPoolingBase):
    """Adaptive Spatial Pooling Operator for ADMM.

    How it works:

    * **Forward**: Downsamples the spatial grid to a specific target size (e.g., `4x4`) using
        average pooling, then flattens the result.
    * **Adjoint**: Reshapes the flat error back to the pooled grid (e.g., `4x4`), then upsamples
        it back to the original grid using nearest-neighbor interpolation. The values are
        scaled by the pooling area ratio so that the adjoint accurately represents the
        transpose of the forward block-averaging matrix.

    Attributes:
        output_size (tuple): The target spatial dimensions `(H, W)` for the pool.
    """

    def __init__(self, output_size=(4, 4)):
        self.output_size = output_size

    def __call__(self, x):
        if x.dim() > 2:
            return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).view(
                x.size(0), -1
            )
        return x.reshape(x.size(0), -1)

    def adjoint(self, output, original_input_shape=None, tb_shape=None):
        if original_input_shape is not None and len(original_input_shape) > 3:
            H_orig, W_orig = original_input_shape[-2:]
            H_pool, W_pool = self.output_size

            grad_pool = output.view(output.size(0), -1, H_pool, W_pool)
            scale_factor = (H_pool * W_pool) / (H_orig * W_orig)
            grad_spatial = (
                torch.nn.functional.interpolate(
                    grad_pool, size=(H_orig, W_orig), mode="nearest"
                )
                * scale_factor
            )

            if tb_shape is not None:
                return grad_spatial.reshape(
                    tb_shape[0], tb_shape[1], *grad_spatial.shape[1:]
                )
            return grad_spatial

        return output

    def expand_weights(self, W: torch.Tensor, original_a_shape: tuple):
        h, w = original_a_shape[-2:]
        in_c = original_a_shape[-3]
        out_f = W.shape[0]

        pool_h, pool_w = self.output_size

        W_spatial = W.view(out_f, in_c, pool_h, pool_w)
        W_expanded = torch.nn.functional.interpolate(
            W_spatial, size=(h, w), mode="nearest"
        )
        scale = (pool_h * pool_w) / (h * w)

        return (W_expanded * scale).view(out_f, -1)
