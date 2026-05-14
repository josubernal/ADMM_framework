r"""
This module contains the linear and convolutional layers for ADMM.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .affine_layer import ADMM_AffineLayer
from .dataclasses import ADMM_LayerConfig
from .functional.fft_convolution import (
    circular_adjoint,
    circular_compute_P,
    circular_forward,
)
from .functional.utils import fold_time, unfold_time
from .spiking_mixin import ADMM_Spiking

####################################################################################################
# LAYERS
####################################################################################################


class ADMM_Linear(ADMM_AffineLayer):
    r"""Standard Fully Connected Layer for ADMM.

    How it works:

    * **Forward**: Applies a linear transformation ($y = xW^T + b$) to the input.
    * **Adjoint**: Maps the target vector back to the input space using the transpose of the weights.
    * **Patch Matrix**: Simply uses the pooled input directly.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        h: nn.Module = None,
        pool_op: nn.Module = None,
        rho: float = None,
        beta: float = None,
        deltas: float = None,
        thetas: float = None,
        use_reset: bool = None,
        config: Optional[ADMM_LayerConfig] = None,
    ):
        super().__init__(
            h=h,
            pool_op=pool_op,
            rho=rho,
            beta=beta,
            deltas=deltas,
            thetas=thetas,
            use_reset=use_reset,
            config=config,
        )
        self.channel_dim = -1  # Targets the [B, C] dimension
        self.in_f = in_f
        self.out_f = out_f
        self.convolution = True

    def _setup(self, global_config=None) -> None:
        """Initializes the weight and bias tensors for the linear layer."""
        super()._setup(global_config)
        self._init_weights_and_bias((self.out_f, self.in_f), (self.out_f,))

    def spatial_forward(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        """Applies the linear transformation to the input data.

        Args:
            x (torch.Tensor): The input tensor.
            use_bias (bool, optional): Whether to apply the layer's bias. Defaults to True.

        Returns:
            torch.Tensor: The linearly transformed output.
        """
        b = (
            self.b
            if (isinstance(use_bias, bool) and use_bias and self.config.use_bias)
            else None
        )
        return torch.nn.functional.linear(self.pool_op(x), self.W, bias=b)

    def adjoint_operator(
        self, target: torch.Tensor, original_input_shape: tuple = None
    ) -> torch.Tensor:
        """Maps the target vector back to the input space via the transposed weights.
        Args:
            target (torch.Tensor): The error or target tensor.
            original_input_shape (tuple, optional): Used by the pooling operator to
                reconstruct geometric boundaries. Defaults to None.

        Returns:
            torch.Tensor: The transposed projection mapped back to the input domain.
        """
        out = torch.matmul(target, self.W)
        return self.pool_op.adjoint(out, original_input_shape=original_input_shape)

    def _compute_P(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Retrieves the patch matrix for linear least-squares updates."""
        return self.pool_op(a_prev)

    def _get_bias_reduction_dims(self) -> int:
        """Returns the reduction dimension index for bias averaging."""
        return (0,)


class ADMM_Conv2d(ADMM_AffineLayer):
    r"""Standard 2D Convolutional Layer for ADMM.

    How it works:

    * **Forward**: Applies a standard 2D convolution.
    * **Adjoint**: Applies a Transposed Convolution (Deconvolution) to map the feature space
      back to the spatial image space. Dynamically calculates output padding to reconstruct
      the original input shape.
    * **Compute P**: Uses `unfold` (im2col) to extract sliding local blocks from the image
      into a flat patch matrix for the weight update step.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int,
        p: int,
        s: int,
        h: nn.Module = None,
        pool_op: nn.Module = None,
        padding_mode: str = "circular",
        config: Optional[ADMM_LayerConfig] = None,
        rho: float = None,
        beta: float = None,
        deltas: float = None,
        thetas: float = None,
        use_reset: bool = None,
    ):
        super().__init__(
            h=h,
            pool_op=pool_op,
            padding_mode=padding_mode,
            rho=rho,
            beta=beta,
            deltas=deltas,
            thetas=thetas,
            use_reset=use_reset,
            config=config,
        )
        self.p = p
        self.s = s
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.channel_dim = -3  # Targets the [B, C, H, W] dimension
        self.padding_mode = padding_mode
        self.convolution = True
        if self.padding_mode == "circular" and not self.config.use_fft:
            warnings.warn(
                "Using circular padding without FFTs (use_fft=False) is highly inefficient. "
                "The spatial solver must construct a massive dense Gram matrix to compute the adjoint. "
                "Consider setting use_fft=True."
            )

    def _setup(self, global_config=None) -> None:
        """Initializes the 4D kernel weights and bias tensors."""
        super()._setup(global_config)
        if self.padding_mode == "circular" and self.s != 1:
            warnings.warn(
                f"Circular padding requires stride=1 to remain mathematically perfectly "
                f"equivalent in the Fourier domain. Changing stride from {self.s} to 1."
            )
            self.s = 1

        self._init_weights_and_bias(
            (self.out_c, self.in_c, self.k, self.k), (self.out_c,)
        )

    def spatial_forward(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        """Applies the spatial 2D convolution over the input images.

        Args:
            x (torch.Tensor): The input image tensor `[B, C, H, W]`.
            use_bias (bool, optional): Whether to apply the bias vector. Defaults to True.

        Returns:
            torch.Tensor: The convolved feature maps.
        """
        if self.padding_mode == "circular":
            return circular_forward(x, use_bias)
        b = (
            self.b
            if (isinstance(use_bias, bool) and use_bias and self.config.use_bias)
            else None
        )
        return torch.nn.functional.conv2d(
            x, self.W, bias=b, padding=self.p, stride=self.s
        )

    def adjoint_operator(
        self, target: torch.Tensor, original_input_shape: tuple = None
    ) -> torch.Tensor:
        """Maps errors back to the image domain via transposed convolutions.

        Args:
            target (torch.Tensor): The target tensor mapped in the output feature space.
            original_input_shape (tuple, optional): The exact `[B, C, H, W]` shape of the
                input, required to perfectly reconstruct edges when strides are used.

        Returns:
            torch.Tensor: The adjoint projection tensor in the input domain.
        """
        if self.padding_mode == "circular":
            return circular_adjoint(target)

        out_pad = (0, 0)
        if original_input_shape is not None:
            H_in, W_in = original_input_shape[-2:]
            H_dim, W_dim = target.shape[-2:]
            H_calc = (H_dim - 1) * self.s - 2 * self.p + self.k
            W_calc = (W_dim - 1) * self.s - 2 * self.p + self.k
            out_pad = (max(0, H_in - H_calc), max(0, W_in - W_calc))

        return torch.nn.functional.conv_transpose2d(
            target, self.W, padding=self.p, stride=self.s, output_padding=out_pad
        )

    def _compute_P(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Extracts image patches using unfold (im2col) for localized weight updates.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            torch.Tensor: The flattened patch matrix $P$.
        """
        if self.padding_mode == "circular":
            return circular_compute_P(a_prev)
        patches = torch.nn.functional.unfold(
            a_prev, kernel_size=self.k, padding=self.p, stride=self.s
        )
        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)

    def _get_bias_reduction_dims(self) -> Tuple[int, int, int]:
        """Returns the reduction dimension tuple for spatial bias averaging."""
        return (0, 2, 3)


class ADMM_SpikingLinear(ADMM_Spiking, ADMM_AffineLayer):
    r"""Spiking Fully Connected Layer.

    How it works:

    * Designed for 5D sequence tensors: `[Time, Batch, Features]`.
    * Folds the Time and Batch dimensions together to process the entire sequence as
      a standard 2D matrix operation efficiently, then unfolds it back to the temporal sequence.
    """

    def __init__(
        self,
        in_f: int,
        out_f: int,
        h: nn.Module = None,
        pool_op: nn.Module = None,
        rho: float = None,
        beta: float = None,
        deltas: float = None,
        thetas: float = None,
        use_reset: bool = None,
        config: Optional[ADMM_LayerConfig] = None,
    ):
        super().__init__(
            h=h,
            pool_op=pool_op,
            rho=rho,
            beta=beta,
            deltas=deltas,
            thetas=thetas,
            use_reset=use_reset,
            config=config,
        )
        self.T = None
        self.spiking = True
        self.use_reset = use_reset
        self.in_f = in_f
        self.out_f = out_f
        self.channel_dim = -1
        self.convolution = False

    def _setup(self, global_config=None) -> None:
        """Initializes the weight and bias tensors for the spiking linear layer."""
        super()._setup(global_config)
        self._init_weights_and_bias((self.out_f, self.in_f), (self.out_f,))

    def spatial_forward(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        """Folds the sequence and applies the fully connected transformation.

        Args:
            x (torch.Tensor): The 5D spiking input sequence `[T, B, Features]`.
            use_bias (bool, optional): Whether to apply the bias vector. Defaults to True.

        Returns:
            torch.Tensor: The transformed spiking output sequence.
        """
        b = (
            self.b
            if (isinstance(use_bias, bool) and use_bias and self.config.use_bias)
            else None
        )
        x_flat, tb_shape = fold_time(x)
        x_pooled = self.pool_op(x_flat)

        out_flat = torch.matmul(x_pooled, self.W.t())
        if b is not None:
            out_flat += b

        return unfold_time(out_flat, tb_shape)

    def adjoint_operator(
        self, target: torch.Tensor, original_input_shape: tuple = None
    ) -> torch.Tensor:
        """Computes the temporal adjoint sequence mapping errors backwards.

        Args:
            target (torch.Tensor): The sequential target/error tensor.
            original_input_shape (tuple, optional): Original geometry parameters. Defaults to None.

        Returns:
            torch.Tensor: The temporal adjoint response mapped to the input space.
        """
        target_flat, tb_shape = fold_time(target)
        deconv_flat = torch.matmul(target_flat, self.W)
        return self.pool_op.adjoint(deconv_flat, original_input_shape, tb_shape)

    def _compute_P(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Folds the sequential activations to retrieve the pooled patch matrix."""
        a_flat, _ = fold_time(a_prev)
        return self.pool_op(a_flat)

    def _get_bias_reduction_dims(self) -> Tuple[int, int]:
        """Returns the reduction dimensions tracking both batch and time indices."""
        return (0, 1)


class ADMM_SpikingConv2d(ADMM_Spiking, ADMM_AffineLayer):
    r"""Spiking 2D Convolutional Layer.

    How it works:

    * Designed for 5D spatiotemporal tensors: `[Time, Batch, Channels, Height, Width]`.
    * Uses time folding to apply standard 2D convolutions efficiently across all timesteps.
    * Includes a chunked covariance computation to prevent Out-Of-Memory (OOM) errors
      during the ADMM dense patch extraction step.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int,
        p: int,
        s: int,
        h: nn.Module = None,
        pool_op: nn.Module = None,
        padding_mode: str = "circular",
        config: Optional[ADMM_LayerConfig] = None,
        rho: float = None,
        beta: float = None,
        deltas: float = None,
        thetas: float = None,
        use_reset: bool = None,
    ):
        super().__init__(
            h=h,
            pool_op=pool_op,
            padding_mode=padding_mode,
            rho=rho,
            beta=beta,
            deltas=deltas,
            thetas=thetas,
            use_reset=use_reset,
            config=config,
        )
        self.T = None
        self.spiking = True
        self.use_reset = use_reset
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.p = p
        self.s = s
        self.channel_dim = -3
        self.convolution = True
        self.padding_mode = padding_mode
        if self.padding_mode == "circular" and not self.config.use_fft:
            warnings.warn(
                "Using circular padding without FFTs (use_fft=False) is highly inefficient. "
                "The spatial solver must construct a massive dense Gram matrix to compute the adjoint. "
                "Consider setting use_fft=True."
            )

    def _setup(self, global_config=None) -> None:
        """Initializes the 4D kernel weights and bias tensors for sequential data."""
        super()._setup(global_config)
        if self.padding_mode == "circular" and self.s != 1:
            warnings.warn(
                f"Circular padding requires stride=1 to remain mathematically perfectly "
                f"equivalent in the Fourier domain. Changing stride from {self.s} to 1."
            )
            self.s = 1
        self._init_weights_and_bias(
            (self.out_c, self.in_c, self.k, self.k), (self.out_c,)
        )

    def spatial_forward(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        """Applies spatial convolution frame-by-frame across the temporal sequence.

        Args:
            x (torch.Tensor): The 5D spiking input sequence `[T, B, C, H, W]`.
            use_bias (bool, optional): Whether to apply the bias vector. Defaults to True.

        Returns:
            torch.Tensor: The convolved spiking video/sequence.
        """
        if self.padding_mode == "circular":
            return circular_forward(x, use_bias)
        b = (
            self.b
            if (isinstance(use_bias, bool) and use_bias and self.config.use_bias)
            else None
        )
        x_flat, tb_shape = fold_time(x)
        out_flat = torch.nn.functional.conv2d(
            x_flat, self.W, bias=b, padding=self.p, stride=self.s
        )
        return unfold_time(out_flat, tb_shape)

    def adjoint_operator(
        self, target: torch.Tensor, original_input_shape: tuple = None
    ) -> torch.Tensor:
        """Maps sequential errors backwards using transposed convolutions frame-by-frame.

        Args:
            target (torch.Tensor): The spatiotemporal target/error tensor.
            original_input_shape (tuple, optional): The exact `[B, C, H, W]` spatial geometry
                required to correctly reconstruct strides. Defaults to None.

        Returns:
            torch.Tensor: The spatiotemporal adjoint response.
        """
        if self.padding_mode == "circular":
            return circular_adjoint(target)
        target_flat, tb_shape = fold_time(target)
        out_pad = (0, 0)
        if original_input_shape is not None:
            H_in, W_in = original_input_shape[-2:]
            H_dim, W_dim = target_flat.shape[-2:]
            out_pad = (
                max(0, H_in - ((H_dim - 1) * self.s - 2 * self.p + self.k)),
                max(0, W_in - ((W_dim - 1) * self.s - 2 * self.p + self.k)),
            )

        out_flat = torch.nn.functional.conv_transpose2d(
            target_flat, self.W, padding=self.p, stride=self.s, output_padding=out_pad
        )
        return unfold_time(out_flat, tb_shape)

    def _compute_P(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Extracts and concatenates spatial patches sequentially using im2col."""
        if self.padding_mode == "circular":
            return circular_compute_P(a_prev)
        a_flat, _ = fold_time(a_prev)
        patches = torch.nn.functional.unfold(
            a_flat, kernel_size=self.k, padding=self.p, stride=self.s
        )
        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)

    def _get_bias_reduction_dims(self) -> Tuple[int, int, int, int]:
        """Returns the reduction dimensions bounding temporal, batch, and spatial traits."""
        return (0, 1, 3, 4)
