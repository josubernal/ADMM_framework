r"""
Fast Fourier Transform (FFT) circular operations for
convolutional layers.

To utilize the benefits of the Convolution Theorem, bypassing $O(N^3)$
bottlenecks, we need to utilize circular convolution. These file defines the
circular convolution operators for [convolutional layers][src.admm.layers], making layers more readable
and easier to implement.
"""

import torch
import torch.fft
import torch.nn.functional as F

from .utils import fold_time, unfold_time


def circular_forward(
    self, spatial_input: torch.Tensor, use_bias: bool = True
) -> torch.Tensor:
    r"""Applies a circular padded convolution.

    Args:
        spatial_input (torch.Tensor): The input tensor (can be 4D static or 5D spiking).
        use_bias (bool, optional): Whether to apply the layer's bias. Defaults to True.

    Returns:
        torch.Tensor: The convoluted output feature map.
    """
    bias_vector = (
        getattr(self, "b", None)
        if (isinstance(use_bias, bool) and use_bias and getattr(self, "bias", False))
        else None
    )

    is_spiking_sequence = spatial_input.dim() == 5

    # If spiking, collapse [Time, Batch] into a single dimension so Conv2D can process it
    folded_sequence, time_batch_shape = (
        fold_time(spatial_input) if is_spiking_sequence else (spatial_input, None)
    )

    padded_input = F.pad(
        folded_sequence, (self.p, self.p, self.p, self.p), mode="circular"
    )
    spatial_output = F.conv2d(
        padded_input, self.W, bias=bias_vector, padding=0, stride=self.s
    )

    # Unfold back to [Time, Batch, ...] if necessary
    if is_spiking_sequence:
        return unfold_time(spatial_output, time_batch_shape)
    return spatial_output


def circular_adjoint(self, target: torch.Tensor) -> torch.Tensor:
    r"""Computes the exact mathematical transpose (adjoint) using FFTs.

    In the frequency domain, cross-correlation (PyTorch's Conv2D) is equivalent
    to multiplying by the complex conjugate of the frequency filter. To compute the
    adjoint, we must spatially reverse (roll) the kernel.

    Args:
        target (torch.Tensor): The error/target tensor mapped in the output space.

    Returns:
        torch.Tensor: The adjoint projection tensor in the input domain.
    """
    is_spiking_sequence = target.dim() == 5
    folded_target, time_batch_shape = (
        fold_time(target) if is_spiking_sequence else (target, None)
    )

    height, width = folded_target.shape[-2:]
    pad_h = height - self.k
    pad_w = width - self.k

    # Pad the kernel to match the image dimensions, then roll (shift) it
    # so the center of the kernel aligns with the origin (0,0) in Fourier space.
    padded_kernel = F.pad(self.W, (0, pad_w, 0, pad_h))
    rolled_kernel = torch.roll(
        padded_kernel, shifts=(-(self.k // 2), -(self.k // 2)), dims=(-2, -1)
    )

    # Transform both to frequency domain
    kernel_fft = torch.fft.fft2(rolled_kernel)
    target_fft = torch.fft.fft2(folded_target)

    # Compute the adjoint multiplication per frequency bin
    adjoint_fft = torch.einsum("bohw,oihw->bihw", target_fft, kernel_fft)

    # Transform back to spatial domain
    adjoint_spatial = torch.fft.ifft2(adjoint_fft).real

    if is_spiking_sequence:
        return unfold_time(adjoint_spatial, time_batch_shape)
    return adjoint_spatial


def circular_compute_P(self, a_prev: torch.Tensor) -> torch.Tensor:
    r"""Extracts circular padded image patches for the weight update step.

    Uses `unfold` (im2col) to extract sliding local blocks from the image
    into a flattened patch matrix $P$.

    Args:
        a_prev (torch.Tensor): The previous layer's activations.

    Returns:
        torch.Tensor: The flattened patch matrix $P$.
    """
    is_spiking_sequence = a_prev.dim() == 5
    folded_activations = fold_time(a_prev)[0] if is_spiking_sequence else a_prev

    padded_activations = F.pad(
        folded_activations, (self.p, self.p, self.p, self.p), mode="circular"
    )
    patches = F.unfold(padded_activations, kernel_size=self.k, padding=0, stride=self.s)

    return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)
