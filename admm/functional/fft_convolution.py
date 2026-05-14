r"""
Fast Fourier Transform (FFT) accelerated operations for
convolutional layers.

By leveraging the Convolution Theorem, it performs heavy matrix inversions
and spatial adjoint operations in the frequency domain, bypassing $O(N^3)$
bottlenecks and drastically improving numerical performance.
"""

from typing import Tuple, Union

import torch
import torch.fft
import torch.nn.functional as F

from ..dataclasses import ADMM_LayerConfig

# ==============================================================================================
# SPATIAL OPERATIONS (Forward & Adjoint)
# ==============================================================================================


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
        self._fold_time(spatial_input) if is_spiking_sequence else (spatial_input, None)
    )

    padded_input = F.pad(
        folded_sequence, (self.p, self.p, self.p, self.p), mode="circular"
    )
    spatial_output = F.conv2d(
        padded_input, self.W, bias=bias_vector, padding=0, stride=self.s
    )

    # Unfold back to [Time, Batch, ...] if necessary
    if is_spiking_sequence:
        return self._unfold_time(spatial_output, time_batch_shape)
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
        self._fold_time(target) if is_spiking_sequence else (target, None)
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
        return self._unfold_time(adjoint_spatial, time_batch_shape)
    return adjoint_spatial


# ==============================================================================================
# COVARIANCE & PATCH EXTRACTION
# ==============================================================================================
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
    folded_activations = self._fold_time(a_prev)[0] if is_spiking_sequence else a_prev

    padded_activations = F.pad(
        folded_activations, (self.p, self.p, self.p, self.p), mode="circular"
    )
    patches = F.unfold(padded_activations, kernel_size=self.k, padding=0, stride=self.s)

    return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)


def get_a_denominator(
    self, config_prev: ADMM_LayerConfig, W_expanded: torch.Tensor, a_shape: tuple
) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict], int]:
    # ... rest of the function remains the same
    r"""Computes the non spiking denominator matrix for the activation ($a$) update.
    Reroutes to specialized solvers if necessary.

    Formula evaluated: $D = \beta_l I + \rho_{l+1} \mathcal{A}_{l+1}^* \circ \mathcal{A}_{l+1}$

    You can find the corresponding spiking version [get_spiking_a_denominator][admm.temporal_helpers.get_spiking_a_denominator] in the [Temporal Helpers][admm.temporal_helpers] module.

    Args:
        beta_current (float): The penalty parameter $\beta$ for the update.
        a_shape (tuple): The physical geometry shape of the activation tensor.

    Returns:
        tuple:
            - Union[torch.Tensor, dict]: The computed denominator matrix (or Woodbury params).
            - Union[torch.Tensor, dict]: The last-step denominator matrix (or Woodbury params).
            - int: The number of input features.
    """
    # --- FFT Accelerated Route ---
    height, width = a_shape[-2:]
    _, in_channels, kernel_h, kernel_w = W_expanded.shape

    pad_h = height - kernel_h
    pad_w = width - kernel_w
    padded_kernel = F.pad(self.W, (0, pad_w, 0, pad_h))

    kernel_fft = torch.fft.fft2(padded_kernel)
    kernel_fft = kernel_fft.permute(2, 3, 0, 1)  # Shape: [H, W, C_out, C_in]

    # Hermitian Transpose: conj(W)^T @ W = W^H W
    denominator = torch.matmul(kernel_fft.conj().transpose(-2, -1), kernel_fft)

    denominator.mul_(self.config.rho)
    denominator.diagonal(dim1=-2, dim2=-1).add_(config_prev.beta)

    return denominator, denominator, in_channels


def compute_batch_covariances(
    self, Y_target: torch.Tensor, a_prev: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the numerator and denominator for the weight update.

    Overrides the base method to compute covariances in chunks. 5D Spiking sequences
    can easily cause Out Of Memory (OOM) errors if im2col is run on the entire video
    at once. This iterates over time steps and accumulates the results securely.

    Args:
        Y_target (torch.Tensor): The target spatial tensor.
        a_prev (torch.Tensor): The previous layer's activations.

    Returns:
        tuple:
            - torch.Tensor: The computed numerator matrix ($Y^T P$).
            - torch.Tensor: The computed denominator matrix ($P^T P$).
    """
    if a_prev.dim() == 4:
        a_prev = a_prev.unsqueeze(0)  # Convert [Batch, C, H, W] to [1, Batch, C, H, W]
        Y_target = Y_target.unsqueeze(0)

    time_steps = a_prev.shape[0]
    patch_dim = self.in_c * self.k * self.k
    out_channels = self.W.shape[0]

    # Initialize accumulation matrices
    denominator = torch.zeros(
        (patch_dim, patch_dim), device=a_prev.device, dtype=a_prev.dtype
    )
    numerator = torch.zeros(
        (out_channels, patch_dim), device=Y_target.device, dtype=Y_target.dtype
    )

    # Accumulate over time to prevent memory blowouts
    for t in range(time_steps):
        if getattr(self, "padding_mode", "zeros") == "circular":
            a_padded = F.pad(
                a_prev[t], (self.p, self.p, self.p, self.p), mode="circular"
            )
            patch_matrix_t = F.unfold(
                a_padded, kernel_size=self.k, padding=0, stride=self.s
            )
        else:
            patch_matrix_t = F.unfold(
                a_prev[t], kernel_size=self.k, padding=self.p, stride=self.s
            )

        # Flatten patch matrix and target for matrix multiplication
        patch_matrix_t = patch_matrix_t.transpose(1, 2).reshape(-1, patch_dim)
        target_flattened = (
            Y_target[t].movedim(self.channel_dim, -1).reshape(-1, out_channels)
        )

        # Accumulate covariances
        denominator += patch_matrix_t.t() @ patch_matrix_t
        numerator += target_flattened.t() @ patch_matrix_t

    return numerator, denominator


# ==============================================================================================
# FREQUENCY DOMAIN SOLVERS
# ==============================================================================================


def _solve_activation_system(
    self,
    numerator: torch.Tensor,
    denominator_main: torch.Tensor,
    denominator_last: torch.Tensor,
    a_shape: tuple,
    in_features: int,
) -> torch.Tensor:
    r"""Universal Solver for the activation ($a$) update in the frequency domain.

    Solves the linear system: $A x = B$ in Fourier space.

    **Important Constraint:** Because our FFT Gram matrix is complex Hermitian ($A \neq A^T$), we must force
    PyTorch to right-multiply ($A^T x = b$) using `.mT` to match the exact behavior
    of the dense row-vector solver.

    Args:
        numerator (torch.Tensor): The precomputed numerator tensor.
        denominator_main (torch.Tensor): The FFT LHS matrix for time $t < T$.
        denominator_last (torch.Tensor): The FFT LHS matrix for time $t = T$.
        a_shape (tuple): The physical shape of the activation tensor.
        in_features (int): Number of input channels.

    Returns:
        torch.Tensor: The exact updated activations mapped back to the spatial domain.
    """

    original_dim = numerator.dim()
    if original_dim == 4:
        numerator = numerator.unsqueeze(0)  # Standardize to 5D

    # Convert numerator to frequency domain and reshape for broadcasting
    numerator_fft = torch.fft.fft2(numerator).permute(0, 1, 3, 4, 2).unsqueeze(-1)

    # Prepare LHS matrices for broadcasting
    A_main = denominator_main.unsqueeze(0).unsqueeze(0)
    A_last = denominator_last.unsqueeze(0).unsqueeze(0)

    # Force the solver to solve A^T x = b (Using .mT)
    activations_fft_main = torch.linalg.solve(A_main.mT, numerator_fft[:-1])
    activations_fft_last = torch.linalg.solve(A_last.mT, numerator_fft[-1:])

    # Combine sequence and reshape back to [Time, Batch, C, H, W]
    activations_fft = torch.cat([activations_fft_main, activations_fft_last], dim=0)
    activations_fft = activations_fft.squeeze(-1).permute(0, 1, 4, 2, 3)

    # Return to spatial reality
    activations_spatial = torch.fft.ifft2(activations_fft).real

    return activations_spatial.squeeze(0) if original_dim == 4 else activations_spatial


def _solve_activation_system_unrolled(
    self, numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    r"""FFT-based unrolled step solver for sequences.

    Accepts a **pre-inverted** per-frequency-bin denominator (shape `[H, W, C, C]`)
    and performs a batched matrix-vector multiplication directly in the frequency domain.

    Args:
        numerator (torch.Tensor): The numerator vector for the current timestep,
            shape `[Batch, C, H, W]`.
        denominator (torch.Tensor): The pre-inverted matrix `inv(D)`,
            shape `[H, W, C, C]`.

    Returns:
        torch.Tensor: The updated activation tensor for the current timestep in
            the spatial domain, shape `[Batch, C, H, W]`.
    """
    if not self.config.use_fft:
        return super()._solve_activation_system_unrolled(numerator, denominator)

    numerator_fft = torch.fft.fft2(numerator)

    # Permute to [H, W, Batch, C] to isolate the channel vector per spatial bin
    numerator_fft = numerator_fft.permute(2, 3, 0, 1)

    # Broadcast denominator over the Batch dimension -> [H, W, 1, C, C]
    denominator_expanded = denominator.unsqueeze(2)

    # Row convention multiplication: n_hat[..., None, :] @ inv(D)
    activations_fft = torch.matmul(numerator_fft.unsqueeze(-2), denominator_expanded)

    # Restore standard dimensions -> [Batch, C, H, W]
    activations_fft = activations_fft.squeeze(-2).permute(2, 3, 0, 1)

    return torch.fft.ifft2(activations_fft).real
