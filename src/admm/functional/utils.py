"""Utility functions for tensor manipulation, validation and temporal dependency calculations.

Provides helper methods for broadcasting, temporal dimension folding and unfolding, asessing woodbury necessity and
evaluating layer state dynamics within the ADMM optimization framework.
"""

import torch

from ..dataclasses import ADMM_LayerConfig, ADMM_LayerState


def broadcast_to_match(
    tensor: torch.Tensor, target_tensor: torch.Tensor
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


def fold_time(x: torch.Tensor) -> tuple[torch.Tensor, tuple | None]:
    """Folds the Time and Batch dimensions together for spatial operations.

    Used to seamlessly pass spiking sequences through standard 2D convolutions
    and feedforward matrix multiplications.

    Args:
        x (torch.Tensor): The input tensor, potentially 5D (Conv) or 3D (Feedforward).

    Returns:
        Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
            - torch.Tensor: The flattened tensor ready for spatial operations.
            - Optional[Tuple[int, int]]: The original (Time, Batch) shape to reconstruct
              the tensor later. Returns None if the input lacks a temporal dimension.
    """
    if x.dim() in [3, 5]:
        return x.reshape(x.size(0) * x.size(1), *x.shape[2:]), x.shape[:2]
    return x, None


def unfold_time(x_flat: torch.Tensor, tb_shape: tuple) -> torch.Tensor:
    """Unfolds the Time and Batch dimensions back out after spatial operations.

    Args:
        x_flat (torch.Tensor): The spatially processed flat tensor.
        tb_shape (Optional[Tuple[int, int]]): The (Time, Batch) shape originally
            returned by fold_time.

    Returns:
        torch.Tensor: The reconstructed temporal sequence tensor.
    """
    if tb_shape is None:
        return x_flat
    return x_flat.reshape(tb_shape[0], tb_shape[1], *x_flat.shape[1:])


def should_use_woodbury(W: torch.Tensor, W_expanded: torch.Tensor) -> bool:
    """Determines if the Woodbury identity is more efficient based on matrix dimensions.

    Args:
        W (torch.Tensor): The standard weight matrix.
        W_expanded (torch.Tensor): The expanded weight matrix.

    Returns:
        bool: True if the number of input features exceeds the output features, making
        Woodbury inversion more computationally efficient.
    """
    if W is None or W.dim() != 2:
        return False
    out_features, in_features = W_expanded.shape
    return in_features > out_features


def compute_temporal_dependencies(
    state: ADMM_LayerState,
    config: ADMM_LayerConfig,
) -> torch.Tensor:
    r"""Computes the physical voltage leakage and threshold reset over time.

    Formula evaluated:

    $$ T_{l} = \begin{cases} \delta S z_l - \theta S a_{l}, & \text{if} \quad l < L, \cr \delta S z_L, & \text{if}\quad  l = L.\end{cases} $$

    Args:
        state (ADMM_LayerState): The [layer state object][src.admm.dataclasses.ADMM_LayerState].
        config (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] for the layer.

    Returns:
        torch.Tensor: The calculated temporal dependencies tensor.
    """
    out = torch.empty_like(state.z)
    out[0].zero_()
    out_slice = out[1:]
    out_slice.copy_(state.z[:-1])
    out_slice.mul_(config.deltas)
    if config.use_reset and state.a is not None:
        out[1:].add_(state.a[:-1], alpha=-config.thetas)

    return out
