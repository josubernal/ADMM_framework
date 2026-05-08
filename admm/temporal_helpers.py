"""
This module isolates the stateless mathematical operations required for
Spiking Neural Networks (SNNs), helping to maintain a clean, modular, and
easily testable codebase.
"""

import torch


def fold_time(x: torch.Tensor) -> tuple[torch.Tensor, tuple | None]:
    """Folds the Time and Batch dimensions together for spatial operations.

    Used to seamlessly pass spiking sequences through standard 2D convolutions
    and Linear matrix multiplications.

    Args:
        x (torch.Tensor): The input tensor, potentially 5D (Conv) or 3D (Linear).

    Returns:
        tuple:
            - torch.Tensor: The flattened tensor ready for spatial operations.
            - tuple or None: The original `(Time, Batch)` shape to reconstruct the tensor later.
    """
    if x.dim() in [3, 5]:
        return x.reshape(x.size(0) * x.size(1), *x.shape[2:]), x.shape[:2]
    return x, None


def unfold_time(x_flat: torch.Tensor, tb_shape: tuple) -> torch.Tensor:
    """Unfolds the Time and Batch dimensions back out after spatial operations.

    Args:
        x_flat (torch.Tensor): The spatially processed flat tensor.
        tb_shape (tuple): The `(Time, Batch)` shape originally returned by `fold_time`.

    Returns:
        torch.Tensor: The reconstructed temporal sequence tensor.
    """
    if tb_shape is None:
        return x_flat
    return x_flat.reshape(tb_shape[0], tb_shape[1], *x_flat.shape[1:])


def compute_temporal_dependencies(
    z: torch.Tensor,
    a: torch.Tensor,
    deltas: float,
    thetas: float,
    use_reset: bool = True,
    include_reset: bool = True,
) -> torch.Tensor:
    r"""Computes the physical voltage leakage and threshold reset over time.

    Formula evaluated: $\Delta z_t = \delta * z_{t-1} - \theta * a_{t-1}$

    Args:
        z (torch.Tensor): The membrane potential tensor.
        a (torch.Tensor): The spike activation tensor.
        deltas (float): The temporal leakage hyperparameter.
        thetas (float): The spiking threshold hyperparameter.
        use_reset (bool, optional): Whether the layer is configured to use voltage reset. Defaults to True.
        include_reset (bool, optional): Whether to factor the reset penalty into the current calculation. Defaults to True.

    Returns:
        torch.Tensor: The calculated temporal dependencies tensor.
    """
    out = torch.empty_like(z)
    out[0].zero_()
    out_slice = out[1:]
    out_slice.copy_(z[:-1])
    out_slice.mul_(deltas)
    if use_reset and include_reset and a is not None:
        out[1:].add_(a[:-1], alpha=-thetas)

    return out


def get_spiking_v(
    z: torch.Tensor,
    bias: torch.Tensor,
    temporal_dependencies: torch.Tensor,
    rho: float,
    lambda_lagrange: torch.Tensor = None,
    broadcast_func=None,
) -> torch.Tensor:
    r"""Returns the spiking target ($v$), incorporating temporal leakage and reset penalties.

    This target is utilized primarily during the weight update phase to determine
    the ideal spatial trajectory of the network.

    Args:
        z (torch.Tensor): The membrane potential tensor.
        bias (torch.Tensor or int): The reshaped bias tensor (or 0 if unused).
        temporal_dependencies (torch.Tensor): The precomputed leakage/reset tensor.
        rho (float): The spatial affine penalty parameter.
        lambda_lagrange (torch.Tensor, optional): The dual variable tensor. Defaults to None.
        broadcast_func (callable, optional): Helper function to align dimensions. Defaults to None.

    Returns:
        torch.Tensor: The calculated spiking target tensor $v$.
    """
    v = z.clone()
    if isinstance(bias, torch.Tensor):
        v.sub_(bias)
    v.sub_(temporal_dependencies)
    if lambda_lagrange is not None and broadcast_func is not None:
        lam_sp = broadcast_func(lambda_lagrange, z[-1])
        v[-1].add_(lam_sp, alpha=1.0 / rho)
    return v


def get_spiking_a_denominator(
    WtW: torch.Tensor,
    in_features: int,
    beta_current: float,
    rho_next: float,
    temporal_penalty: float,
    unrolled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Computes the denominator (LHS) matrices for the activation ($a$) update step.

    For spiking networks, the system matrix varies depending on the time step due
    to recurrent dependencies.

    Formula evaluated:

     - For $t < T$: $A_{main} = \beta I + \rho W^TW + \rho \theta^2 I$
     - For $t = T$: $A_{last} = \beta I + \rho W^TW$

    Args:
        WtW (torch.Tensor): The computed $W^T W$ covariance matrix of the next layer.
        in_features (int): The number of input features.
        beta_current (float): The $\beta$ penalty parameter of the current layer.
        rho_next (float): The $\rho$ penalty parameter of the next layer.
        temporal_penalty (float): Precomputed penalty constraint ($\rho * \theta^2$).
        unrolled (bool, optional): If True, computes and transposes the inverse directly. Defaults to False.

    Returns:
        tuple:
            - torch.Tensor: The main denominator matrix (for $t < T$).
            - torch.Tensor: The final denominator matrix (for $t = T$).
            - int: The tracked input features count.
    """

    denominator_last = WtW * rho_next

    denominator_last.diagonal().add_(beta_current)

    denominator_main = denominator_last.clone()
    denominator_main.diagonal().add_(temporal_penalty)

    if unrolled:
        denominator_main = torch.linalg.inv(denominator_main).transpose(-2, -1)
        denominator_last = torch.linalg.inv(denominator_last).transpose(-2, -1)

    return denominator_main, denominator_last, in_features


def get_spiking_a_adjoint(
    layer: torch.nn.Module, next_layer: torch.nn.Module
) -> torch.Tensor:
    r"""Computes the spatial linear components of the ADMM numerator for activation updates.

    Maps the errors of the subsequent layer backwards through its spatial weights
    using the transposed adjoint operator.

    Args:
        layer (ADMM_Layer): The current layer being updated.
        next_layer (ADMM_Layer): The subsequent layer in the network architecture.

    Returns:
        torch.Tensor: The resulting adjoint tensor shaped to match `layer.a`.
    """
    v = next_layer._get_v(include_reset=False)
    adjoint = next_layer.adjoint_operator(v, original_input_shape=layer.a.shape)
    # temporal_penalty = torch.zeros_like(layer.z)
    # temporal_penalty[:-1] = -layer.thetas * layer.rho * (layer.z[1:] - layer.deltas * layer.z[:-1] - forward_pass[1:]) DEPRECATED
    return adjoint.mul_(next_layer.config.rho)  # + temporal_penalty
