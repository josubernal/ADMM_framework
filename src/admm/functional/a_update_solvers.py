"""This module isolates the numerical linear algebra required for the ADMM a-updates.

By keeping these functions stateless, they are easier to optimize, maintain,
and read independently from the layer mechanics.
"""

from typing import Union

import torch


def solve_fft_system_static(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """Static solver for the activation ($a$) update in the frequency domain.

    Converts the numerator to the frequency domain and solves the linear system
    using the provided main and last denominator tensors.

    Args:
        numerator (torch.Tensor): The numerator tensor.
        denominator (torch.Tensor): The denominator matrix.

    Returns:
        torch.Tensor: The updated activation tensor in the spatial domain.
    """

    numerator_fft = torch.fft.fft2(numerator).permute(0, 2, 3, 1).unsqueeze(-1)
    A_fft = denominator.unsqueeze(0)
    activations_fft = torch.linalg.solve(A_fft.mT, numerator_fft)
    activations_fft = activations_fft.squeeze(-1).permute(0, 3, 1, 2)
    return torch.fft.ifft2(activations_fft).real


def solve_fft_system_spiking(
    numerator: torch.Tensor,
    denominator_main: torch.Tensor,
    denominator_last: torch.Tensor,
) -> torch.Tensor:
    """Spiking solver for the activation ($a$) update in the frequency domain.

    Converts the numerator to the frequency domain and solves the linear system
    using the provided main and last denominator tensors.

    Args:
        numerator (torch.Tensor): The numerator tensor, potentially 4D or 5D.
        denominator_main (torch.Tensor): The denominator matrix for the main timesteps.
        denominator_last (torch.Tensor): The denominator matrix for the final boundary timestep.

    Returns:
        torch.Tensor: The updated activation tensor in the spatial domain.
    """

    # Convert numerator to frequency domain and reshape for broadcasting
    numerator_fft = torch.fft.fft2(numerator).permute(0, 1, 3, 4, 2).unsqueeze(-1)

    denominator_main = denominator_main.unsqueeze(0).unsqueeze(0)
    denominator_last = denominator_last.unsqueeze(0).unsqueeze(0)

    # Force the solver to solve denominator^T x = b (Using .mT)
    activations_fft_main = torch.linalg.solve(denominator_main.mT, numerator_fft[:-1])
    activations_fft_last = torch.linalg.solve(denominator_last.mT, numerator_fft[-1:])

    activations_fft = torch.cat([activations_fft_main, activations_fft_last], dim=0)
    activations_fft = activations_fft.squeeze(-1).permute(0, 1, 4, 2, 3)

    activations_spatial = torch.fft.ifft2(activations_fft).real

    return torch.clamp(activations_spatial.squeeze(0), min=0.0, max=1.0)


def solve_fft_system_unrolled(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """Unrolled spiking solver for the activation ($a$) update in the frequency domain.

    Converts the numerator to the frequency domain and solves the linear system
    using the provided main and last denominator tensors.

    Args:
        numerator (torch.Tensor): The numerator tensor.
        denominator (torch.Tensor): The denominator matrix already inverted.

    Returns:
        torch.Tensor: The updated activation tensor in the spatial domain.
    """
    numerator_fft = torch.fft.fft2(numerator)
    # Permute to [H, W, Batch, C] to isolate the channel vector per spatial bin
    numerator_fft = numerator_fft.permute(2, 3, 0, 1)
    # Broadcast denominator over the Batch dimension -> [H, W, 1, C, C]
    denominator_expanded = denominator.unsqueeze(2)
    activations_fft = torch.matmul(numerator_fft.unsqueeze(-2), denominator_expanded)
    activations_fft = activations_fft.squeeze(-2).permute(2, 3, 0, 1)

    return torch.fft.ifft2(activations_fft).real


def solve_woodbury_system_static(
    W_expanded: torch.Tensor,
    numerator: torch.Tensor,
    beta: float,
    rho: float,
    a_shape: tuple,
) -> torch.Tensor:
    r"""Solves the system $(\beta_l  I_N + \rho_{l+1} \mathcal{A}_{l+1}^* \circ \mathcal{A}_{l+1}) x = numerator$ using the Woodbury Matrix Identity.

    Inverting a massive $N \times N$ matrix takes $O(N^3)$ memory and time. Woodbury allows us
    to instead invert a tiny $M \times M$ matrix, dropping complexity to $O(M^3)$.

    Mathematical Derivation:

    * Standard Woodbury Formula: $(A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}$
    * Our Substitutions:

        - $A = \beta  I_N$,
        - $U = W^T$,
        - $C = \rho I_M$,
        - $V = W$

    * The Simplified Result applied to the numerator:

    $$ x = \frac{1}{\beta}\text{numerator} - \frac{\rho}{\beta} W^T (\beta I_M + \rho W W^T)^{-1} W \text{numerator} $$

    Args:
        W_expanded (torch.Tensor): Weight matrix of shape [out_features, in_features].
        numerator (torch.Tensor): Target tensor to solve against.
        beta (float): Penalty parameter for the activation constraint.
        rho (float): Penalty parameter for the affine constraint.
        a_shape (Tuple[int, ...]): The original geometric shape of the activation tensor
            (e.g., [Batch, Features]).

    Returns:
        torch.Tensor: The solved activations x, reshaped back to a_shape.
    """
    out_features, in_features = W_expanded.shape
    numerator_flat = numerator.reshape(-1, in_features).t()

    # parenthesis = (β_l I + ρ_{l+1}WW^T)
    parenthesis = torch.matmul(W_expanded, W_expanded.t())
    parenthesis.mul_(rho)
    parenthesis.diagonal().add_(beta)

    # (parenthesis)^{-1}W*numerator
    W_num = torch.matmul(W_expanded, numerator_flat)
    parenthesis_inv_W_num = torch.linalg.solve(parenthesis, W_num)

    # term2 = W^T(parenthesis)^{-1}WB
    term2 = torch.matmul(W_expanded.t(), parenthesis_inv_W_num)

    # x= 1/β_l I - ρ_{l+1}/β_l term2
    x_flat = numerator_flat.div(beta)
    x_flat.sub_(term2, alpha=(rho / beta))

    return x_flat.t().view(a_shape)


def solve_woodbury_system_spiking(
    W_expanded: torch.Tensor,
    numerator: torch.Tensor,
    beta: float,
    rho: float,
    temporal_penalty: float,
    a_shape: tuple,
    T: int,
) -> torch.Tensor:
    """Solves the activation system for spiking sequences using Woodbury.

    Handles the temporal dimension by solving the main timesteps (t < T) and the final
    boundary timestep (t = T) separately due to differing penalty formulations.

    Args:
        W_expanded (torch.Tensor): Weight matrix of shape [out_features, in_features].
        numerator (torch.Tensor): Target tensor to solve against.
        beta (float): Penalty parameter for the activation constraint.
        rho (float): Penalty parameter for the affine constraint.
        temporal_penalty (float): Additional penalty applied to non-boundary timesteps.
        a_shape (tuple): The geometric shape of the activation tensor.
        T (int): Total number of timesteps.

    Returns:
        torch.Tensor: The solved activations x, reshaped and concatenated across time.
    """
    beta_main = beta + temporal_penalty  # for t < T
    beta_last = beta  # for t = T

    # Solve the last timestep
    new_a_last = solve_woodbury_system_static(
        numerator=numerator[-1:],
        W_expanded=W_expanded,
        beta=beta_last,
        rho=rho,
        a_shape=(1, *a_shape[1:]),
    )
    # Solve the main timesteps
    if T > 1:
        new_a_main = solve_woodbury_system_static(
            numerator=numerator[:-1],
            W_expanded=W_expanded,
            beta=beta_main,
            rho=rho,
            a_shape=(a_shape[0] - 1, *a_shape[1:]),
        )

    else:
        new_a_main = torch.empty((0, *a_shape[1:]), device=numerator.device)

    return torch.clamp(torch.cat([new_a_main, new_a_last], dim=0), min=0.0, max=1.0)


def solve_woodbury_system_unrolled(
    numerator: torch.Tensor,
    W_expanded: torch.Tensor,
    inv_denominator: torch.Tensor,
    beta_effective: float,
    rho: float,
    a_shape: tuple,
) -> torch.Tensor:
    r"""Executes the unrolled Woodbury step using a pre-inverted cached matrix.

    Formula evaluated:

    $$ x = \frac{1}{\beta_{eff}} numerator - \frac{\rho}{\beta_{eff}} W^T \cdot \text{inv_denominator} \cdot (W numerator) $$

    Args:
        numerator (torch.Tensor): Target tensor to solve against ($u$).
        W_expanded (torch.Tensor): Expanded weight matrix of shape `[M, N]`.
        inv_denominator (torch.Tensor): Cached inverse of $(\beta_{eff} I_M + \rho W W^T)$.
        beta_effective (float): The effective activation penalty for the timestep.
        rho (float): Penalty parameter for the affine constraint.
        a_shape (tuple): The original geometric shape of the activation tensor.

    Returns:
        torch.Tensor: The solved activations $x$, reshaped back to `a_shape`.
    """
    out_features, in_features = W_expanded.shape
    numerator_flat = numerator.reshape(-1, in_features).t()

    # 1. W * numerator
    WB = torch.matmul(W_expanded, numerator_flat)

    # 2. inv_denominator * (W * numerator)
    inv_WB = torch.matmul(inv_denominator, WB)

    # 3. W^T * inv_denominator * W * numerator
    term2 = torch.matmul(W_expanded.t(), inv_WB)

    # 4. (1/beta) * numerator - (rho/beta) * term2
    x_flat = numerator_flat.div(beta_effective)
    x_flat.sub_(term2, alpha=(rho / beta_effective))

    return x_flat.t().view(a_shape)


def solve_standard_system_static(
    denominator: torch.Tensor, numerator: torch.Tensor, a_shape: tuple, in_features: int
) -> torch.Tensor:
    """Solves a standard linear system $Ax = B$.

    Args:
        denominator (torch.Tensor): The left-hand side matrix.
        numerator (torch.Tensor): The right-hand side tensor.
        a_shape (tuple): The desired shape of the output tensor.
        in_features (int): The number of input features used to reshape the solver output.

    Returns:
        torch.Tensor: The solved system reshaped to match a_shape.
    """
    numerator_flat = numerator.reshape(-1, in_features)
    x_flat = torch.linalg.solve(denominator, numerator_flat.t()).t()
    return x_flat.view(a_shape)


def solve_standard_system_spiking(
    denominator_main: torch.Tensor,
    denominator_last: torch.Tensor,
    numerator: torch.Tensor,
    a_shape: tuple,
    in_features: int,
    T: int,
) -> torch.Tensor:
    r"""Solves the linear system $Ax = B$ for spiking sequences.

    Handles the boundary condition at the final timestep ($t=T$) separately
    from the main timesteps ($t<T$). We do this because the denominators are different.

    Args:
        denominator_main (torch.Tensor): The left-hand side matrix for the main timesteps.
        denominator_last (torch.Tensor): The left-hand side matrix for the final boundary timestep.
        numerator (torch.Tensor): The right-hand side tensor.
        a_shape (tuple): The desired output shape for the spatial dimensions.
        in_features (int): The number of input features.
        T (int): The total number of timesteps in the sequence.

    Returns:
        torch.Tensor: The concatenated solved activations across all timesteps.
    """

    # Solve the last timestep
    new_a_last = solve_standard_system_static(
        denominator_last, numerator[-1], (1, *a_shape[1:]), in_features
    )

    # Solve the main timesteps
    if T > 1:
        new_a_main = solve_standard_system_static(
            denominator_main, numerator[:-1], (T - 1, *a_shape[1:]), in_features
        )
    else:
        new_a_main = torch.empty((0, *a_shape[1:]), device=numerator.device)

    return torch.clamp(torch.cat([new_a_main, new_a_last], dim=0), min=0.0, max=1.0)


def solve_standard_system_unrolled(
    numerator: torch.Tensor, denominator: Union[torch.Tensor, dict]
) -> torch.Tensor:
    """Solves a standard linear system  $Ax = B$ for spiking unrolled path.

    Args:
        numerator (torch.Tensor): The numerator for a single timestep $t$.
        denominator (torch.Tensor): The pre-inverted matrix $D^{-1}$, already transposed
            to match the row-vector convention used by the dense solver.

    Returns:
        torch.Tensor: The solved activation for timestep $t$.
    """
    original_shape = numerator.shape
    in_features = denominator.size(1)
    numerator_flat = numerator.reshape(-1, in_features)
    a_t_flat = torch.matmul(numerator_flat, denominator)
    return a_t_flat.view(original_shape)
