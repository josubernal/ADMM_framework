"""
ADMM Solvers Module

This module isolates the numerical linear algebra required for ADMM updates.
By keeping these functions stateless, they are easier to optimize,
and read independently from the layer mechanics.
"""

import torch

def solve_least_squares_weights(numerator: torch.Tensor, denominator: torch.Tensor, cached_pinv: torch.Tensor = None):
    """Solves the regularized least-squares problem for the Weight matrix W.

    W_new = (Y^T @ P) @ (P^T @ P)^-1

    Args:
        numerator (torch.Tensor): The numerator matrix (Y^T @ P).
        denominator (torch.Tensor): The denominator matrix (P^T @ P).
        cached_pinv (torch.Tensor, optional): A pre-computed pseudoinverse of the 
            denominator. Defaults to None.

    Returns:
        tuple:
            - torch.Tensor: The newly computed weight matrix.
            - torch.Tensor: The computed or utilized pseudoinverse matrix.
    """
    if cached_pinv is None:
        pinv = torch.linalg.pinv(denominator)
    else:
        pinv = cached_pinv
        
    new_W = numerator @ pinv
    return new_W, pinv


def solve_woodbury_system(W: torch.Tensor, B: torch.Tensor, beta_eff: float, rho: float, out_shape: tuple, in_features: int):
    """Solves (beta_eff * I + rho * W^T W) x = B using the Woodbury Matrix Identity."""
    out_features = W.size(0)
    B_flat = B.reshape(-1, in_features).t() # Shape: [in_features, Batch]
    
    # 1. Compute tiny S matrix
    WWT = torch.matmul(W, W.t()) 
    I_k = torch.eye(out_features, device=W.device, dtype=W.dtype)
    S = beta_eff * I_k + rho * WWT
    
    # 2. Solve tiny system
    WB = torch.matmul(W, B_flat) 
    S_inv_WB = torch.linalg.solve(S, WB) 
    
    # 3. Final Woodbury assembly
    term2 = torch.matmul(W.t(), S_inv_WB)
    x_flat = (1.0 / beta_eff) * B_flat - (rho / beta_eff) * term2
    
    return x_flat.t().view(out_shape)

def solve_linear_system(A: torch.Tensor, B: torch.Tensor, out_shape: tuple, in_features: int):
    """Solves a standard linear system Ax = B.

    Used primarily for the exact activation (a) updates.

    Args:
        A (torch.Tensor): The left-hand side matrix.
        B (torch.Tensor): The right-hand side tensor.
        out_shape (tuple): The desired shape of the output tensor.
        in_features (int): The number of input features for reshaping.

    Returns:
        torch.Tensor: The solved system reshaped to `out_shape`.
    """
    if isinstance(A, dict):
        return solve_woodbury_system(A['W'], B, A['beta_eff'], A['rho'], out_shape, in_features)
    B_flat = B.reshape(-1, in_features)
    x_flat = torch.linalg.solve(A, B_flat.t()).t()
    return x_flat.view(out_shape)


def solve_spiking_system(A_main: torch.Tensor, A_last: torch.Tensor, 
                              B: torch.Tensor, out_shape: tuple, in_features: int, T: int):
    """Solves the linear system  Ax = B for spiking networks.

    Handles the boundary condition at the final timestep separately from the main timesteps.

    Args:
        A_main (torch.Tensor): The left-hand side matrix for the main timesteps.
        A_last (torch.Tensor): The left-hand side matrix for the final timestep.
        B (torch.Tensor): The right-hand side tensor.
        out_shape (tuple): The desired output shape.
        in_features (int): The number of input features.
        T (int): The total number of timesteps.

    Returns:
        torch.Tensor: The concatenated solved activations across all timesteps.
    """
    # Solve the last timestep
    a_last = solve_linear_system(A_last, B[-1], (1, *out_shape[1:]), in_features)
    
    # Solve the main timesteps
    if T > 1:
        a_main = solve_linear_system(A_main, B[:-1], (T - 1, *out_shape[1:]), in_features)
    else:
        a_main = torch.empty((0, *out_shape[1:]), device=B.device)
        
    return torch.cat([a_main, a_last], dim=0)

