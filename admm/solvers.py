"""
ADMM Solvers Module

This module isolates the numerical linear algebra required for ADMM updates.
By keeping these functions stateless, they are easier to optimize,
and read independently from the layer mechanics.
"""

import torch

def conjugate_gradient(A: torch.Tensor, B: torch.Tensor, x0: torch.Tensor = None, tol: float = 1e-4, max_iter: int = 50):
    """
    Solves AX = B using the Conjugate Gradient method.
    Designed for batched solving (X has multiple columns).
    """
    # X is our W^T. Start with a cached guess or zeros.
    x = x0 if x0 is not None else torch.zeros_like(B)
    
    # Initial residual: R = B - AX
    r = B - torch.matmul(A, x)
    p = r.clone()
    
    # Squared norm of the residuals per column
    rs_old = torch.sum(r * r, dim=0) 
    
    for i in range(max_iter):
        Ap = torch.matmul(A, p)
        
        # Step size alpha
        p_Ap = torch.sum(p * Ap, dim=0).clamp(min=1e-8)
        alpha = rs_old / p_Ap
        
        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = torch.sum(r * r, dim=0)
        
        # Check for convergence across all columns
        if torch.max(rs_new) < tol:
            break
            
        # Update conjugate direction
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        
    return x

def solve_least_squares_weights(numerator: torch.Tensor, denominator: torch.Tensor, cached_pinv: torch.Tensor = None, use_cholesky: bool = True, use_cg: bool = False):
    """Solves the regularized least-squares problem for the Weight matrix W.

    W_new = (Y^T @ P) @ (P^T @ P)^-1
    """
    # ---------------------------------------------------------
    # NEW: Conjugate Gradient Route
    # ---------------------------------------------------------
    if use_cg:
        # Add slight jitter for positive-definiteness
        max_val = torch.max(torch.abs(denominator)).clamp(min=1.0)
        jitter = 1e-4 * max_val
        denominator.diagonal().add_(jitter)
        
        # Here, cached_pinv acts as the warm-start guess for W^T (X0)
        W_new_T = conjugate_gradient(A=denominator, B=numerator.mT, x0=cached_pinv)
        
        # Return the new weights and the new weights transposed as the cache for next time
        return W_new_T.mT, W_new_T

    # ---------------------------------------------------------
    # Legacy: Cholesky / Pinv Route
    # ---------------------------------------------------------
    is_cached_cholesky = False
    if cached_pinv is not None and cached_pinv.shape == denominator.shape:
        is_cached_cholesky = torch.allclose(cached_pinv, torch.tril(cached_pinv))

    if use_cholesky:
        if cached_pinv is None or not is_cached_cholesky:
            denominator.add_(denominator.mT.clone()).div_(2.0)
            max_val = torch.max(torch.abs(denominator)).clamp(min=1.0)
            jitter = 1e-4 * max_val
            denominator.diagonal().add_(jitter)
            try:
                L = torch.linalg.cholesky(denominator)
                W_new_T = torch.cholesky_solve(numerator.mT, L)
                return W_new_T.mT, L
            except torch._C._LinAlgError:
                print("⚠️ Cholesky failed. Falling back to pinv.")
                pinv = torch.linalg.pinv(denominator)
                return numerator @ pinv, pinv
        else:
            L = cached_pinv 
            W_new_T = torch.cholesky_solve(numerator.mT, L)
            return W_new_T.mT, L
            
    else:
        if cached_pinv is None or is_cached_cholesky:
            pinv = torch.linalg.pinv(denominator)
        else:
            pinv = cached_pinv
            
        return numerator @ pinv, pinv

# def solve_least_squares_weights(numerator: torch.Tensor, denominator: torch.Tensor, cached_pinv: torch.Tensor = None, use_cholesky: bool = True):
#     """Solves the regularized least-squares problem for the Weight matrix W.

#     W_new = (Y^T @ P) @ (P^T @ P)^-1

#     Args:
#         numerator (torch.Tensor): The numerator matrix (Y^T @ P).
#         denominator (torch.Tensor): The denominator matrix (P^T @ P).
#         cached_pinv (torch.Tensor, optional): A pre-computed pseudoinverse of the 
#             denominator. Defaults to None.

#     Returns:
#         tuple:
#             - torch.Tensor: The newly computed weight matrix.
#             - torch.Tensor: The computed or utilized pseudoinverse matrix.
#     """
#     is_cached_cholesky = False
#     if cached_pinv is not None:
#         is_cached_cholesky = torch.allclose(cached_pinv, torch.tril(cached_pinv))

#     if use_cholesky:
#         if cached_pinv is None or not is_cached_cholesky:
#             denominator.add_(denominator.mT.clone()).div_(2.0)
#             max_val = torch.max(torch.abs(denominator)).clamp(min=1.0)
#             jitter = 1e-4 * max_val
#             denominator.diagonal().add_(jitter)
#             try:
#                 L = torch.linalg.cholesky(denominator)
#                 W_new_T = torch.cholesky_solve(numerator.mT, L)
#                 return W_new_T.mT, L
#             except torch._C._LinAlgError:
#                 print("⚠️ Cholesky failed. Falling back to pinv.")
#                 pinv = torch.linalg.pinv(denominator)
#                 return numerator @ pinv, pinv
#         else:
#             L = cached_pinv 
#             W_new_T = torch.cholesky_solve(numerator.mT, L)
#             return W_new_T.mT, L
            
#     else:
#         # Standard pinv route
#         if cached_pinv is None or is_cached_cholesky:
#             pinv = torch.linalg.pinv(denominator)
#         else:
#             pinv = cached_pinv
            
#         return numerator @ pinv, pinv
    
def solve_woodbury_system(W: torch.Tensor, B: torch.Tensor, beta: float, rho: float, a_shape: tuple):
    """
    Solves the linear system (beta * I_N + rho * W^T W) x = B for x, using the Woodbury Matrix Identity.
    
    Why use this? 
    Inverting the massive (N x N) matrix takes O(N^3) memory and time. Woodbury allows us 
    to instead invert a tiny (M x M) matrix, dropping complexity to O(M^3).
    
    Mathematical Derivation:
    1. Standard Woodbury Formula: 
       (A + UCV)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1
       
    2. Our Substitutions:
       A = beta * I_N
       U = W^T
       C = rho * I_M
       V = W
       
    3. The Simplified Result applied to B:
       x = (1/beta)*B - (rho/beta) * W^T * (beta * I_M + rho * W W^T)^-1 * W * B
       
    Args:
        W (torch.Tensor): Weight matrix of shape [out_features, in_features].
        B (torch.Tensor): Target tensor to solve against.
        beta (float): Penalty parameter for the activation.
        rho (float): Penalty parameter for the affine constraint.
        a_shape (tuple): The original geometric shape of the activation tensor (e.g., [Batch, Features]).
        out_features (int): The number of rows in W (M).
        in_features (int): The number of columns in W (N).
        
    Returns:
        torch.Tensor: The solved activations x, reshaped back to a_shape.
    """
    out_features, in_features = W.shape
    B_flat = B.reshape(-1, in_features).t() 
    
    WWT = torch.matmul(W, W.t()) 
    WWT.mul_(rho)
    WWT.diagonal().add_(beta)

    WB = torch.matmul(W, B_flat) 
    parenthesis_inv_WB = torch.linalg.solve(WWT, WB) 
    
    term2 = torch.matmul(W.t(), parenthesis_inv_WB)

    x_flat = B_flat.div(beta)
    x_flat.sub_(term2, alpha=(rho / beta))
    
    return x_flat.t().view(a_shape)

def solve_linear_system(A: torch.Tensor, B: torch.Tensor, a_shape: tuple, in_features: int):
    """Solves a standard linear system Ax = B.

    Used primarily for the exact activation (a) updates.

    Args:
        A (torch.Tensor): The left-hand side matrix.
        B (torch.Tensor): The right-hand side tensor.
        a_shape (tuple): The desired shape of the output tensor.
        in_features (int): The number of input features for reshaping.

    Returns:
        torch.Tensor: The solved system reshaped to `a_shape`.
    """
    if isinstance(A, dict):
        return solve_woodbury_system(A['W'], B, A['beta'], A['rho'], a_shape)
    B_flat = B.reshape(-1, in_features)
    x_flat = torch.linalg.solve(A, B_flat.t()).t()
    return x_flat.view(a_shape)


def solve_spiking_system(A_main: torch.Tensor, A_last: torch.Tensor, 
                              B: torch.Tensor, a_shape: tuple, in_features: int, T: int):
    """Solves the linear system  Ax = B for spiking networks.

    Handles the boundary condition at the final timestep separately from the main timesteps.

    Args:
        A_main (torch.Tensor): The left-hand side matrix for the main timesteps.
        A_last (torch.Tensor): The left-hand side matrix for the final timestep.
        B (torch.Tensor): The right-hand side tensor.
        a_shape (tuple): The desired output shape.
        in_features (int): The number of input features.
        T (int): The total number of timesteps.

    Returns:
        torch.Tensor: The concatenated solved activations across all timesteps.
    """
    # Solve the last timestep
    a_last = solve_linear_system(A_last, B[-1], (1, *a_shape[1:]), in_features)
    
    # Solve the main timesteps
    if T > 1:
        a_main = solve_linear_system(A_main, B[:-1], (T - 1, *a_shape[1:]), in_features)
    else:
        a_main = torch.empty((0, *a_shape[1:]), device=B.device)
        
    return torch.cat([a_main, a_last], dim=0)

