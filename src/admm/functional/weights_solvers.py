"""
This module isolates the numerical algebra required for ADMM weight updates.
By keeping these functions stateless, they are easier to optimize, maintain,
and read independently from the layer mechanics. Additionally it allows us to experient
with different solvers modularly.
"""

import torch

from ..dataclasses import ADMM_Config

#####################################################################
# weight update
#####################################################################


def solve_weights(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    config: ADMM_Config,
    cached_pinv: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solves the regularized least-squares problem for the Weight matrix $W$.

    Formula evaluated: $W_{new} = (Y^T P) (P^T P)^{-1}$

    Args:
        numerator (torch.Tensor): The numerator matrix ($Y^T P$).
        denominator (torch.Tensor): The denominator matrix ($P^T P$).
        config (ADMM_LayerConfig): [The configuration object][src.admm.dataclasses.ADMM_LayerConfig] of the layer.
        cached_pinv (torch.Tensor, optional): A pre-computed decomposition or inverse of the
            denominator to accelerate the solve. Defaults to None.

    Returns:
        tuple:
            - torch.Tensor: The newly computed weight matrix $W$.
            - torch.Tensor: The computed/utilized cache matrix (Cholesky $L$, $P_{inv}$, or CG cache) for the next iteration.
    """
    is_cached_cholesky = False
    if cached_pinv is not None and cached_pinv.shape == denominator.shape:
        is_cached_cholesky = torch.allclose(cached_pinv, torch.tril(cached_pinv))

    if config.solver == "conjugate-gradient":
        max_val = torch.max(torch.abs(denominator)).clamp(min=1.0)
        jitter = 1e-4 * max_val
        denominator.diagonal().add_(jitter)
        W_new_T = conjugate_gradient(A=denominator, B=numerator.mT, x0=cached_pinv)
        return W_new_T.mT, W_new_T

    elif config.solver == "cholesky":
        result, pinv = cholesky_solver(
            A=denominator,
            B=numerator,
            cached_pinv=cached_pinv,
            is_cached_cholesky=is_cached_cholesky,
        )

    elif config.solver == "standard":
        result, pinv = standard_solver(
            A=denominator,
            B=numerator,
            cached_pinv=cached_pinv,
            is_cached_cholesky=is_cached_cholesky,
        )
    else:
        raise ValueError(
            f"Invalid solver: '{config.solver}'. Allowed: conjugate-grandient, cholesky and standard"
        )

    return result, pinv


def standard_solver(A, B, cached_pinv, is_cached_cholesky):
    if cached_pinv is None or is_cached_cholesky:
        pinv = torch.linalg.pinv(A)
    else:
        pinv = cached_pinv

    return B @ pinv, pinv


def cholesky_solver(A, B, cached_pinv, is_cached_cholesky):
    if cached_pinv is None or not is_cached_cholesky:
        A.add_(A.mT.clone()).div_(2.0)
        max_val = torch.max(torch.abs(A)).clamp(min=1.0)
        jitter = 1e-4 * max_val
        A.diagonal().add_(jitter)
        try:
            L = torch.linalg.cholesky(A)
            W_new_T = torch.cholesky_solve(B.mT, L)
            return W_new_T.mT, L
        except torch._C._LinAlgError:
            pinv = torch.linalg.pinv(A)
            return B @ pinv, pinv
    else:
        L = cached_pinv
        W_new_T = torch.cholesky_solve(B.mT, L)
        return W_new_T.mT, L


def conjugate_gradient(
    A: torch.Tensor,
    B: torch.Tensor,
    x0: torch.Tensor = None,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> torch.Tensor:
    r"""Solves the system $AX = B$ using the Conjugate Gradient method.

    Designed for batched solving where $X$ has multiple columns. This is
    often used as an efficient alternative to full matrix inversion.

    Args:
        A (torch.Tensor): The left-hand side positive-definite matrix.
        B (torch.Tensor): The right-hand side target matrix.
        x0 (torch.Tensor, optional): A warm-start initial guess for $X$. Defaults to None.
        tol (float, optional): The convergence tolerance for the residuals. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations before forcing a stop. Defaults to 50.

    Returns:
        torch.Tensor: The solved matrix $X$.
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
