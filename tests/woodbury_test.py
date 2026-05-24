"""Validates the mathematical accuracy of the Woodbury Matrix Identity solver.

Compares the activations calculated using the highly optimized $O(M^3)$ Woodbury
inversion against the exact solution derived from a standard, computationally
expensive $O(N^3)$ dense matrix inversion. Ensures numerical stability under
highly rectangular matrix dimensions.
"""

import torch

from src.admm.functional.a_update_solvers import (
    solve_standard_system_static,
    solve_woodbury_system_static,
)


def test_woodbury_identity_math_equivalence():
    # Force double precision for strict mathematical verification
    torch.set_default_dtype(torch.float64)

    in_features = 400
    out_features = 10
    batch_size = 50
    out_shape = (batch_size, in_features)
    beta_eff = 0.1
    rho = 0.01

    # Random weights and targets
    W = torch.randn((out_features, in_features), dtype=torch.float64)
    B = torch.randn(out_shape, dtype=torch.float64)

    # Dense solver
    WtW = torch.matmul(W.t(), W)
    I_dense = torch.eye(in_features, dtype=torch.float64)
    A_dense = beta_eff * I_dense + rho * WtW
    x_dense = solve_standard_system_static(A_dense, B, out_shape, in_features)

    # Woodbury solver
    x_wood = solve_woodbury_system_static(W, B, beta_eff, rho, out_shape)

    torch.testing.assert_close(
        x_wood,
        x_dense,
        rtol=0.0,
        atol=1e-9,
        msg=lambda msg: f"Woodbury solver mathematically diverged from Dense! {msg}",
    )
