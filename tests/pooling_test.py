"""Validates the adjoint mathematical properties of spatial pooling operators.

Ensures that the forward pooling operations and their respective backward (adjoint)
operations strictly satisfy the inner product definition of an adjoint operator:
$\langle Ax, y \rangle = \langle x, A^T y \rangle$.
"""

import pytest
import torch

from src.admm.pooling import ADMM_GAP, ADMM_Flatten, ADMM_SpatialPool


@pytest.mark.parametrize(
    "operator", [ADMM_Flatten(), ADMM_GAP(), ADMM_SpatialPool(output_size=(4, 4))]
)
def test_pooling_adjoint_math(operator):
    test_shape = (4, 16, 32, 32)

    # 1. Forward Pass
    x = torch.randn(test_shape, dtype=torch.float64)
    Ax = operator(x)

    # 2. Adjoint (Backward) Pass
    y = torch.randn_like(Ax, dtype=torch.float64)
    A_T_y = operator.adjoint(y, original_input_shape=x.shape)

    # 3. Inner Products
    inner_1 = torch.sum(Ax * y)
    inner_2 = torch.sum(x * A_T_y)

    # $\sum A(x) \cdot y = \sum x \cdot A^T(y)$
    diff = torch.abs(inner_1 - inner_2).item()

    assert diff < 1e-9, (
        f"Adjoint math failed for {operator.__class__.__name__}. Difference: {diff:.8e}"
    )
