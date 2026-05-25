"""Validates the mathematical equivalence of frequency-domain solvers.

Checks that the FFT-based system solvers and covariance accumulators yield
the exact same numerical results (within floating-point tolerance) as their dense,
spatial-domain counterparts. Covers both 4D static convolutions and 5D spiking
convolutions, including adjoint operator checks and unrolled timestep slices.
"""

import pytest
import torch

from src.admm.dataclasses import ADMM_LayerConfig
from src.admm.functional.a_update_solvers import (
    solve_fft_system_unrolled,
    solve_standard_system_unrolled,
)
from src.admm.layers import ADMM_Conv2d, ADMM_SpikingConv2d

BATCH, IN_C, OUT_C, H, W = 2, 2, 4, 8, 8
T, K = 3, 3


@pytest.fixture
def base_tensors():
    torch.set_default_dtype(torch.float64)
    W_weight = torch.randn(OUT_C, IN_C, K, K, dtype=torch.float64)
    return W_weight


def test_fft_math_4d_standard_conv(base_tensors):
    shape_4d = (BATCH, IN_C, H, W)
    W_weight = base_tensors

    fft_config = ADMM_LayerConfig(beta=1.0, rho=1.0, use_fft=True)
    non_fft_config = ADMM_LayerConfig(beta=1.0, rho=1.0, use_fft=False)

    l_conv_fft = ADMM_Conv2d(
        IN_C, OUT_C, K, p=1, s=1, config=fft_config, padding_mode="circular"
    )
    l_conv_dense = ADMM_Conv2d(
        IN_C, OUT_C, K, p=1, s=1, config=non_fft_config, padding_mode="circular"
    )

    l_conv_fft.W = torch.nn.Parameter(W_weight.clone())
    l_conv_dense.W = torch.nn.Parameter(W_weight.clone())

    # Adjoint Check
    x = torch.randn(shape_4d, dtype=torch.float64)
    Ax = l_conv_fft.spatial_forward(x)
    y = torch.randn_like(Ax, dtype=torch.float64)
    A_T_y = l_conv_fft.adjoint_operator(y)

    diff_adj = torch.max(torch.abs(torch.sum(Ax * y) - torch.sum(x * A_T_y))).item()
    assert diff_adj < 1e-9, "4D Adjoint Operator Math Failed"

    # System Solver
    num_4d = torch.randn(shape_4d, dtype=torch.float64)

    flat_spatial_size = IN_C * H * W
    I_spatial = torch.eye(flat_spatial_size, dtype=torch.float64).reshape(
        flat_spatial_size, IN_C, H, W
    )
    W_exp_d = l_conv_dense.spatial_forward(I_spatial, use_bias=False)
    W_exp_d = W_exp_d.reshape(flat_spatial_size, OUT_C * H * W).t()

    a_d = l_conv_dense._solve_standard_system(
        next_layer=l_conv_dense,
        W_expanded=W_exp_d,
        numerator=num_4d,
        a_shape=shape_4d,
    )

    a_f = l_conv_fft._solve_fft_system(
        numerator=num_4d,
        next_layer=l_conv_fft,
        a_shape=shape_4d,
    )

    diff_4d = torch.max(torch.abs(a_d - a_f)).item()
    assert diff_4d < 1e-9, "4D System Solver Failed"

    # --- 1.3 Covariances ---
    Y_4d = torch.randn((BATCH, OUT_C, H, W), dtype=torch.float64)
    a_prev_4d = torch.randn((BATCH, IN_C, H, W), dtype=torch.float64)
    mock_state = type(
        "obj", (object,), {"z": Y_4d, "lambda_lagrange": None, "a": None}
    )()
    num_fft, den_fft = l_conv_fft.compute_batch_covariances(
        state=mock_state,
        a_prev=a_prev_4d,
    )[:2]

    num_dense, den_dense = l_conv_dense.compute_batch_covariances(
        state=mock_state,
        a_prev=a_prev_4d,
    )[:2]

    diff_cov = max(
        torch.max(torch.abs(num_fft - num_dense)).item(),
        torch.max(torch.abs(den_fft - den_dense)).item(),
    )
    assert diff_cov < 1e-9, "4D Covariance Mismatch: FFT vs Dense"


def test_fft_math_5d_spiking_conv(base_tensors):
    shape_5d = (T, BATCH, IN_C, H, W)
    W_weight = base_tensors

    fft_config = ADMM_LayerConfig(
        beta=1.0, rho=1.0, thetas=1.0, deltas=0.8, use_fft=True
    )
    non_fft_config = ADMM_LayerConfig(
        beta=1.0, rho=1.0, thetas=1.0, deltas=0.8, use_fft=False
    )

    l_spk_fft = ADMM_SpikingConv2d(
        IN_C, OUT_C, K, p=1, s=1, config=fft_config, padding_mode="circular"
    )
    l_spk_dense = ADMM_SpikingConv2d(
        IN_C, OUT_C, K, p=1, s=1, config=non_fft_config, padding_mode="circular"
    )

    for layer in [l_spk_fft, l_spk_dense]:
        layer.T = T
        layer.W = torch.nn.Parameter(W_weight.clone())

    # Adjoint Math
    x_5d = torch.randn(shape_5d, dtype=torch.float64)
    Ax_5d = l_spk_fft.spatial_forward(x_5d)
    y_5d = torch.randn_like(Ax_5d, dtype=torch.float64)
    A_T_y_5d = l_spk_fft.adjoint_operator(y_5d)

    diff_adj_5d = torch.max(
        torch.abs(torch.sum(Ax_5d * y_5d) - torch.sum(x_5d * A_T_y_5d))
    ).item()
    assert diff_adj_5d < 1e-9, "5D Adjoint Operator Math Failed"

    # System Solver
    num_5d = torch.randn(shape_5d, dtype=torch.float64)

    flat_spatial_size = IN_C * H * W
    I_spatial = torch.eye(flat_spatial_size, dtype=torch.float64).reshape(
        flat_spatial_size, IN_C, H, W
    )
    W_exp_d5 = l_spk_dense.spatial_forward(I_spatial, use_bias=False)
    W_exp_d5 = W_exp_d5.reshape(flat_spatial_size, OUT_C * H * W).t()

    a_d_5d = l_spk_dense._solve_standard_system(
        next_layer=l_spk_dense,
        W_expanded=W_exp_d5,
        numerator=num_5d,
        a_shape=shape_5d,
    )

    a_f_5d = l_spk_fft._solve_fft_system(
        numerator=num_5d, next_layer=l_spk_fft, a_shape=shape_5d
    )

    diff_5d = torch.max(torch.abs(a_d_5d - a_f_5d)).item()

    assert diff_5d < 1e-7, f"5D System Solver Failed with Diff: {diff_5d:.8e}"
    # Unrolled Solver

    den_d_5d_m, _, _ = l_spk_dense._get_a_denominator(
        config_prev=l_spk_dense.config,
        W_expanded=W_exp_d5,
    )

    den_f_5d_m_penalized, _ = l_spk_fft._get_fft_a_denominator(
        config_prev=l_spk_fft.config, a_shape=shape_5d
    )

    inv_den_d = torch.linalg.inv(den_d_5d_m)
    inv_den_f = torch.linalg.inv(den_f_5d_m_penalized)
    num_slice = num_5d[0]

    a_unroll_d = solve_standard_system_unrolled(
        numerator=num_slice, denominator=inv_den_d
    )
    a_unroll_f = solve_fft_system_unrolled(numerator=num_slice, denominator=inv_den_f)

    diff_unroll = torch.max(torch.abs(a_unroll_d - a_unroll_f)).item()
    assert diff_unroll < 1e-9, "4D Unrolled Slice Math Failed"

    # Covariances
    Y_5d = torch.randn((T, BATCH, OUT_C, H, W), dtype=torch.float64)
    a_prev_5d = torch.randn((T, BATCH, IN_C, H, W), dtype=torch.float64)
    a_dummy_5d = torch.randn_like(Y_5d)

    from src.admm.dataclasses import ADMM_LayerState

    mock_state = ADMM_LayerState(z=Y_5d, a=a_dummy_5d, lambda_lagrange=None)

    num_fft_5d, den_fft_5d = l_spk_fft.compute_batch_covariances(
        state=mock_state, a_prev=a_prev_5d
    )[:2]

    num_dense_5d, den_dense_5d = l_spk_dense.compute_batch_covariances(
        state=mock_state, a_prev=a_prev_5d
    )[:2]
    diff_cov_5d = max(
        torch.max(torch.abs(num_fft_5d - num_dense_5d)).item(),
        torch.max(torch.abs(den_fft_5d - den_dense_5d)).item(),
    )

    assert diff_cov_5d < 1e-9, "5D Covariance Mismatch: FFT vs Dense"
