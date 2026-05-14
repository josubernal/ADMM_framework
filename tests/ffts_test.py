"""
ADMM Convolutional Mixin Verification (FFT vs Spatial Math Test)

PURPOSE:
Verifies that the highly optimized FFT frequency-domain math in the
convolutional mixin produces equivalent results to the dense matrix calculations.
"""

import torch

from admm.dataclasses import ADMM_LayerConfig
from admm.layers import ADMM_Conv2d, ADMM_SpikingConv2d


def test_fft_math():
    print("\n" + "=" * 55)
    print("  FFT MATH VERIFICATION TEST           ")
    print("=" * 55)

    batch, in_c, out_c, h, w = 2, 2, 4, 8, 8
    T, k = 3, 3
    shape_4d = (batch, in_c, h, w)
    shape_5d = (T, batch, in_c, h, w)
    W = torch.randn(out_c, in_c, k, k, dtype=torch.float64)

    # =======================================================
    # PART 1: STANDARD 2D CONVOLUTION (4D)
    # =======================================================
    print("Standard 2D Convolutions (4D Tensors)\n")

    fft_config = ADMM_LayerConfig(beta=1, rho=1, use_fft=True)
    non_fft_config = ADMM_LayerConfig(beta=1, rho=1, use_fft=False)
    l_conv_fft = ADMM_Conv2d(
        in_c, out_c, k, p=1, s=1, config=fft_config, padding_mode="circular"
    )
    l_conv_dense = ADMM_Conv2d(
        in_c, out_c, k, p=1, s=1, config=non_fft_config, padding_mode="circular"
    )

    for layer in [l_conv_fft, l_conv_dense]:
        layer.W = torch.nn.Parameter(W.clone())

    # 1.1 Adjoint Math
    x = torch.randn(shape_4d, dtype=torch.float64)
    Ax = l_conv_fft.spatial_forward(x)
    y = torch.randn_like(Ax, dtype=torch.float64)
    A_T_y = l_conv_fft.adjoint_operator(y)

    diff_adj = torch.abs(torch.sum(Ax * y) - torch.sum(x * A_T_y)).item()
    print(
        f"[4D Adjoint Operator]   Difference: {diff_adj:.8e} "
        + ("✅" if diff_adj < 1e-9 else "❌")
    )

    # 1.2 System Solver
    num_4d = torch.randn(shape_4d, dtype=torch.float64)
    den_d_main, _, in_f_d = l_conv_dense.get_a_denominator(1.0, shape_4d)
    den_f_main, _, in_f_f = l_conv_fft.get_a_denominator(1.0, shape_4d)

    a_d = l_conv_dense._solve_activation_system(
        num_4d, den_d_main, den_d_main, shape_4d, in_f_d
    )
    a_f = l_conv_fft._solve_activation_system(
        num_4d, den_f_main, den_f_main, shape_4d, in_f_f
    )
    diff_sol = torch.max(torch.abs(a_d - a_f)).item()
    print(
        f"[4D System Solver]      Difference: {diff_sol:.8e} "
        + ("✅" if diff_sol < 1e-9 else "❌")
    )

    # 1.3 Covariances (Chunked vs Monolithic)
    Y_4d = torch.randn((batch, out_c, h, w), dtype=torch.float64)
    a_prev_4d = torch.randn((batch, in_c, h, w), dtype=torch.float64)

    num_chunk, den_chunk = l_conv_fft.compute_covariances(Y_4d, a_prev_4d)

    P_4d = l_conv_fft._compute_P(a_prev_4d)
    Y_flat_4d = Y_4d.movedim(l_conv_fft.channel_dim, -1).reshape(-1, out_c)
    num_mono, den_mono = Y_flat_4d.t() @ P_4d, P_4d.t() @ P_4d

    diff_cov = max(
        torch.max(torch.abs(num_chunk - num_mono)).item(),
        torch.max(torch.abs(den_chunk - den_mono)).item(),
    )
    print(
        f"[4D Covariance Calc]    Difference: {diff_cov:.8e} "
        + ("✅" if diff_cov < 1e-9 else "❌")
    )

    print("\n-------------------------------------------------------\n")

    # =======================================================
    # PART 2: SPIKING 2D CONVOLUTION (5D)
    # =======================================================
    print("Spiking 2D Convolutions (5D Tensors)\n")

    l_spk_fft = ADMM_SpikingConv2d(
        in_c, out_c, k, p=1, s=1, config=fft_config, padding_mode="circular"
    )
    l_spk_dense = ADMM_SpikingConv2d(
        in_c, out_c, k, p=1, s=1, config=non_fft_config, padding_mode="circular"
    )

    for layer in [l_spk_fft, l_spk_dense]:
        layer.T = T
        layer.W = torch.nn.Parameter(W.clone())

    # 2.1 Adjoint Math (5D)
    x_5d = torch.randn(shape_5d, dtype=torch.float64)
    Ax_5d = l_spk_fft.spatial_forward(x_5d)
    y_5d = torch.randn_like(Ax_5d, dtype=torch.float64)
    A_T_y_5d = l_spk_fft.adjoint_operator(y_5d)

    diff_adj_5d = torch.abs(torch.sum(Ax_5d * y_5d) - torch.sum(x_5d * A_T_y_5d)).item()
    print(
        f"[5D Adjoint Operator]   Difference: {diff_adj_5d:.8e} "
        + ("✅" if diff_adj_5d < 1e-9 else "❌")
    )

    # 2.2 System Solver (5D)
    num_5d = torch.randn(shape_5d, dtype=torch.float64)
    den_d_5d_m, den_d_5d_l, in_f_d5 = l_spk_dense.get_a_denominator(
        beta_current=1.0, rho_current=0.1, thetas_current=0.3, a_shape=shape_5d
    )
    den_f_5d_m, den_f_5d_l, in_f_f5 = l_spk_fft.get_a_denominator(
        beta_current=1.0, rho_current=0.1, thetas_current=0.3, a_shape=shape_5d
    )

    a_d_5d = l_spk_dense._solve_activation_system(
        num_5d, den_d_5d_m, den_d_5d_l, shape_5d, in_f_d5
    )
    a_f_5d = l_spk_fft._solve_activation_system(
        num_5d, den_f_5d_m, den_f_5d_l, shape_5d, in_f_f5
    )
    diff_sol_5d = torch.max(torch.abs(a_d_5d - a_f_5d)).item()
    print(
        f"[5D System Solver]      Difference: {diff_sol_5d:.8e} "
        + ("✅" if diff_sol_5d < 1e-9 else "❌")
    )

    # 2.3 Unrolled Solver (Single Timestep - 4D Slice)
    inv_den_d = torch.linalg.inv(den_d_5d_m)
    inv_den_f = torch.linalg.inv(den_f_5d_m)

    # Simulating a single timestep passed into the unrolled solver
    num_slice = num_5d[0]
    a_unroll_d = l_spk_dense._solve_activation_system_unrolled(num_slice, inv_den_d)
    a_unroll_f = l_spk_fft._solve_activation_system_unrolled(num_slice, inv_den_f)

    diff_unroll = torch.max(torch.abs(a_unroll_d - a_unroll_f)).item()
    print(
        f"[4D Unrolled Slice]     Difference: {diff_unroll:.8e} "
        + ("✅" if diff_unroll < 1e-9 else "❌")
    )

    # 2.4 Covariances (5D Time-Chunked vs Monolithic)
    Y_5d = torch.randn((T, batch, out_c, h, w), dtype=torch.float64)
    a_prev_5d = torch.randn((T, batch, in_c, h, w), dtype=torch.float64)

    num_chunk_5d, den_chunk_5d = l_spk_fft.compute_covariances(Y_5d, a_prev_5d)

    P_5d = l_spk_fft._compute_P(a_prev_5d)
    Y_flat_5d = Y_5d.movedim(l_spk_fft.channel_dim, -1).reshape(-1, out_c)
    num_mono_5d, den_mono_5d = Y_flat_5d.t() @ P_5d, P_5d.t() @ P_5d

    diff_cov_5d = max(
        torch.max(torch.abs(num_chunk_5d - num_mono_5d)).item(),
        torch.max(torch.abs(den_chunk_5d - den_mono_5d)).item(),
    )
    print(
        f"[5D Covariance Calc]    Difference: {diff_cov_5d:.8e} "
        + ("✅" if diff_cov_5d < 1e-9 else "❌")
    )

    print("\n=======================================================\n")
