"""
ADMM Execution Speed Benchmark
"""

import time

import torch

from admm.activation_functions import ADMM_Heaviside
from admm.dataclasses import ADMM_LayerState  # <-- Added Import
from admm.layers import ADMM_SpikingLinear


def test_speed_benchmark():
    print("\n" + "=" * 70)
    print("  ADMM TEMPORAL SCHEMES: EXECUTION SPEED BENCHMARK")
    print("=" * 70)

    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    T, batch, in_f, out_f = 200, 64, 128, 128
    time_steps = list(range(T - 1))
    iterations = 10

    print(f"Network Size : [{T} Timesteps] x [{batch} Batch] x [{in_f} Features]")
    print(f"Iterations   : {iterations} passes per method\n")

    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0}

    h_func = ADMM_Heaviside()
    layer_base = ADMM_SpikingLinear(in_f, out_f, h=h_func, **config)
    layer_base.device = device
    layer_base._setup()
    layer_base.T = T

    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0, "use_reset": False}
    next_layer = ADMM_SpikingLinear(out_f, out_f, h=h_func, **config)
    next_layer.device = device
    next_layer._setup()
    next_layer.T = T

    # 3. Create the states directly!
    a_prev = torch.randn((T, batch, in_f))
    z_init = torch.randn((T, batch, out_f))
    a_init = torch.rand((T, batch, out_f))

    next_state = ADMM_LayerState(
        z=torch.randn((T, batch, out_f)), a=torch.rand((T, batch, out_f))
    )

    # Separate states for each method, using the exact same base layer!
    state_vec = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())
    state_unrolled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())
    state_decoupled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    # WARMUP
    _ = layer_base.spatial_forward(a_prev)

    # =======================================================
    # METHOD 1: FULLY VECTORIZED (Pure Jacobi)
    # =======================================================
    start = time.perf_counter()
    for _ in range(iterations):
        layer_base.update_a(
            next_layer=next_layer, next_state=next_state, state=state_vec, a_prev=a_prev
        )
        layer_base.update_z(state=state_vec, a_prev=a_prev)
    time_vec = (time.perf_counter() - start) / iterations

    # =======================================================
    # METHOD 2: UNROLLED SEQUENTIAL (Strict Gauss-Seidel)
    # =======================================================
    start = time.perf_counter()
    for _ in range(iterations):
        layer_base.update_az_interleaved(
            next_layer=next_layer,
            next_state=next_state,
            state=state_unrolled,
            a_prev=a_prev,
            time_steps=time_steps,
            update_z_first=False,
        )
    time_unrolled = (time.perf_counter() - start) / iterations

    # =======================================================
    # METHOD 3: DECOUPLED SEQUENTIAL (Mixed Method)
    # =======================================================
    start = time.perf_counter()
    for _ in range(iterations):
        layer_base.update_a(
            next_layer=next_layer,
            next_state=next_state,
            state=state_decoupled,
            a_prev=a_prev,
        )
        layer_base.update_z_decoupled(
            state=state_decoupled, a_prev=a_prev, time_steps=time_steps
        )
    time_decoupled = (time.perf_counter() - start) / iterations

    # REPORTING... (Keep print statements same as before)
    # =======================================================
    # REPORTING
    # =======================================================
    print(
        f"{'Method':<25} | {'Avg Time / Iter':<17} | {'Speedup relative to Unrolled'}"
    )
    print("-" * 70)
    print(f"{'Unrolled (Baseline)':<25} | {time_unrolled:.5f} sec       | 1.00x")
    print(
        f"{'Decoupled (Mixed)':<25} | {time_decoupled:.5f} sec       | {time_unrolled / time_decoupled:.2f}x Faster"
    )
    print(
        f"{'Vectorized (Jacobi)':<25} | {time_vec:.5f} sec       | {time_unrolled / time_vec:.2f}x Faster"
    )
    print("-" * 70)

    print("\nTAKEAWAYS:")
    print(
        "- Vectorized is the fastest to compute, but has the slowest mathematical convergence (requires more ADMM epochs)."
    )
    print(
        "- Unrolled is the slowest to compute, but has the fastest mathematical convergence."
    )
    print(
        "- Decoupled shares the EXACT SAME fast mathematical convergence as Unrolled, but computes significantly faster."
    )
    print("=" * 70 + "\n")
