"""
ADMM Temporal Schemes Verification (Unrolled vs. Decoupled)

PURPOSE:
Verifies that performing a strictly chronological Gauss-Seidel unrolled sweep
(interleaved a and z) produces mathematically identical results to the
Decoupled approach (fully vectorized 'a' update followed by a causal 'z' sweep).
"""

import copy
import time

import torch

from admm.activation_functions import ADMM_Heaviside
from admm.dataclasses import ADMM_LayerState
from admm.layers import ADMM_SpikingLinear


def test_temporal_schemes_equivalence():
    print("\n" + "=" * 55)
    print("  UNROLLED VS DECOUPLED SCHEME VERIFICATION TEST")
    print("=" * 55)

    # Force double precision to verify exact mathematical equivalence
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # 1. Setup dimensions and time sequence
    T, batch, in_f, out_f = 6, 4, 16, 16
    time_steps = list(range(T))  # Strict chronological forward sweep

    # ADMM Hyperparameters
    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0}

    # 2. Instantiate Base Layers
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

    # Replace step 3 and 4 with this:
    a_prev = torch.randn((T, batch, in_f))
    z_init = torch.randn((T, batch, out_f))
    a_init = torch.rand((T, batch, out_f))

    next_state = ADMM_LayerState(
        z=torch.randn((T, batch, out_f)), a=torch.rand((T, batch, out_f))
    )

    layer_unrolled = copy.deepcopy(layer_base)
    state_unrolled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    layer_decoupled = copy.deepcopy(layer_base)
    state_decoupled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    # =======================================================
    # EXECUTE METHOD 1: UNROLLED SEQUENTIAL (Gauss-Seidel)
    # =======================================================
    start_unrolled = time.perf_counter()
    layer_unrolled.update_az_interleaved(
        next_layer=next_layer,
        next_state=next_state,
        state=state_unrolled,
        a_prev=a_prev,
        time_steps=time_steps,
        update_z_first=False,
    )
    time_unrolled = time.perf_counter() - start_unrolled

    # =======================================================
    # EXECUTE METHOD 2: DECOUPLED SEQUENTIAL (Mixed Jacobi)
    # =======================================================
    start_decoupled = time.perf_counter()
    layer_decoupled.update_a(
        next_layer=next_layer,
        next_state=next_state,
        state=state_decoupled,
        a_prev=a_prev,
    )
    layer_decoupled.update_z_decoupled(
        state=state_decoupled, a_prev=a_prev, time_steps=time_steps
    )
    time_decoupled = time.perf_counter() - start_decoupled

    a_diff = torch.max(torch.abs(state_unrolled.a - state_decoupled.a)).item()
    z_diff = torch.max(torch.abs(state_unrolled.z - state_decoupled.z)).item()

    print(
        f"[{'Activation (a) Update':<26}] Max Difference: {a_diff:.8e} "
        + ("✅" if a_diff < 1e-9 else "❌")
    )
    print(
        f"[{'Pre-activation (z) Update':<26}] Max Difference: {z_diff:.8e} "
        + ("✅" if z_diff < 1e-9 else "❌")
    )

    if z_diff < 1e-9 and a_diff < 1e-9:
        if time_decoupled < time_unrolled:
            speedup = time_unrolled / time_decoupled
            print(f"\n🚀 Decoupled method is {speedup:.2f}x faster!")
        else:
            speedup = time_decoupled / time_unrolled
            print(f"\n🐢 Decoupled method is {speedup:.2f}x slower!")
    else:
        print("\nWARNING: Outputs diverged. The methods are not evaluating equally.")

    print("=" * 65 + "\n")
