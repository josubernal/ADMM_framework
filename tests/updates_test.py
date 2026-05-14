"""
ADMM Temporal Schemes Isolated Verification

PURPOSE:
Verifies that isolated unrolled loops for 'a' and 'z' produce mathematically
identical results to their fully vectorized/decoupled counterparts.
This prevents the state-contamination caused by interleaved updates.
"""

import copy
import time
from types import SimpleNamespace

import torch

from admm.activation_functions import ADMM_Heaviside
from admm.dataclasses import ADMM_LayerState
from admm.layers import ADMM_SpikingLinear
from admm.loss_functions import ADMM_SSE


def test_isolated_temporal_schemes():
    # Force double precision to verify exact mathematical equivalence
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # 1. Setup dimensions and sequence
    T, batch, in_f, out_f = 6, 4, 16, 16
    time_steps = list(range(T))

    # ADMM Hyperparameters
    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0}
    h_func = ADMM_Heaviside()

    # 2. Instantiate Base Layers
    layer_base = ADMM_SpikingLinear(in_f, out_f, h=h_func, **config)
    layer_base.device = device
    layer_base._setup()
    layer_base.T = T

    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0, "use_reset": False}
    next_layer = ADMM_SpikingLinear(out_f, out_f, h=h_func, **config)
    next_layer.device = device
    next_layer._setup()
    next_layer.T = T

    # 3. Create identical random states
    a_prev = torch.randn((T, batch, in_f))
    z_init = torch.randn((T, batch, out_f))
    a_init = torch.rand((T, batch, out_f))

    next_state = ADMM_LayerState(
        z=torch.randn((T, batch, out_f)), a=torch.rand((T, batch, out_f))
    )

    print("\n" + "=" * 65)
    print("  TEST 1: ISOLATED 'A' UPDATE (Vectorized vs Unrolled Loop)")
    print("=" * 65)

    # Setup test copies for 'a'
    layer_a_vec = copy.deepcopy(layer_base)
    state_a_vec = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    layer_a_unrolled = copy.deepcopy(layer_base)
    state_a_unrolled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    # METHOD 1: FULLY VECTORIZED A
    start_a_vec = time.perf_counter()
    layer_a_vec.update_a(next_layer, next_state, state_a_vec, a_prev)
    time_a_vec = time.perf_counter() - start_a_vec

    # METHOD 2: ISOLATED UNROLLED A LOOP
    start_a_unrolled = time.perf_counter()
    cache_a = layer_a_unrolled._create_cache(
        next_layer, next_state, a_prev, state_a_unrolled
    )
    for t in range(T):
        layer_a_unrolled.update_a_unrolled(t, cache_a, next_layer, state_a_unrolled)
    time_a_unrolled = time.perf_counter() - start_a_unrolled

    # Verification 'A'
    a_diff = torch.max(torch.abs(state_a_vec.a - state_a_unrolled.a)).item()
    print(
        f"[{'Activation (a) Update':<26}] Max Difference: {a_diff:.8e} "
        + ("✅" if a_diff < 1e-9 else "❌")
    )
    if a_diff < 1e-9:
        speedup_a = (
            (time_a_unrolled / time_a_vec)
            if time_a_vec < time_a_unrolled
            else (time_a_vec / time_a_unrolled)
        )
        icon_a = "🚀" if time_a_vec < time_a_unrolled else "🐢"
        print(
            f"[{icon_a} Speed]: Vectorized method is {speedup_a:.2f}x {'faster' if time_a_vec < time_a_unrolled else 'slower'}."
        )

    print("\n" + "=" * 65)
    print("  TEST 2: ISOLATED 'Z' UPDATE (Vectorized vs Unrolled Loop)")
    print("=" * 65)

    # Setup test copies for 'z' using ADMM_LayerState!
    layer_z_vec = copy.deepcopy(layer_base)
    state_z_vec = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    layer_z_unrolled = copy.deepcopy(layer_base)
    state_z_unrolled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    # METHOD 1: VECTORIZED Z
    start_z_vec = time.perf_counter()
    layer_z_vec.update_z(state_z_vec, a_prev)
    time_z_vec = time.perf_counter() - start_z_vec

    # METHOD 2: ISOLATED UNROLLED Z LOOP
    start_z_unrolled = time.perf_counter()
    forward_pass = layer_z_unrolled.spatial_forward(a_prev)
    mock_cache = SimpleNamespace(forward_pass=forward_pass)

    z_to_use = state_z_unrolled.z.clone()
    for t in range(T):
        layer_z_unrolled.update_z_unrolled(
            t, mock_cache, state_z_unrolled, z_to_use=z_to_use
        )
    time_z_unrolled = time.perf_counter() - start_z_unrolled

    # Verification 'Z'
    z_diff = torch.max(torch.abs(state_z_vec.z - state_z_unrolled.z)).item()
    print(
        f"[{'Pre-activation (z) Update':<26}] Max Difference: {z_diff:.8e} "
        + ("✅" if z_diff < 1e-9 else "❌")
    )
    if z_diff < 1e-9:
        speedup_z = (
            (time_z_unrolled / time_z_vec)
            if time_z_vec < time_z_unrolled
            else (time_z_vec / time_z_unrolled)
        )
        icon_z = "🚀" if time_z_vec < time_z_unrolled else "🐢"
        print(
            f"[{icon_z} Speed]: Vectorized method is {speedup_z:.2f}x {'faster' if time_z_vec < time_z_unrolled else 'slower'}."
        )

    print("=" * 65 + "\n")

    print("=" * 65)
    print("  TEST 3: FINAL LAYER (Z LAST UPDATE)  (Vectorized vs Unrolled Loop)")
    print("=" * 65)

    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0, "use_reset": False}
    # Setup Layers for Test 3
    last_layer_base = ADMM_SpikingLinear(in_f, out_f, h=h_func, **config)
    last_layer_base.device = device
    last_layer_base._setup()
    last_layer_base.T = T

    labels = torch.randn((batch, out_f))

    # Setup test copies using ADMM_LayerState!
    last_layer_vectorized = copy.deepcopy(last_layer_base)
    loss_f = ADMM_SSE()
    state_last_vec = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    last_layer_unrolled = copy.deepcopy(last_layer_base)
    state_last_unrolled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    # METHOD 1: VECTORIZED Z LAST
    start_vec_last = time.perf_counter()
    last_layer_vectorized.update_z_last(state_last_vec, a_prev, labels, loss_f=loss_f)
    time_vec_last = time.perf_counter() - start_vec_last

    # METHOD 2: UNROLLED Z LAST
    start_unrolled_last = time.perf_counter()
    last_layer_unrolled.update_z_last_unrolled(
        state_last_unrolled, a_prev, labels, time_steps, jacobi=True, loss_f=loss_f
    )
    time_unrolled_last = time.perf_counter() - start_unrolled_last

    # Verification 3
    z_last_diff = torch.max(torch.abs(state_last_vec.z - state_last_unrolled.z)).item()

    print(
        f"[{'Final Layer (z) Update':<26}] Max Difference: {z_last_diff:.8e} "
        + ("✅" if z_last_diff < 1e-9 else "❌")
    )

    if z_last_diff < 1e-9:
        speedup = (
            (time_unrolled_last / time_vec_last)
            if time_vec_last < time_unrolled_last
            else (time_vec_last / time_unrolled_last)
        )
        icon = "🚀" if time_vec_last < time_unrolled_last else "🐢"
        print(
            f"[{icon} Speed]: Vectorized method is {speedup:.2f}x {'faster' if time_vec_last < time_unrolled_last else 'slower'}."
        )
    else:
        print(
            "\nWARNING: Final Layer outputs diverged. The methods are not evaluating equally."
        )

    print("=" * 65 + "\n")
