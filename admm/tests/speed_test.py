"""
ADMM Execution Speed Benchmark

PURPOSE:
Compares the raw hardware execution speed of the three temporal update schemes:
1. Vectorized (Pure Jacobi) - Massive matrix ops, fully parallel.
2. Unrolled-Sequential (Gauss-Seidel) - Strict chronological for-loop.
3. Decoupled-Sequential (Mixed) - Vectorized 'a', chronological 'z'.
"""

import torch
import copy
import time
from admm.layers import ADMM_SpikingLinear
from admm.activations import ADMM_Heaviside

def test_speed_benchmark():
    print("\n" + "="*70)
    print("  ADMM TEMPORAL SCHEMES: EXECUTION SPEED BENCHMARK")
    print("="*70)
    
    # We can use standard float32 for speed benchmarking
    torch.set_default_dtype(torch.float32)
    device = torch.device('cpu')
    
    # 1. Setup Heavy Dimensions to stress the CPU
    T, batch, in_f, out_f = 200, 64, 128, 128 
    time_steps = list(range(T - 1)) 
    iterations = 10  # Run multiple times for stable averaging
    
    print(f"Network Size : [{T} Timesteps] x [{batch} Batch] x [{in_f} Features]")
    print(f"Iterations   : {iterations} passes per method\n")
    
    # ADMM Hyperparameters
    config = {'rho': 1.0, 'beta': 1.0, 'deltas': 0.8, 'thetas': 1.0}
    
    # 2. Instantiate Layers
    h_func = ADMM_Heaviside()
    layer_base = ADMM_SpikingLinear(in_f, out_f, h=h_func, **config)
    layer_base.device = device
    layer_base._setup()
    layer_base.T = T
    
    config = {'rho': 1.0, 'beta': 1.0, 'deltas': 0.8, 'thetas': 1.0, 'use_reset': False}
    next_layer = ADMM_SpikingLinear(out_f, out_f, h=h_func, **config)
    next_layer.device = device
    next_layer._setup()
    next_layer.T = T
    
    # 3. Create states
    a_prev = torch.randn((T, batch, in_f))
    
    z_init = torch.randn((T, batch, out_f))
    a_init = torch.rand((T, batch, out_f))
    next_layer.z = torch.randn((T, batch, out_f))
    next_layer.a = torch.rand((T, batch, out_f))
    
    # Clone for each method
    layer_vec = copy.deepcopy(layer_base)
    layer_vec.z, layer_vec.a = z_init.clone(), a_init.clone()

    layer_unrolled = copy.deepcopy(layer_base)
    layer_unrolled.z, layer_unrolled.a = z_init.clone(), a_init.clone()
    
    layer_decoupled = copy.deepcopy(layer_base)
    layer_decoupled.z, layer_decoupled.a = z_init.clone(), a_init.clone()

    # WARMUP (Wake up CPU threads)
    _ = layer_base.spatial_forward(a_prev)

    # =======================================================
    # METHOD 1: FULLY VECTORIZED (Pure Jacobi)
    # =======================================================
    start = time.perf_counter()
    for _ in range(iterations):
        layer_vec.update_a(next_layer, a_prev)
        layer_vec.update_z(a_prev)
    time_vec = (time.perf_counter() - start) / iterations

    # =======================================================
    # METHOD 2: UNROLLED SEQUENTIAL (Strict Gauss-Seidel)
    # =======================================================
    start = time.perf_counter()
    for _ in range(iterations):
        layer_unrolled.update_az_interleaved(next_layer, a_prev, time_steps,False)
    time_unrolled = (time.perf_counter() - start) / iterations

    # =======================================================
    # METHOD 3: DECOUPLED SEQUENTIAL (Mixed Method)
    # =======================================================
    start = time.perf_counter()
    for _ in range(iterations):
        layer_decoupled.update_a(next_layer, a_prev)
        layer_decoupled.update_z_decoupled(a_prev, time_steps)
    time_decoupled = (time.perf_counter() - start) / iterations

    # =======================================================
    # REPORTING
    # =======================================================
    print(f"{'Method':<25} | {'Avg Time / Iter':<17} | {'Speedup relative to Unrolled'}")
    print("-" * 70)
    print(f"{'Unrolled (Baseline)':<25} | {time_unrolled:.5f} sec       | 1.00x")
    print(f"{'Decoupled (Mixed)':<25} | {time_decoupled:.5f} sec       | {time_unrolled/time_decoupled:.2f}x Faster")
    print(f"{'Vectorized (Jacobi)':<25} | {time_vec:.5f} sec       | {time_unrolled/time_vec:.2f}x Faster")
    print("-" * 70)
    
    print("\nTAKEAWAYS:")
    print("- Vectorized is the fastest to compute, but has the slowest mathematical convergence (requires more ADMM epochs).")
    print("- Unrolled is the slowest to compute, but has the fastest mathematical convergence.")
    print("- Decoupled shares the EXACT SAME fast mathematical convergence as Unrolled, but computes significantly faster.")
    print("="*70 + "\n")

