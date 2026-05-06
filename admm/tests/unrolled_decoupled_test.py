"""
ADMM Temporal Schemes Verification (Unrolled vs. Decoupled)

PURPOSE:
Verifies that performing a strictly chronological Gauss-Seidel unrolled sweep 
(interleaved a and z) produces mathematically identical results to the 
Decoupled approach (fully vectorized 'a' update followed by a causal 'z' sweep).
"""

import torch
import copy
from admm.layers import ADMM_SpikingLinear
from admm.activations import ADMM_Heaviside
import time

def test_temporal_schemes_equivalence():
    print("\n" + "="*55)
    print("  UNROLLED VS DECOUPLED SCHEME VERIFICATION TEST")
    print("="*55)
    
    # Force double precision to verify exact mathematical equivalence
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    
    # 1. Setup dimensions and time sequence
    T, batch, in_f, out_f = 6, 4, 16, 16
    time_steps = list(range(T)) # Strict chronological forward sweep
    
    # ADMM Hyperparameters
    config = {'rho': 1.0, 'beta': 1.0, 'deltas': 0.8, 'thetas': 1.0}
    
    # 2. Instantiate Base Layers
    h_func = ADMM_Heaviside()
    layer_base = ADMM_SpikingLinear(in_f, out_f, h=h_func, bias=False)
    layer_base.device = device
    layer_base.setup(config)
    layer_base.T = T
    
    next_layer = ADMM_SpikingLinear(out_f, out_f, h=h_func, bias=False)
    next_layer.device = device
    next_layer.setup(config, is_last_layer=True)
    next_layer.T = T
    
    # 3. Create identical random states for the environment
    a_prev = torch.randn((T, batch, in_f))

    # Initialize random auxiliary variables to ensure they start from the exact same point
    z_init = torch.randn((T, batch, out_f))
    a_init = torch.rand((T, batch, out_f)) # 'a' is usually [0, 1] bounded
    
    next_layer.z = torch.randn((T, batch, out_f))
    next_layer.a = torch.rand((T, batch, out_f))
    
    # 4. Create identical parallel layers for the test
    layer_unrolled = copy.deepcopy(layer_base)
    layer_unrolled.z = z_init.clone()
    layer_unrolled.a = a_init.clone()
    
    layer_decoupled = copy.deepcopy(layer_base)
    layer_decoupled.z = z_init.clone()
    layer_decoupled.a = a_init.clone()

    # =======================================================
    # EXECUTE METHOD 1: UNROLLED SEQUENTIAL (Gauss-Seidel)
    # =======================================================
    start_unrolled = time.perf_counter()
    layer_unrolled.update_az_interleaved(next_layer, a_prev, time_steps,False)
    time_unrolled = time.perf_counter() - start_unrolled
    
    # =======================================================
    # EXECUTE METHOD 2: DECOUPLED SEQUENTIAL (Mixed Jacobi)
    # =======================================================
    start_decoupled = time.perf_counter()
    layer_decoupled.update_a(next_layer, a_prev)
    layer_decoupled.update_z_decoupled(a_prev, time_steps)
    time_decoupled = time.perf_counter() - start_decoupled
    
    # =======================================================
    # VERIFICATION
    # =======================================================
    a_diff = torch.max(torch.abs(layer_unrolled.a - layer_decoupled.a)).item()
    z_diff = torch.max(torch.abs(layer_unrolled.z - layer_decoupled.z)).item()
    
    print(f"[{'Activation (a) Update':<26}] Max Difference: {a_diff:.8e} " + ("✅" if a_diff < 1e-9 else "❌"))
    print(f"[{'Pre-activation (z) Update':<26}] Max Difference: {z_diff:.8e} " + ("✅" if z_diff < 1e-9 else "❌"))
    
    if z_diff < 1e-9 and a_diff < 1e-9:
        if time_decoupled < time_unrolled:
            speedup = (time_unrolled / time_decoupled)
            print(f"\n🚀 Decoupled method is {speedup:.2f}x faster!")
        else:
            speedup = (time_decoupled/time_unrolled )
            print(f"\n🐢 Decoupled method is {speedup:.2f}x slower!")
    else:
        print("\nWARNING: Outputs diverged. The methods are not evaluating equally.")
        
    print("="*65 + "\n")
