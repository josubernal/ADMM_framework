"""
ADMM Safety Test: In-Place Aliasing (Ghost Overwrite)
"""

import torch
from admm.layers import ADMM_SpikingLinear
from admm.activations import ADMM_Heaviside

def test_inplace_aliasing():
    print("\n" + "="*55)
    print("  IN-PLACE ALIASING (GHOST OVERWRITE) TEST")
    print("="*55)
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    T, batch, feats = 5, 4, 16
    
    config = {'rho': 1.0, 'beta': 1.0, 'deltas': 0.8, 'thetas': 1.0}
    
    # 1. Setup 3 sequential layers
    layer1 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), bias=True)
    layer2 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), bias=True)
    layer3 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), bias=True)
    
    for i, layer in enumerate([layer1, layer2, layer3]):
        layer.device = device
        layer.setup(config, is_last_layer=(i == 2))
        layer.T = T
        # Initialize raw tensors
        layer._init_weights_and_bias((feats, feats), (feats,))
        layer.z = torch.randn((T, batch, feats))
        layer.a = torch.rand((T, batch, feats))
        
    # 2. Take a secure, deep-copy snapshot of Layer 1 and Layer 3
    # We use .clone() to ensure entirely new memory addresses are allocated
    l1_snap_W = layer1.W.clone()
    l1_snap_z = layer1.z.clone()
    l1_snap_a = layer1.a.clone()
    
    l3_snap_W = layer3.W.clone()
    l3_snap_z = layer3.z.clone()
    l3_snap_a = layer3.a.clone()

    # 3. Aggressively update ONLY Layer 2
    # We pass Layer 1's 'a' as input, and Layer 3 as the next layer.
    time_steps = list(range(T - 1))
    
    # Weight & Bias update
    layer2.update_weights(layer1.a, cache_pinv=False)
    layer2.update_bias(layer1.a)
    
    # Decoupled activation and pre-activation update
    layer2.update_a(layer3, layer1.a, lambda_lagrange=None)
    layer2.update_z_decoupled(layer1.a, time_steps)

    # 4. Verify Layer 1 and Layer 3 remained completely untouched
    diff_l1_W = torch.max(torch.abs(layer1.W - l1_snap_W)).item()
    diff_l1_z = torch.max(torch.abs(layer1.z - l1_snap_z)).item()
    diff_l1_a = torch.max(torch.abs(layer1.a - l1_snap_a)).item()
    
    diff_l3_W = torch.max(torch.abs(layer3.W - l3_snap_W)).item()
    diff_l3_z = torch.max(torch.abs(layer3.z - l3_snap_z)).item()
    diff_l3_a = torch.max(torch.abs(layer3.a - l3_snap_a)).item()

    l1_total_diff = diff_l1_W + diff_l1_z + diff_l1_a
    l3_total_diff = diff_l3_W + diff_l3_z + diff_l3_a

    print(f"[{'Layer 1 Stability (Input)':<26}] Max Shift: {l1_total_diff:.8e} " + ("✅" if l1_total_diff == 0 else "❌"))
    print(f"[{'Layer 3 Stability (Output)':<26}] Max Shift: {l3_total_diff:.8e} " + ("✅" if l3_total_diff == 0 else "❌"))

    if l1_total_diff == 0 and l3_total_diff == 0:
        print("\nCONCLUSION: In-place operations are safely contained. No memory aliasing detected. 🚀")
    else:
        print("\nWARNING: Ghost overwriting detected! Check your .copy_() and .view() logic.")
        
    print("="*65 + "\n")
