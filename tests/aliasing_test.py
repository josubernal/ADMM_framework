"""
ADMM Safety Test: In-Place Aliasing (Ghost Overwrite)
"""

import torch

from admm.activation_functions import ADMM_Heaviside
from admm.dataclasses import ADMM_LayerConfig, ADMM_LayerState  # <-- Added Import
from admm.layers import ADMM_SpikingLinear


def test_inplace_aliasing():
    print("\n" + "=" * 55)
    print("  IN-PLACE ALIASING (GHOST OVERWRITE) TEST")
    print("=" * 55)

    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    T, batch, feats = 5, 4, 16

    config = ADMM_LayerConfig(rho=1.0, beta=1.0, deltas=0.8, thetas=1.0, use_bias=True)

    # 1. Setup 3 sequential layers (Stateless!)
    layer1 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), config=config)
    layer2 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), config=config)
    layer3 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), config=config)

    for i, layer in enumerate([layer1, layer2, layer3]):
        layer.device = device
        if i == 2:
            layer.config.use_reset = False
        layer._setup()
        layer.T = T
        # Initialize raw weights ONLY
        layer._init_weights_and_bias((feats, feats), (feats,))

    # 2. Create the "dumb" state objects instead of injecting into the layers
    l1_state = ADMM_LayerState(
        z=torch.randn((T, batch, feats)), a=torch.rand((T, batch, feats))
    )
    l2_state = ADMM_LayerState(
        z=torch.randn((T, batch, feats)), a=torch.rand((T, batch, feats))
    )
    l3_state = ADMM_LayerState(
        z=torch.randn((T, batch, feats)), a=torch.rand((T, batch, feats))
    )

    # Take a secure, deep-copy snapshot
    l1_snap_W = layer1.W.clone()
    l1_snap_z = l1_state.z.clone()
    l1_snap_a = l1_state.a.clone()

    l3_snap_W = layer3.W.clone()
    l3_snap_z = l3_state.z.clone()
    l3_snap_a = l3_state.a.clone()

    # 3. Aggressively update ONLY Layer 2
    time_steps = list(range(T - 1))

    # Explicitly pass the state objects and a_prev!
    layer2.update_weights(state=l2_state, a_prev=l1_state.a, cache_pinv=False)
    layer2.update_bias(state=l2_state, a_prev=l1_state.a)

    layer2.update_a(
        next_layer=layer3, next_state=l3_state, state=l2_state, a_prev=l1_state.a
    )
    layer2.update_z_decoupled(state=l2_state, a_prev=l1_state.a, time_steps=time_steps)

    # 4. Verify Layer 1 and Layer 3 remained completely untouched
    diff_l1_W = torch.max(torch.abs(layer1.W - l1_snap_W)).item()
    diff_l1_z = torch.max(torch.abs(l1_state.z - l1_snap_z)).item()
    diff_l1_a = torch.max(torch.abs(l1_state.a - l1_snap_a)).item()

    diff_l3_W = torch.max(torch.abs(layer3.W - l3_snap_W)).item()
    diff_l3_z = torch.max(torch.abs(l3_state.z - l3_snap_z)).item()
    diff_l3_a = torch.max(torch.abs(l3_state.a - l3_snap_a)).item()

    l1_total_diff = diff_l1_W + diff_l1_z + diff_l1_a
    l3_total_diff = diff_l3_W + diff_l3_z + diff_l3_a

    print(
        f"[{'Layer 1 Stability (Input)':<26}] Max Shift: {l1_total_diff:.8e} "
        + ("✅" if l1_total_diff == 0 else "❌")
    )
    print(
        f"[{'Layer 3 Stability (Output)':<26}] Max Shift: {l3_total_diff:.8e} "
        + ("✅" if l3_total_diff == 0 else "❌")
    )

    if l1_total_diff == 0 and l3_total_diff == 0:
        print(
            "\nCONCLUSION: In-place operations are safely contained. No memory aliasing detected. 🚀"
        )
    else:
        print(
            "\nWARNING: Ghost overwriting detected! Check your .copy_() and .view() logic."
        )
    print("=" * 65 + "\n")
