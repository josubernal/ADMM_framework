"""Validates the memory independence of layer states and parameters.

Ensures that in-place tensor operations and shared state objects do not cause memory
aliasing between independent layers. Specifically verifies that updating the
covariances, weights, and activations of one layer leaves the state and parameters
of adjacent layers completely untouched.
"""

import torch

from src.admm.activation_functions import ADMM_Heaviside
from src.admm.dataclasses import (
    ADMM_Config,
    ADMM_LayerConfig,
    ADMM_LayerCovariance,
    ADMM_LayerState,
)
from src.admm.layers import ADMM_SpikingLinear


def test_inplace_aliasing():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    T, batch, feats = 5, 4, 16

    config = ADMM_LayerConfig(rho=1.0, beta=1.0, deltas=0.8, thetas=1.0, use_bias=True)
    global_config = ADMM_Config()

    #  Setup 3 layers
    layer1 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), config=config)
    layer2 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), config=config)
    layer3 = ADMM_SpikingLinear(feats, feats, h=ADMM_Heaviside(), config=config)

    for i, layer in enumerate([layer1, layer2, layer3]):
        layer.device = device
        if i == 2:
            layer.config.use_reset = False
        layer._setup(global_config=global_config)
        layer.T = T
        layer._init_weights_and_bias((feats, feats), (feats,))

    # Create the state objects
    l1_state = ADMM_LayerState(
        z=torch.randn((T, batch, feats)), a=torch.rand((T, batch, feats))
    )
    l2_state = ADMM_LayerState(
        z=torch.randn((T, batch, feats)), a=torch.rand((T, batch, feats))
    )
    l3_state = ADMM_LayerState(
        z=torch.randn((T, batch, feats)), a=torch.rand((T, batch, feats))
    )

    # Take a snapshot
    l1_snap_W = layer1.W.clone()
    l1_snap_z = l1_state.z.clone()
    l1_snap_a = l1_state.a.clone()

    l3_snap_W = layer3.W.clone()
    l3_snap_z = l3_state.z.clone()
    l3_snap_a = l3_state.a.clone()

    # Update ONLY Layer 2
    time_steps = list(range(T - 1))

    num_l2, den_l2, b_sum_l2, b_count_l2 = layer2.compute_batch_covariances(
        state=l2_state, a_prev=l1_state.a
    )
    cov_l2 = ADMM_LayerCovariance(
        numerator=num_l2, denominator=den_l2, bias_sum=b_sum_l2, bias_count=b_count_l2
    )
    layer2.update_weights(covariances=cov_l2, cache_pinv=False)
    layer2.update_bias(covariances=cov_l2)

    layer2.update_a(
        next_layer=layer3, next_state=l3_state, state=l2_state, a_prev=l1_state.a
    )

    forward_l2 = layer2.spatial_forward(l1_state.a)
    layer2.h.update_z_decoupled(
        state=l2_state,
        forward=forward_l2,
        time_steps=time_steps,
        config=layer2.config,
    )

    # Verify Layer 1 and Layer 3 remained completely untouched
    diff_l1_W = torch.max(torch.abs(layer1.W - l1_snap_W)).item()
    diff_l1_z = torch.max(torch.abs(l1_state.z - l1_snap_z)).item()
    diff_l1_a = torch.max(torch.abs(l1_state.a - l1_snap_a)).item()

    diff_l3_W = torch.max(torch.abs(layer3.W - l3_snap_W)).item()
    diff_l3_z = torch.max(torch.abs(l3_state.z - l3_snap_z)).item()
    diff_l3_a = torch.max(torch.abs(l3_state.a - l3_snap_a)).item()

    l1_total_diff = diff_l1_W + diff_l1_z + diff_l1_a
    l3_total_diff = diff_l3_W + diff_l3_z + diff_l3_a

    # Strict Assertions for CI/CD
    assert l1_total_diff == 0.0, "FATAL: In-place memory aliasing altered Layer 1!"
    assert l3_total_diff == 0.0, "FATAL: In-place memory aliasing altered Layer 3!"
