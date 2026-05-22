"""Verifies the equivalence of sequential and decoupled temporal updates.

Checks that executing the optimization loop sequentially across time steps
(unrolled) yields the exact same internal states (activations and pre-activations)
and parameters as computing the dynamics in a single, parallelized pass (decoupled).
"""

import copy

import torch

from src.admm.activation_functions import ADMM_Heaviside
from src.admm.dataclasses import ADMM_LayerState, TemporalCache
from src.admm.layers import ADMM_SpikingLinear


def test_temporal_schemes_equivalence():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # Set experiments
    T, BATCH, IN_F, OUT_F = 6, 4, 16, 16
    time_steps = list(range(T))

    config = {"rho": 1.0, "beta": 1.0, "deltas": 0.8, "thetas": 1.0}
    h_func = ADMM_Heaviside()

    layer_base = ADMM_SpikingLinear(IN_F, OUT_F, h=h_func, **config)
    layer_base.device = device
    layer_base._setup()
    layer_base.T = T

    config_next = {
        "rho": 1.0,
        "beta": 1.0,
        "deltas": 0.8,
        "thetas": 1.0,
        "use_reset": False,
    }
    next_layer = ADMM_SpikingLinear(OUT_F, OUT_F, h=h_func, **config_next)
    next_layer.device = device
    next_layer._setup()
    next_layer.T = T

    a_prev = torch.randn((T, BATCH, IN_F))
    z_init = torch.randn((T, BATCH, OUT_F))
    a_init = torch.rand((T, BATCH, OUT_F))

    next_state = ADMM_LayerState(
        z=torch.randn((T, BATCH, OUT_F)), a=torch.rand((T, BATCH, OUT_F))
    )

    layer_unrolled = copy.deepcopy(layer_base)
    state_unrolled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    layer_decoupled = copy.deepcopy(layer_base)
    state_decoupled = ADMM_LayerState(z=z_init.clone(), a=a_init.clone())

    # Unrolled sequential
    cache = TemporalCache.build(
        layer=layer_unrolled,
        next_layer=next_layer,
        next_state=next_state,
        state=state_unrolled,
        a_prev=a_prev,
    )
    for t in time_steps:
        layer_unrolled.update_a_unrolled(
            t=t, cache=cache, next_layer=next_layer, state=state_unrolled
        )
        layer_unrolled.h.update_z_unrolled(
            t=t, cache=cache, state=state_unrolled, config=layer_unrolled.config
        )

    # Decoupled sequential
    forward = layer_decoupled.spatial_forward(a_prev)
    layer_decoupled.update_a(
        next_layer=next_layer,
        next_state=next_state,
        state=state_decoupled,
        a_prev=a_prev,
    )
    layer_decoupled.h.update_z_decoupled(
        state=state_decoupled,
        forward=forward,
        time_steps=time_steps,
        config=layer_decoupled.config,
    )

    # Assertions
    torch.testing.assert_close(
        state_unrolled.a,
        state_decoupled.a,
        rtol=0.0,
        atol=1e-9,
        msg="Activation (a) Update diverged",
    )
    torch.testing.assert_close(
        state_unrolled.z,
        state_decoupled.z,
        rtol=0.0,
        atol=1e-9,
        msg="Pre-activation (z) Update diverged",
    )
