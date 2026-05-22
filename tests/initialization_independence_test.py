"""Verifies parameter independence during network initialization.

Ensures that parameter initialization does not affect to the result
if states are initialized equally, when bias is turned off.
"""

import pytest
import torch

from src.admm.activation_functions import ADMM_ReLU
from src.admm.dataclasses import (
    ADMM_Config,
    ADMM_LayerConfig,
    ADMM_LayerCovariance,
    ADMM_LayerState,
)
from src.admm.layers import ADMM_Linear

BATCH, IN_F, OUT_F = 16, 32, 32


@pytest.fixture
def init_data():
    torch.set_default_dtype(torch.float64)
    a_prev = torch.randn((BATCH, IN_F), dtype=torch.float64)
    z_target = torch.randn((BATCH, OUT_F), dtype=torch.float64)
    a_dummy = torch.zeros_like(z_target)
    return a_prev, z_target, a_dummy


def test_parameter_initialization_independence_no_bias(init_data):
    a_prev, z_target, a_dummy = init_data
    config = ADMM_LayerConfig(use_bias=False)
    global_config = ADMM_Config()

    # Setup layer states equally
    layer_z = ADMM_Linear(IN_F, OUT_F, h=ADMM_ReLU(), config=config)
    layer_r = ADMM_Linear(IN_F, OUT_F, h=ADMM_ReLU(), config=config)
    layer_z._setup(global_config=global_config)
    layer_r._setup(global_config=global_config)

    # Set different weights
    layer_z.W.data = torch.zeros_like(layer_z.W)
    layer_r.W.data = torch.randn_like(layer_r.W) * 100.0

    state_z = ADMM_LayerState(z=z_target.clone(), a=a_dummy.clone())
    state_r = ADMM_LayerState(z=z_target.clone(), a=a_dummy.clone())

    for _ in range(10):
        # Update layer z
        num_z, den_z, b_sum_z, b_count_z = layer_z.compute_batch_covariances(
            state=state_z, a_prev=a_prev
        )
        cov_z = ADMM_LayerCovariance(
            numerator=num_z, denominator=den_z, bias_sum=b_sum_z, bias_count=b_count_z
        )
        layer_z.update_weights(covariances=cov_z)

        # Update layer r
        num_r, den_r, b_sum_r, b_count_r = layer_r.compute_batch_covariances(
            state=state_r, a_prev=a_prev
        )
        cov_r = ADMM_LayerCovariance(
            numerator=num_r, denominator=den_r, bias_sum=b_sum_r, bias_count=b_count_r
        )
        layer_r.update_weights(covariances=cov_r)

    torch.testing.assert_close(
        layer_z.W, layer_r.W, rtol=0.0, atol=1e-9, msg="Coupled W Divergence"
    )
