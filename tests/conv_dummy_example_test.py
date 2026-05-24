import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.admm.activation_functions import ADMM_Identity, ADMM_ReLU
from src.admm.dataclasses import ADMM_Config, ADMM_LayerConfig
from src.admm.layers import ADMM_Conv2d, ADMM_Linear
from src.admm.loss_functions import ADMM_SSE
from src.admm.manager import ADMM


@pytest.mark.parametrize(
    "z0_val, a0_val, z1_val, Y_val, exp_W0, exp_W1, exp_a0, exp_z0, exp_z1",
    [
        # Example 1: Active ReLU with Mixed Features
        (
            [[[[2.0]]], [[[-1.0]]], [[[-1.0]]], [[[2.0]]]],  # Initial z0 (4, 1, 1, 1)
            [[[[2.0]]], [[[0.0]]], [[[0.0]]], [[[2.0]]]],  # Initial a0
            [[3.0], [4.0], [1.0], [2.0]],  # Initial z1 (4, 1)
            [[4.0], [5.0], [1.0], [1.0]],  # Targets Y
            [[[[2.0, -1.0], [-1.0, 2.0]]]],  # Expected W0 (1, 1, 2, 2)
            [[1.25]],  # Expected W1
            [[[[92 / 41]]], [[[80 / 41]]], [[[20 / 41]]], [[[72 / 41]]]],  # Expected a0
            [[[[87 / 41]]], [[[39 / 82]]], [[[-1.0]]], [[[77 / 41]]]],  # Expected z0
            [[443 / 123], [170 / 41], [107 / 123], [172 / 123]],  # Expected z1
        ),
        # Example 2: Dead ReLU feature tracking
        (
            [[[[1.0]]], [[[-2.0]]], [[[-2.0]]], [[[1.0]]]],  # Initial z0
            [[[[1.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]]],  # Initial a0
            [[2.0], [0.0], [0.0], [2.0]],  # Initial z1
            [[2.0], [-2.0], [2.0], [-2.0]],  # Targets Y
            [[[[1.0, -2.0], [-2.0, 1.0]]]],  # Expected W0
            [[2.0]],  # Expected W1
            [[[[1.0]]], [[[0.0]]], [[[0.0]]], [[[1.0]]]],  # Expected a0
            [[[[1.0]]], [[[-2.0]]], [[[-2.0]]], [[[1.0]]]],  # Expected z0
            [[2.0], [-4 / 3], [4 / 3], [-2 / 3]],  # Expected z1
        ),
    ],
)
def test_conv2d_to_linear_admm_fit(
    z0_val, a0_val, z1_val, Y_val, exp_W0, exp_W1, exp_a0, exp_z0, exp_z1
):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # 0. Setup Batch Data (4 independent active pixels to map the 2x2 kernel)
    X = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]]],
        ],
        dtype=torch.float64,
        device=device,
    )

    Y = torch.tensor(Y_val, dtype=torch.float64, device=device)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 1. Setup Layer 0 (Conv2d)
    # Using padding_mode="zeros" to bypass FFT and force exact spatial Woodbury patches
    config0 = ADMM_LayerConfig(rho=1.0, beta=1.0, use_bias=False, use_lagrange=False)
    layer0 = ADMM_Conv2d(
        in_c=1,
        out_c=1,
        k=2,
        p=0,
        s=1,
        padding_mode="zeros",
        h=ADMM_ReLU(),
        config=config0,
    )

    # 2. Setup Layer 1 (Linear)
    # in_f=1 because the Conv2d outputs shape (4, 1, 1, 1), which ADMM_Flatten pools to (4, 1)
    config1 = ADMM_LayerConfig(rho=1.0, beta=1.0, use_bias=False, use_lagrange=False)
    layer1 = ADMM_Linear(in_f=1, out_f=1, h=ADMM_Identity(), config=config1)

    # 3. Setup Global Manager
    global_config = ADMM_Config(train_method="vectorized", layer_order="sequential")
    manager = ADMM(
        layers=nn.ModuleList([layer0, layer1]),
        loss_f=ADMM_SSE(),
        config=global_config,
        device=device,
    )

    # 4. Initialize Handlers
    manager.state_handler.initialize_all_batches(dataloader=dataloader)
    manager.initialized = True

    # 5. Inject Hand-Calculated Initial States
    batch_ids = manager.state_handler.get_batch_ids()
    _, _, batch_state = manager.state_handler.load_batch(batch_ids[0])

    batch_state.layer_states[0].z.copy_(torch.tensor(z0_val, dtype=torch.float64))
    batch_state.layer_states[0].a.copy_(torch.tensor(a0_val, dtype=torch.float64))
    batch_state.layer_states[1].z.copy_(torch.tensor(z1_val, dtype=torch.float64))

    manager.state_handler.save_batch(batch_state, inputs=X, labels=Y)

    # ==========================================
    # Execute the Real ADMM Fit Loop
    # ==========================================
    manager.fit(dataloader, warming=False)

    # ==========================================
    # Retrieve & Assert
    # ==========================================
    _, _, updated_batch = manager.state_handler.load_batch(batch_ids[0])
    updated_state0 = updated_batch.layer_states[0]
    updated_state1 = updated_batch.layer_states[1]

    # Layer 0 (Conv2D) assertions
    torch.testing.assert_close(
        layer0.W.data,
        torch.tensor(exp_W0, dtype=torch.float64, device=device),
        rtol=0.0,
        atol=1e-9,
        msg="Layer 0: Convolutional Weight (W0) update diverged.",
    )
    torch.testing.assert_close(
        updated_state0.a,
        torch.tensor(exp_a0, dtype=torch.float64, device=device),
        rtol=0.0,
        atol=1e-9,
        msg="Layer 0: Spatial Activation (a0) update diverged.",
    )
    torch.testing.assert_close(
        updated_state0.z,
        torch.tensor(exp_z0, dtype=torch.float64, device=device),
        rtol=0.0,
        atol=1e-9,
        msg="Layer 0: Spatial Pre-activation (z0) update diverged.",
    )

    # Layer 1 (Linear) assertions
    torch.testing.assert_close(
        layer1.W.data,
        torch.tensor(exp_W1, dtype=torch.float64, device=device),
        rtol=0.0,
        atol=1e-9,
        msg="Layer 1: Dense Weight (W1) update diverged.",
    )
    torch.testing.assert_close(
        updated_state1.z,
        torch.tensor(exp_z1, dtype=torch.float64, device=device),
        rtol=0.0,
        atol=1e-9,
        msg="Layer 1: SSE Pre-activation (z1) update diverged.",
    )
