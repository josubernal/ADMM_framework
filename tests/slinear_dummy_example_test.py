import pytest
import torch
import torch.nn as nn

from src.admm.activation_functions import ADMM_Heaviside, ADMM_Identity
from src.admm.dataclasses import ADMM_Config, ADMM_LayerConfig
from src.admm.layers import ADMM_SpikingFeedForward
from src.admm.loss_functions import ADMM_SSE
from src.admm.manager import ADMM


@pytest.mark.parametrize("method", ["vectorized", "unrolled", "decoupled"])
@pytest.mark.parametrize(
    "case",
    [
        # Example 1: Active Sequence (All layers spiking)
        {
            "z0_val": [[[2.0, 2.0]], [[2.0, 2.0]]],
            "a0_val": [[[1.0, 1.0]], [[1.0, 1.0]]],
            "z1_val": [[[0.0, 0.0]], [[0.0, 0.0]]],
            "Y_val": [[3.0, 3.0]],
            "exp_W0": [[2.0, 2.0], [2.0, 2.0]],
            "exp_W1": [[0.0, 0.0], [0.0, 0.0]],
            "exp_a0": [[[1.0, 1.0]], [[1.0, 1.0]]],
            "exp_z0": [[[2.0, 2.0]], [[2.0, 2.0]]],
            "exp_z1": [[[0.0, 0.0]], [[2.0, 2.0]]],
        },
        # Example 2: Dead-to-Active Sequence
        {
            "z0_val": [[[0.0, 0.0]], [[2.0, 2.0]]],
            "a0_val": [[[0.0, 0.0]], [[1.0, 1.0]]],
            "z1_val": [[[0.0, 0.0]], [[0.0, 0.0]]],
            "Y_val": [[6.0, 6.0]],
            "exp_W0": [[0.0, 2.0], [0.0, 2.0]],
            "exp_W1": [[0.0, 0.0], [0.0, 0.0]],
            "exp_a0": [[[0.0, 0.0]], [[1.0, 1.0]]],
            "exp_z0": [[[0.0, 0.0]], [[2.0, 2.0]]],
            "exp_z1": [[[0.0, 0.0]], [[4.0, 4.0]]],
        },
    ],
)
def test_all_methods_equivalence(method, case):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # Standard setup
    X = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64)
    Y = torch.tensor(case["Y_val"], dtype=torch.float64)
    dataloader = [(X, Y)]

    layer0 = ADMM_SpikingFeedForward(
        in_f=2,
        out_f=2,
        h=ADMM_Heaviside(1.0),
        config=ADMM_LayerConfig(
            rho=1.0, beta=1.0, deltas=0.5, thetas=1.0, use_reset=True
        ),
    )
    layer1 = ADMM_SpikingFeedForward(
        in_f=2,
        out_f=2,
        h=ADMM_Identity(),
        config=ADMM_LayerConfig(
            rho=1.0, beta=1.0, deltas=0.0, thetas=0.0, use_reset=False
        ),
    )

    manager = ADMM(
        nn.ModuleList([layer0, layer1]),
        loss_f=ADMM_SSE(),
        T=2,
        config=ADMM_Config(
            train_method=method, layer_order="sequential", time_order="sequential"
        ),
        device=device,
    )

    # Init and Inject
    manager.state_handler.initialize_all_batches(dataloader=dataloader)
    manager.initialized = True
    _, _, batch_state = manager.state_handler.load_batch(0)
    batch_state.layer_states[0].z.copy_(torch.tensor(case["z0_val"]))
    batch_state.layer_states[0].a.copy_(torch.tensor(case["a0_val"]))
    batch_state.layer_states[1].z.copy_(torch.tensor(case["z1_val"]))
    manager.state_handler.save_batch(batch_state, inputs=X, labels=Y)

    # Execute
    manager.fit(dataloader, warming=False)

    # Assert
    _, _, final_batch = manager.state_handler.load_batch(0)
    torch.testing.assert_close(
        layer0.W.data,
        torch.tensor(case["exp_W0"], dtype=torch.float64),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        layer1.W.data,
        torch.tensor(case["exp_W1"], dtype=torch.float64),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        final_batch.layer_states[0].a,
        torch.tensor(case["exp_a0"], dtype=torch.float64),
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        final_batch.layer_states[0].z,
        torch.tensor(case["exp_z0"], dtype=torch.float64),
        rtol=0,
        atol=1e-7,
    )

    torch.testing.assert_close(
        final_batch.layer_states[1].z,
        torch.tensor(case["exp_z1"], dtype=torch.float64),
        rtol=0,
        atol=1e-7,
    )
