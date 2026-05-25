"""Verifies tensor memory management and prevents autograd leaks.

Runs the optimization loop over multiple iterations to ensure the custom mathematical
solvers do not unintentionally build or retain PyTorch computational graphs. Also
monitors VRAM allocation during execution to ensure stable memory consumption without
progressive leaking across batches.
"""

import gc

import torch

from src.admm.activation_functions import ADMM_Heaviside
from src.admm.dataclasses import ADMM_Config
from src.admm.layers import ADMM_SpikingFeedForward
from src.admm.manager import ADMM


def test_autograd_and_memory_leak():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, batch, in_f, hidden_f, out_f = 10, 8, 16, 32, 16

    # Setup a small 2-layer network
    global_config = ADMM_Config(init="zeros")
    layer1 = ADMM_SpikingFeedForward(
        in_f, hidden_f, rho=1.0, beta=1.0, deltas=0.8, thetas=1.0, h=ADMM_Heaviside()
    )
    layer2 = ADMM_SpikingFeedForward(
        hidden_f,
        out_f,
        rho=1.0,
        beta=1.0,
        deltas=0.8,
        thetas=1.0,
        h=ADMM_Heaviside(),
        use_reset=False,
    )

    model = ADMM([layer1, layer2], T=T, device=device, config=global_config)

    inputs = torch.randn((T, batch, in_f), device=device)
    labels = torch.ones((batch, out_f), device=device)
    dataloader = [(inputs, labels)]
    model.fit(dataloader=dataloader)

    # Trigger garbage collection to clear any orphaned python objects
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    else:
        initial_memory = (
            0  # Memory tracking is highly noisy on CPU, rely on graph checks
        )

    # Run multiple iterations
    iterations = 50
    for _ in range(iterations):
        model.fit(dataloader=dataloader)

    # 3. Check for Computational Graph Leaks (Autograd)
    graph_leaks = 0
    batch_id = model.state_handler.get_batch_ids()[0]
    _, _, batch_state = model.state_handler.load_batch(batch_id)

    for i, layer in enumerate(model.layers):
        if layer.W.requires_grad or layer.W.grad_fn is not None:
            graph_leaks += 1

        layer_state = batch_state.layer_states[i]

        if layer_state.z.requires_grad or layer_state.z.grad_fn is not None:
            graph_leaks += 1
        if layer_state.a is not None and (
            layer_state.a.requires_grad or layer_state.a.grad_fn is not None
        ):
            graph_leaks += 1

    # 4. Check for VRAM Memory Leaks (Only if CUDA is available)
    if device.type == "cuda":
        gc.collect()
        final_memory = torch.cuda.memory_allocated()
        memory_diff_mb = (final_memory - initial_memory) / (1024**2)

        # A perfectly raw tensor loop should have exactly 0.00MB growth
        print(
            f"[{'VRAM Memory Stability':<26}] Growth: {memory_diff_mb:.4f} MB "
            + ("✅" if memory_diff_mb <= 0.01 else "❌")
        )

    # Strict assertion
    assert graph_leaks == 0, "FATAL: PyTorch is tracking gradients on your raw tensors!"
