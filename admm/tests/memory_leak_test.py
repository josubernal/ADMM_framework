"""
ADMM Safety Test: Autograd and Memory Leak Verification
"""

import torch
import gc
from admm.manager import ADMM
from admm.layers import ADMM_SpikingLinear
from admm.activations import ADMM_Heaviside

def test_autograd_and_memory_leak():
    print("\n" + "="*55)
    print("  AUTOGRAD & MEMORY LEAK SAFETY TEST")
    print("="*55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, batch, in_f, hidden_f, out_f = 10, 8, 16, 32, 16
    
    # Setup a small 2-layer network
    config = {'rho': 1.0, 'beta': 1.0, 'deltas': 0.8, 'thetas': 1.0}
    layer1 = ADMM_SpikingLinear(in_f, hidden_f, h=ADMM_Heaviside(), bias=True)
    layer2 = ADMM_SpikingLinear(hidden_f, out_f, h=ADMM_Heaviside(), bias=True)
    
    model = ADMM([layer1, layer2], T=T, device=device, init="zeros", **config)
    
    inputs = torch.randn((T, batch, in_f), device=device)
    labels = torch.ones((batch, out_f), device=device)

    # 1. Warmup and initial state capture
    model.fit(inputs, labels)
    
    # Trigger garbage collection to clear any orphaned python objects
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    else:
        initial_memory = 0 # Memory tracking is highly noisy on CPU, we'll rely on graph checks

    # 2. Run multiple iterations
    iterations = 50
    for _ in range(iterations):
        model.fit(inputs, labels)

    # 3. Check for Computational Graph Leaks (Autograd)
    graph_leaks = 0
    for i, layer in enumerate(model.layers):
        if layer.W.requires_grad or layer.W.grad_fn is not None: graph_leaks += 1
        if layer.z.requires_grad or layer.z.grad_fn is not None: graph_leaks += 1
        if layer.a.requires_grad or layer.a.grad_fn is not None: graph_leaks += 1

    print(f"[{'Autograd Graph Check':<26}] Leaks found: {graph_leaks} " + ("✅" if graph_leaks == 0 else "❌"))

    # 4. Check for VRAM Memory Leaks (Only if CUDA is available)
    if device.type == 'cuda':
        gc.collect()
        final_memory = torch.cuda.memory_allocated()
        memory_diff_mb = (final_memory - initial_memory) / (1024 ** 2)
        
        # A perfectly raw tensor loop should have exactly 0.00MB growth
        print(f"[{'VRAM Memory Stability':<26}] Growth: {memory_diff_mb:.4f} MB " + ("✅" if memory_diff_mb <= 0.01 else "❌"))

    assert graph_leaks == 0, "FATAL: PyTorch is tracking gradients on your raw tensors!"
    print("="*55 + "\n")
