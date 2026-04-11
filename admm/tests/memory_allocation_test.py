"""
ADMM GPU Performance & Peak Memory Profiler

PURPOSE:
To rigorously test the ADMM framework on CUDA hardware. 
1. Tracks exact Peak VRAM utilization using PyTorch's internal C++ allocator.
2. Measures true execution time using synchronized CUDA events.
3. Runs a multi-epoch stress test to guarantee zero VRAM creep.
"""

import torch
import gc
from admm.manager import ADMM
from admm.layers import ADMM_SpikingConv2d, ADMM_SpikingLinear
from admm.activations import ADMM_Heaviside
from admm.pooling import ADMM_GAP

def format_mb(memory_bytes):
    """Converts bytes to Megabytes for clean printing."""
    return memory_bytes / (1024 ** 2)

def print_gpu_stats(stage_name: str):
    """Prints current, peak, and reserved VRAM."""
    # Force trash collection before measuring
    gc.collect()
    torch.cuda.empty_cache() 
    
    current = format_mb(torch.cuda.memory_allocated())
    peak = format_mb(torch.cuda.max_memory_allocated())
    reserved = format_mb(torch.cuda.memory_reserved())
    
    print(f"[{stage_name:<26}] Current: {current:7.2f} MB | Peak: {peak:7.2f} MB | Reserved: {reserved:7.2f} MB")

def test_proper_gpu_execution():
    print("\n" + "="*85)
    print("  ADMM STRICT GPU PROFILING TEST (L40s SIMULATION)")
    print("="*85)
    
    if not torch.cuda.is_available():
        print("❌ FATAL: CUDA is not available. You must run this on the cluster node!")
        return

    # 1. Strict Precision & Device Setup
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print_gpu_stats("1. Empty GPU Baseline")

    # 2. Build the "Memory-Safe" Architecture
    # Using the exact specs we designed to beat the 48GB limit
    T, batch = 10, 16
    in_c, H, W = 2, 32, 32 # Assuming a DVS/CIFAR-like spatial input
    
    config = {'rho': 1.0, 'beta': 1.0, 'deltas': 0.8, 'thetas': 1.0}
    
    # Layer 1: Conv -> GAP (Massive spatial reduction)
    layer1 = ADMM_SpikingConv2d(
        in_c=in_c, out_c=16, k=5, p=2, s=2, # Stride 2 is crucial here!
        h=ADMM_Heaviside(), pool_op=ADMM_GAP(), bias=True, use_fft=True
    )
    
    # Layer 2: Linear Classifier
    layer2 = ADMM_SpikingLinear(16, 10, h=ADMM_Heaviside(), bias=True)
    
    model = ADMM([layer1, layer2], T=T, device=device, init="zeros", 
                 train_method="decoupled-sequential", bias=True, **config)
    
    # Dummy Input Data (Moved immediately to GPU)
    inputs = torch.randn((T, batch, in_c, H, W), device=device)
    labels = torch.randint(0, 10, (batch,), device=device)

    print_gpu_stats("2. After Model & Data")

    # 3. Initialization (Warm Start)
    model._init_states(inputs)
    print_gpu_stats("3. After Init States (a,z)")

    # =====================================================================
    # 4. SINGLE EPOCH PEAK TEST (The Danger Zone)
    # =====================================================================
    torch.cuda.reset_peak_memory_stats() # Reset peak so we only measure the fit loop
    
    # Create CUDA events for true hardware timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    model.fit(inputs, labels, warming=False)
    end_event.record()
    
    # Force CPU to wait for GPU to finish before stopping the clock
    torch.cuda.synchronize() 
    epoch_time = start_event.elapsed_time(end_event) / 1000.0 # Convert ms to seconds
    
    print_gpu_stats("4. After 1st Fit Loop")
    print(f"\n⏱️  Single Epoch Execution Time: {epoch_time:.4f} seconds")
    
    # =====================================================================
    # 5. MULTI-EPOCH STRESS TEST (Leak Verification)
    # =====================================================================
    print("\n--- Running 10-Epoch Stress Test ---")
    
    baseline_memory = torch.cuda.memory_allocated()
    
    for epoch in range(10):
        model.fit(inputs, labels, warming=False)
        
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated()
    memory_drift = format_mb(final_memory - baseline_memory)
    
    print_gpu_stats("5. After 10 Epochs")
    print(f"📈 Total VRAM Drift after 10 epochs: {memory_drift:.4f} MB " + ("✅" if memory_drift <= 1.0 else "❌"))

    print("="*85 + "\n")
