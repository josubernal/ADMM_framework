"""
ADMM GPU Performance & Peak Memory Profiler (Multi-Architecture)

PURPOSE:
To rigorously test the ADMM framework on CUDA hardware across 4 architectures:
1. Spiking-Conv
2. Spiking-Linear
3. Static-Conv
4. Static-Linear

Tracks exact Peak VRAM utilization and true execution time.
"""

import torch
import gc
from admm.manager import ADMM
from admm.layers import ADMM_SpikingConv2d, ADMM_SpikingLinear, ADMM_Conv2d, ADMM_Linear
from admm.activations import ADMM_Heaviside, ADMM_ReLU  # Assuming ReLU for non-spiking
from admm.pooling import ADMM_GAP, ADMM_Flatten

def format_mb(memory_bytes):
    """Converts bytes to Megabytes for clean printing."""
    return memory_bytes / (1024 ** 2)

def print_gpu_stats(stage_name: str):
    """Prints current, peak, and reserved VRAM."""
    gc.collect()
    torch.cuda.empty_cache() 
    
    current = format_mb(torch.cuda.memory_allocated())
    peak = format_mb(torch.cuda.max_memory_allocated())
    reserved = format_mb(torch.cuda.memory_reserved())
    
    print(f"[{stage_name:<26}] Current: {current:7.2f} MB | Peak: {peak:7.2f} MB | Reserved: {reserved:7.2f} MB")

def profile_architecture(arch_name, layers, inputs, labels, device, T_val):
    """Runs the full memory and time profiling suite for a given architecture."""
    print("\n" + "="*85)
    print(f"  PROFILING: {arch_name.upper()}")
    print("="*85)
    
    # 1. Reset everything
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_gpu_stats("1. Empty GPU Baseline")

    # 2. Build Model
    model = ADMM(layers, T=T_val, device=device, init="zeros", 
                 train_method="decoupled-sequential", bias=True, 
                 deltas=0.8, thetas=1.0, rho=1.0, beta=1.0)
    
    print_gpu_stats("2. After Model & Data")

    # 3. Initialization
    model._init_states(inputs)
    print_gpu_stats("3. After Init States (a,z)")

    # 4. Single Epoch Peak Test
    torch.cuda.reset_peak_memory_stats() 
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    model.fit(inputs, labels, warming=False)
    end_event.record()
    
    torch.cuda.synchronize() 
    epoch_time = start_event.elapsed_time(end_event) / 1000.0 
    
    print_gpu_stats("4. After 1st Fit Loop")
    print(f"\n⏱️  Single Epoch Execution Time: {epoch_time:.4f} seconds")
    
    # 5. Leak Verification Test
    print("\n--- Running 10-Epoch Stress Test ---")
    baseline_memory = torch.cuda.memory_allocated()
    
    for _ in range(10):
        model.fit(inputs, labels, warming=False)
        
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated()
    memory_drift = format_mb(final_memory - baseline_memory)
    
    print_gpu_stats("5. After 10 Epochs")
    print(f"📈 Total VRAM Drift after 10 epochs: {memory_drift:.4f} MB " + ("✅" if memory_drift <= 1.0 else "❌"))

def run_all_profiles():
    if not torch.cuda.is_available():
        print("❌ FATAL: CUDA is not available. You must run this on the cluster node!")
        return

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda')
    
    # Shared Dimensions
    T, batch = 10, 16
    in_c, H, W = 2, 32, 32 
    flat_dim = in_c * H * W  # 2048
    classes = 10
    hidden_dim = 256 # For linear models

    labels = torch.randint(0, classes, (batch,), device=device)

    # -----------------------------------------------------------------
    # 1. SPIKING CONVOLUTIONAL
    # -----------------------------------------------------------------
    inputs_s_conv = torch.randn((T, batch, in_c, H, W), device=device)
    layers_s_conv = [
        ADMM_SpikingConv2d(in_c, 16, k=5, p=2, s=2, h=ADMM_Heaviside(), bias=True, use_fft=False, padding_mode="zeros"),
        ADMM_SpikingLinear(16, classes, pool_op=ADMM_GAP(), h=ADMM_Heaviside(), bias=True)
    ]
    profile_architecture("Spiking Conv", layers_s_conv, inputs_s_conv, labels, device, T_val=T)

    # -----------------------------------------------------------------
    # 2. SPIKING LINEAR
    # -----------------------------------------------------------------
    # Linear needs flat inputs (T, Batch, Features)
    inputs_s_lin = torch.randn((T, batch, flat_dim), device=device)
    layers_s_lin = [
        ADMM_SpikingLinear(flat_dim, hidden_dim, h=ADMM_Heaviside(), bias=True),
        ADMM_SpikingLinear(hidden_dim, classes, h=ADMM_Heaviside(), bias=True)
    ]
    profile_architecture("Spiking Linear", layers_s_lin, inputs_s_lin, labels, device, T_val=T)

    # -----------------------------------------------------------------
    # 3. NON-SPIKING (STATIC) CONVOLUTIONAL
    # -----------------------------------------------------------------
    # Static models lose the Time (T) dimension
    inputs_conv = torch.randn((batch, in_c, H, W), device=device)
    layers_conv = [
        ADMM_Conv2d(in_c, 16, k=5, p=2, s=2, h=ADMM_ReLU(), bias=True), # Assuming ReLU is used for static
        ADMM_Linear(16, classes, pool_op=ADMM_GAP(), h=ADMM_ReLU(), bias=True)
    ]
    # For non-spiking, T is effectively 1
    profile_architecture("Static Conv", layers_conv, inputs_conv, labels, device, T_val=1)

    # -----------------------------------------------------------------
    # 4. NON-SPIKING (STATIC) LINEAR
    # -----------------------------------------------------------------
    inputs_lin = torch.randn((batch, flat_dim), device=device)
    layers_lin = [
        ADMM_Linear(flat_dim, hidden_dim, h=ADMM_ReLU(), bias=True),
        ADMM_Linear(hidden_dim, classes, h=None, bias=True)
    ]
    profile_architecture("Static Linear", layers_lin, inputs_lin, labels, device, T_val=1)

if __name__ == "__main__":
    run_all_profiles()