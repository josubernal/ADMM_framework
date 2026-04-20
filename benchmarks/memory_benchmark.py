"""
ADMM GPU Peak Memory Profiler (Multi-Architecture)
Modified for Multi-Batch Profiling - Tracking Init, 1-Epoch Peak, and 10-Epoch Peak.
"""

import os
import csv
import torch
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import configparser

from admm.manager import ADMM
from admm.layers import ADMM_SpikingConv2d, ADMM_SpikingLinear, ADMM_Conv2d, ADMM_Linear
from admm.activations import ADMM_Heaviside, ADMM_ReLU
from admm.pooling import ADMM_Flatten

def format_mb(memory_bytes):
    """Converts bytes to Megabytes for clean printing."""
    return memory_bytes / (1024 ** 2)

def profile_architecture(arch_name, layers, inputs, labels, device, T_val):
    """Runs the profiling suite and returns Init Memory, 1-Epoch Peak, and 10-Epoch Peak."""
    print("\n" + "="*85)
    print(f"  PROFILING: {arch_name.upper()} | BATCH: {inputs.shape[1] if T_val is not None else inputs.shape[0]}")
    print("="*85)
    
    # 1. Reset everything
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. Build Model
    model = ADMM(layers, T=T_val, device=device, init="zeros", 
                 train_method="decoupled-sequential", bias=True, 
                 deltas=0.8, thetas=1.0, rho=1.0, beta=1.0)
    
    # 3. Initialization Memory (Parameters, a, z)
    model._init_states(inputs)
    torch.cuda.synchronize()
    mem_init = format_mb(torch.cuda.memory_allocated())
    print(f"🔹 Init (a,z) Allocated: {mem_init:.2f} MB")

    # 4. Single Epoch Peak Test
    torch.cuda.reset_peak_memory_stats() 
    model.fit(inputs, labels, warming=False)
    torch.cuda.synchronize() 
    mem_peak_1 = format_mb(torch.cuda.max_memory_allocated())
    print(f"🔹 1-Epoch Peak VRAM:   {mem_peak_1:.2f} MB")
    
    # 5. 10 Epochs Stress Test Peak
    torch.cuda.reset_peak_memory_stats() 
    for _ in range(10):
        model.fit(inputs, labels, warming=False)
    torch.cuda.synchronize()
    mem_peak_10 = format_mb(torch.cuda.max_memory_allocated())
    print(f"🔹 10-Epoch Peak VRAM:  {mem_peak_10:.2f} MB")
    
    # Cleanup to prevent OOM on next iteration
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return mem_init, mem_peak_1, mem_peak_10

def save_and_plot_results(results_dict, batch_sizes, output_dir):
    """Saves results to CSV and generates matplotlib plots for the 3 memory metrics."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_memory_metrics.csv")

    # 1. Write to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Architecture", "Batch Size", "Init Memory (MB)", "Peak 1 Epoch (MB)", "Peak 10 Epochs (MB)"])
        for arch, metrics in results_dict.items():
            for i, b in enumerate(batch_sizes):
                writer.writerow([arch, b, metrics['init'][i], metrics['peak_1'][i], metrics['peak_10'][i]])
    print(f"\n💾 Data saved to {csv_path}")

    # # 2. Plotting Utilities
    # plt.style.use('ggplot')
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    # markers = ['o', 's', '^', 'D']

    # def generate_plot(metric_key, title, filename):
    #     plt.figure(figsize=(10, 6))
    #     for i, (arch, metrics) in enumerate(results_dict.items()):
    #         plt.plot(batch_sizes, metrics[metric_key], marker=markers[i], color=colors[i], label=arch, linewidth=2)
    #     plt.title(title, fontsize=14, fontweight='bold')
    #     plt.xlabel("Batch Size", fontsize=12)
    #     plt.ylabel("VRAM (MB)", fontsize=12)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend()
    #     plt.tight_layout()
    #     filepath = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    #     plt.savefig(filepath, dpi=300)
    #     plt.close()
    #     print(f"📊 Plot saved to {filepath}")

    # generate_plot('init', "Allocated Memory After Init (a, z) vs Batch Size", "init_memory_vs_batch.png")
    # generate_plot('peak_1', "Peak VRAM (1 Epoch) vs Batch Size", "peak_memory_1_epoch_vs_batch.png")
    # generate_plot('peak_10', "Peak VRAM (10 Epochs) vs Batch Size", "peak_memory_10_epochs_vs_batch.png")

def run_all_profiles():
    if not torch.cuda.is_available():
        print("❌ FATAL: CUDA is not available. You must run this on the cluster node!")
        return

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda')
    
    config = configparser.ConfigParser()
    config.read('benchmarks/config/config.ini')
    # ==========================================
    # 1. GLOBAL SETUP & SEEDING
    # ==========================================
    seed = config.getint('config','seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
   
    hidden_dim = config.getint('config', 'hidden_size')
    hidden_channels = config.getint('config', 'hidden_channels')
    k = config.getint('config', 'k')
    p = config.getint('config', 'p')
    s = config.getint('config', 's')
    T= config.getint('config', 'n_timesteps')   

    in_c, H, W = 2, 32, 32 
    flat_dim = in_c * H * W  
    classes = 10
    spatial = ((32 + 2 * p - k) // s) + 1
    
    # Test Matrix
    batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    architectures = ["Spiking Conv", "Spiking Linear", "Static Conv", "Static Linear"]
    
    # Initialize dictionary to hold all results
    results = {arch: {'init': [], 'peak_1': [], 'peak_10': []} for arch in architectures}

    for batch in batch_sizes:
        print(f"\n\n{'*'*85}\n*** STARTING BATCH SIZE: {batch} ***\n{'*'*85}")
        labels = torch.randint(0, classes, (batch,), device=device)

        # -----------------------------------------------------------------
        # 1. SPIKING CONVOLUTIONAL
        # -----------------------------------------------------------------
        inputs = torch.randn((T, batch, in_c, H, W), device=device)
        layers = [
            ADMM_SpikingConv2d(in_c, hidden_channels, k=k, p=p, s=s, h=ADMM_Heaviside(), bias=True, use_fft=False, padding_mode="zeros"),
            ADMM_SpikingLinear( hidden_channels * spatial * spatial, classes, pool_op=ADMM_Flatten(), h=ADMM_Heaviside(), bias=True)
        ]
        m_i, m_1, m_10 = profile_architecture("Spiking Conv", layers, inputs, labels, device, T_val=T)
        results["Spiking Conv"]['init'].append(m_i)
        results["Spiking Conv"]['peak_1'].append(m_1)
        results["Spiking Conv"]['peak_10'].append(m_10)
        del inputs, layers; gc.collect(); torch.cuda.empty_cache()

        # -----------------------------------------------------------------
        # 2. SPIKING LINEAR
        # -----------------------------------------------------------------
        inputs = torch.randn((T, batch, flat_dim), device=device)
        layers = [
            ADMM_SpikingLinear(flat_dim, hidden_dim, h=ADMM_Heaviside(), bias=True),
            ADMM_SpikingLinear(hidden_dim, classes, h=ADMM_Heaviside(), bias=True)
        ]
        m_i, m_1, m_10 = profile_architecture("Spiking Linear", layers, inputs, labels, device, T_val=T)
        results["Spiking Linear"]['init'].append(m_i)
        results["Spiking Linear"]['peak_1'].append(m_1)
        results["Spiking Linear"]['peak_10'].append(m_10)
        del inputs, layers; gc.collect(); torch.cuda.empty_cache()

        # -----------------------------------------------------------------
        # 3. NON-SPIKING (STATIC) CONVOLUTIONAL
        # -----------------------------------------------------------------
        inputs = torch.randn((batch, in_c, H, W), device=device)
        layers = [
            ADMM_Conv2d(in_c, hidden_channels, k=k, p=p, s=s, h=ADMM_ReLU(), bias=True, use_fft=False, padding_mode="zeros"), 
            ADMM_Linear( hidden_channels * spatial * spatial, classes, pool_op=ADMM_Flatten(), h=ADMM_ReLU(), bias=True)
        ]
        m_i, m_1, m_10 = profile_architecture("Static Conv", layers, inputs, labels, device, T_val=None)
        results["Static Conv"]['init'].append(m_i)
        results["Static Conv"]['peak_1'].append(m_1)
        results["Static Conv"]['peak_10'].append(m_10)
        del inputs, layers; gc.collect(); torch.cuda.empty_cache()

        # -----------------------------------------------------------------
        # 4. NON-SPIKING (STATIC) LINEAR
        # -----------------------------------------------------------------
        inputs = torch.randn((batch, flat_dim), device=device)
        layers = [
            ADMM_Linear(flat_dim, hidden_dim, h=ADMM_ReLU(), bias=True),
            ADMM_Linear(hidden_dim, classes, h=None, bias=True)
        ]
        m_i, m_1, m_10 = profile_architecture("Static Linear", layers, inputs, labels, device, T_val=None)
        results["Static Linear"]['init'].append(m_i)
        results["Static Linear"]['peak_1'].append(m_1)
        results["Static Linear"]['peak_10'].append(m_10)
        del inputs, layers; gc.collect(); torch.cuda.empty_cache()

    # Save and Plot
    save_and_plot_results(results, batch_sizes, output_dir="benchmarks/results/memory_benchmark")

if __name__ == "__main__":
    run_all_profiles()
