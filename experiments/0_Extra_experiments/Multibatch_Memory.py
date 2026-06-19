import gc
import json
import os
import time

import torch
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_SpikingFeedForward,
)


def format_mb(memory_bytes):
    return memory_bytes / (1024**2)


def benchmark_multibatch_vs_full():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure accurate GPU memory.")

    device = torch.device("cuda")

    # --- cuBLAS WARMUP ---
    _ = torch.randn(1, 1, device=device) @ torch.randn(1, 1, device=device)
    torch.cuda.empty_cache()

    # Benchmark Parameters
    total_sizes = [16, 32, 64, 128, 256, 512, 1024]
    chunk_size = 16  # The fixed batch size for the Multi-batch runs
    n_timesteps = 100

    # Layer Dimensions
    in_features = 1024
    out_features = 1024

    spiking_config = ADMM_LayerConfig(
        rho=0.1,
        beta=0.1,
        deltas=0.95,
        thetas=1.0,
        use_bias=False,
        use_reset=True,
        use_lagrange=True,
    )
    manager_config = ADMM_Config(init="zeros", train_method="vectorized")

    results = {
        "Full_Batch": {"memory_mb": [], "time_s": []},
        "Multi_Batch": {"memory_mb": [], "time_s": []},
    }

    for N in total_sizes:
        print(f"\n--- Testing Total Dataset Size: {N} ---")

        # Generate the monolithic dataset strictly on CPU
        dummy_x = torch.randn(n_timesteps, N, in_features)
        dummy_y = torch.ones(N, out_features)

        # 3. Execution Loop (Build loaders sequentially so they don't pollute each other)
        for mode in ["Full_Batch", "Multi_Batch"]:
            if mode == "Full_Batch":
                # Push to GPU only for the Full Batch run
                loader = [(dummy_x.to(device), dummy_y.to(device))]
            else:
                # Keep on CPU for Multi Batch streaming
                loader = []
                for i in range(0, N, chunk_size):
                    x_chunk = dummy_x[:, i : i + chunk_size, :].contiguous()
                    y_chunk = dummy_y[i : i + chunk_size, :].contiguous()
                    loader.append((x_chunk, y_chunk))

            # Setup layer and manager
            layer = ADMM_SpikingFeedForward(
                in_f=in_features,
                out_f=out_features,
                h=ADMM_Heaviside(1.0),
                config=spiking_config,
            )
            model = ADMM(
                nn.ModuleList([layer]),
                loss_f=ADMM_SSE(),
                T=n_timesteps,
                config=manager_config,
            ).to(device)

            # Scrub the VRAM perfectly clean before starting the measurement
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # FIT TIMING AND MEMORY
            start_time = time.time()
            model.fit(loader, warming=False)
            torch.cuda.synchronize()
            exec_time = time.time() - start_time

            peak_mem = format_mb(torch.cuda.max_memory_allocated())

            results[mode]["memory_mb"].append(peak_mem)
            results[mode]["time_s"].append(exec_time)

            print(
                f"{mode:12} -> Peak Mem: {peak_mem:7.2f} MB | Time: {exec_time:5.2f} s | Chunks: {len(loader)}"
            )

            # Strict cleanup
            del model, layer, loader
            gc.collect()
            torch.cuda.empty_cache()

        # Scrub CPU memory for the next loop size
        del dummy_x, dummy_y
        gc.collect()

    # Save to JSON
    output_dir = "experiments/0_Extra_experiments/results/Multibatch_Memory"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(
            {"total_sizes": total_sizes, "chunk_size": chunk_size, "metrics": results},
            f,
            indent=4,
        )

    print("\nBenchmark complete. Results saved.")


if __name__ == "__main__":
    benchmark_multibatch_vs_full()
