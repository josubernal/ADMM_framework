import json
import os

import torch
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_Config,
    ADMM_Conv2d,
    ADMM_FeedForward,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_ReLU,
    ADMM_SpikingConv2d,
    ADMM_SpikingFeedForward,
)


def format_mb(memory_bytes):
    return memory_bytes / (1024**2)


def benchmark_covariance_peak_memory():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure accurate GPU memory.")

    device = torch.device("cuda")
    # --- WARMUP ---
    # Force cuBLAS to initialize its workspace before we start measuring
    _ = torch.randn(1, 1, device=device) @ torch.randn(1, 1, device=device)

    # Benchmark Parameters (Trimmed max batch size to avoid immediate OOM on Conv)
    batch_sizes = [4, 8, 16, 32, 64]
    n_timesteps = 50  # Reduced to prevent instant OOM for Spiking Conv2d

    # Layer Dimensions
    in_features = 1024
    out_features = 1024
    in_channels = 32
    out_channels = 64
    image_size = 16

    static_config = ADMM_LayerConfig(rho=0.1, beta=0.1, use_bias=False)
    spiking_config = ADMM_LayerConfig(
        rho=0.1, beta=0.1, deltas=0.95, thetas=1.0, use_bias=False, use_reset=True
    )

    manager_config = ADMM_Config(init="zeros", train_method="vectorized")

    results = {
        "FeedForward_Peak": [],
        "Conv2d_Peak": [],
        "Spiking_FeedForward_Peak": [],
        "Spiking_Conv2d_Peak": [],
    }

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")

        # 1. Dummy Data Generators
        dummy_flat_y = torch.ones(bs, out_features).to(device)
        dummy_img_y = torch.ones(bs, out_channels, image_size, image_size).to(device)

        dummy_flat_x = torch.randn(bs, in_features).to(device)
        dummy_img_x = torch.randn(bs, in_channels, image_size, image_size).to(device)

        dummy_spiking_flat_x = torch.randn(n_timesteps, bs, in_features).to(device)
        dummy_spiking_img_x = torch.randn(
            n_timesteps, bs, in_channels, image_size, image_size
        ).to(device)

        # 2. Define the Layer Test Cases
        test_cases = {
            "FeedForward_Peak": {
                "layer": ADMM_FeedForward(
                    in_f=in_features,
                    out_f=out_features,
                    h=ADMM_ReLU(),
                    config=static_config,
                ),
                "data_loader": [(dummy_flat_x, dummy_flat_y)],
                "T": None,
            },
            "Conv2d_Peak": {
                "layer": ADMM_Conv2d(
                    in_c=in_channels,
                    out_c=out_channels,
                    k=3,
                    p=1,
                    s=1,
                    h=ADMM_ReLU(),
                    config=static_config,
                ),
                "data_loader": [(dummy_img_x, dummy_img_y)],
                "T": None,
            },
            "Spiking_FeedForward_Peak": {
                "layer": ADMM_SpikingFeedForward(
                    in_f=in_features,
                    out_f=out_features,
                    h=ADMM_Heaviside(1.0),
                    config=spiking_config,
                ),
                "data_loader": [(dummy_spiking_flat_x, dummy_flat_y)],
                "T": n_timesteps,
            },
            "Spiking_Conv2d_Peak": {
                "layer": ADMM_SpikingConv2d(
                    in_c=in_channels,
                    out_c=out_channels,
                    k=3,
                    p=1,
                    s=1,
                    h=ADMM_Heaviside(1.0),
                    config=spiking_config,
                ),
                "data_loader": [(dummy_spiking_img_x, dummy_img_y)],
                "T": n_timesteps,
            },
        }

        # 3. Execution Loop
        for name, setup in test_cases.items():
            loader = setup["data_loader"]
            layer = setup["layer"]

            # Init framework manager and allocate states
            model = ADMM(
                nn.ModuleList([layer]), T=setup["T"], config=manager_config
            ).to(device)
            model.state_handler.initialize_all_batches(loader)

            # Retrieve exactly what Phase 1 of `manager.py` needs
            inputs, _, batch_state = model.state_handler.load_batch(0)
            state = batch_state.layer_states[0]
            a_prev = model._get_a_prev(0, inputs, batch_state)

            # --- ISOLATED BENCHMARK START ---
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # Execute the exact framework bottleneck
            try:
                _ = layer.compute_batch_covariances(state=state, a_prev=a_prev)
                peak_mem_mb = format_mb(torch.cuda.max_memory_allocated(device))
            except torch.cuda.OutOfMemoryError:
                peak_mem_mb = float("inf")  # Mark as OOM
                print(f"{name:25} -> OUT OF MEMORY!")

            # --- ISOLATED BENCHMARK END ---

            if peak_mem_mb != float("inf"):
                print(f"{name:25} -> {peak_mem_mb:.2f} MB")

            results[name].append(peak_mem_mb)

            # Strict cleanup
            del model, loader, inputs, batch_state, state, a_prev
            torch.cuda.empty_cache()

    # Save to JSON
    os.makedirs("experiments/0_Extra_experiments/results/GramMemory", exist_ok=True)
    with open(
        "experiments/0_Extra_experiments/results/GramMemory/results.json", "w"
    ) as f:
        json.dump({"batch_sizes": batch_sizes, "memory_mb": results}, f, indent=4)

    print("\nBenchmark complete. Results saved.")


if __name__ == "__main__":
    benchmark_covariance_peak_memory()
