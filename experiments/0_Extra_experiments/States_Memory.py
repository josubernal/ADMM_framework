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


def benchmark_state_memory():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure accurate GPU memory.")

    device = torch.device("cuda")
    # --- WARMUP ---
    # Force cuBLAS to initialize its 8MB workspace before we start measuring
    _ = torch.randn(1, 1, device=device) @ torch.randn(1, 1, device=device)

    # Benchmark Parameters
    batch_sizes = [4, 8, 16, 32, 64, 128, 256]
    n_timesteps = 100

    # Layer Dimensions (Keep them comparable in parameter count/output size if possible)
    in_features = 1024
    out_features = 1024

    in_channels = 32
    out_channels = 64
    image_size = 16  # e.g., 16x16 feature map

    # Standard layer configurations
    static_config = ADMM_LayerConfig(rho=0.1, beta=0.1, use_bias=False)
    spiking_config = ADMM_LayerConfig(
        rho=0.1, beta=0.1, deltas=0.95, thetas=1.0, use_bias=False, use_reset=True
    )

    manager_config = ADMM_Config(init="zeros", train_method="vectorized")

    results = {
        "FeedForward": [],
        "Conv2d": [],
        "Spiking_FeedForward": [],
        "Spiking_Conv2d": [],
    }

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")

        # 1. Dummy Data Generators
        dummy_flat_y = torch.ones(bs, out_features).to(device)  # Dummy targets
        dummy_img_y = torch.ones(bs, out_channels, image_size, image_size).to(device)

        dummy_flat_x = torch.randn(bs, in_features).to(device)
        dummy_img_x = torch.randn(bs, in_channels, image_size, image_size).to(device)

        # Spiking inputs need the time dimension [T, B, ...]
        dummy_spiking_flat_x = torch.randn(n_timesteps, bs, in_features).to(device)
        dummy_spiking_img_x = torch.randn(
            n_timesteps, bs, in_channels, image_size, image_size
        ).to(device)

        # 2. Define the Layer Test Cases (Using direct lists instead of DataLoader)
        test_cases = {
            "FeedForward": {
                "layer": ADMM_FeedForward(
                    in_f=in_features,
                    out_f=out_features,
                    h=ADMM_ReLU(),
                    config=static_config,
                ),
                "data_loader": [(dummy_flat_x, dummy_flat_y)],
                "T": None,
            },
            "Conv2d": {
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
            "Spiking_FeedForward": {
                "layer": ADMM_SpikingFeedForward(
                    in_f=in_features,
                    out_f=out_features,
                    h=ADMM_Heaviside(1.0),
                    config=spiking_config,
                ),
                "data_loader": [(dummy_spiking_flat_x, dummy_flat_y)],
                "T": n_timesteps,
            },
            "Spiking_Conv2d": {
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
            model = ADMM(
                nn.ModuleList([setup["layer"]]), T=setup["T"], config=manager_config
            ).to(device)

            # Wipe memory clean before allocating states
            torch.cuda.empty_cache()
            mem_weights_only = torch.cuda.memory_allocated()

            # This triggers the dry-run forward pass and allocates z, a, and lambda
            model.state_handler.initialize_all_batches(loader)

            # Wipe temporary buffers from the dry-run forward pass
            torch.cuda.empty_cache()
            mem_weights_and_states = torch.cuda.memory_allocated()

            # The difference is strictly the state tensors
            state_mem_mb = format_mb(mem_weights_and_states - mem_weights_only)
            results[name].append(state_mem_mb)

            print(f"{name:20} -> {state_mem_mb:.2f} MB")

            # Strict cleanup
            del model, loader
            torch.cuda.empty_cache()

    # Save to JSON
    os.makedirs("experiments/0_Extra_experiments/results/States_Memory", exist_ok=True)
    with open(
        "experiments/0_Extra_experiments/results/States_Memory/results.json", "w"
    ) as f:
        json.dump({"batch_sizes": batch_sizes, "memory_mb": results}, f, indent=4)

    print("\nBenchmark complete. Results saved.")


if __name__ == "__main__":
    benchmark_state_memory()
