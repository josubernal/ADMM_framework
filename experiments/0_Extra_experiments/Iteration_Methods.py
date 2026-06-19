import json
import os
import random
import time

import torch
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_SpikingFeedForward,
)

from .utils.dataset import get_dataset


def format_mb(memory_bytes):
    return memory_bytes / (1024**2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 1000
    WARMING_ITERS = 300
    N_TIMESTEPS = 150
    BATCH_SIZE = 200
    HIDDEN_DIMS = 512
    RHO = 1
    BETA = 0.1
    DELTAS = 0.95
    THETAS = 1

    methods_to_benchmark = ["unrolled", "decoupled", "vectorized"]

    for method in methods_to_benchmark:
        print(f"\n{'=' * 50}")
        print(f"STARTING BENCHMARK: {method.upper()}")
        print(f"{'=' * 50}")

        # Reset seed for exact deterministic comparison across methods
        seed = 8281003564
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_loader = get_dataset(
            model_name="spiking-feedforward",
            batch_size=BATCH_SIZE,
            device=device,
            n_timesteps=N_TIMESTEPS,
            seed=seed,
        )

        hidden_layer_config = ADMM_LayerConfig(
            rho=RHO, beta=BETA, deltas=DELTAS, thetas=THETAS, use_bias=False
        )
        out_layer_config = ADMM_LayerConfig(
            rho=RHO,
            beta=BETA,
            deltas=DELTAS,
            thetas=THETAS,
            use_bias=False,
            use_lagrange=True,
        )

        layers = nn.ModuleList(
            [
                ADMM_SpikingFeedForward(
                    in_f=34 * 34 * 2,
                    out_f=HIDDEN_DIMS,
                    h=ADMM_Heaviside(THETAS),
                    config=hidden_layer_config,
                ),
                ADMM_SpikingFeedForward(
                    in_f=HIDDEN_DIMS,
                    out_f=10,
                    h=None,
                    config=out_layer_config,
                    use_reset=False,
                ),
            ]
        )

        config = ADMM_Config(
            init="s-uniform",
            train_method=method,
            time_order="sequential",
            layer_order="sequential",
            update_z_first=False,
            block_method="multi-block",
        )

        model = ADMM(layers, loss_f=ADMM_SSE(), T=N_TIMESTEPS, config=config).to(device)

        # ---------------------------------------------------------
        # MEMORY ISOLATION: Pre-allocate states to find the baseline
        # ---------------------------------------------------------
        model.state_handler.initialize_all_batches(dataloader=train_loader)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_baseline = torch.cuda.memory_allocated()

        m_tracker = ADMM_Metrics(model)

        peak_comp_memory = 0.0

        # --- START TIMING ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        for epoch in range(EPOCHS + 1):
            # Reset peak stats before each fit to strictly track overhead
            torch.cuda.reset_peak_memory_stats()

            model.fit(train_loader, warming=epoch < WARMING_ITERS)

            # Calculate the pure computational overhead (Peak VRAM - State Variables)
            mem_peak_current = torch.cuda.max_memory_allocated()
            comp_mem_mb = format_mb(mem_peak_current - mem_baseline)

            # Track the absolute highest computational memory seen across all epochs
            if comp_mem_mb > peak_comp_memory:
                peak_comp_memory = comp_mem_mb

            if epoch % 1 == 0:
                with torch.no_grad():
                    m_tracker.save_metrics()
                    print(
                        f"Epoch [{epoch:3d}/{EPOCHS}] | Comp Mem: {comp_mem_mb:.2f} MB | {m_tracker}"
                    )

        # --- END TIMING ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start_time

        print(f"\n[{method.upper()}] completed in {total_time:.2f} seconds.")

        #########################################
        # SAVING RESULTS
        metrics = m_tracker.get_dic()

        # Inject our isolated memory footprint and total execution time
        metrics["isolated_peak_computational_memory_mb"] = peak_comp_memory
        metrics["total_execution_time_s"] = total_time

        metrics_filename = f"experiments/0_Extra_experiments/results/Iteration_Methods/{method}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory strictly before generating the next model
        del model, layers, train_loader, m_tracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
