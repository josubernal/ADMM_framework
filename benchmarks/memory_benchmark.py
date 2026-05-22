"""
ADMM GPU Peak Memory Profiler (Multi-Architecture)
Refactored to match the centralized get_model/get_dataset pipeline.
"""

import configparser
import gc
import json
import os

import torch

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.models import get_model
from src.admm import ADMM_SSE, ADMM_CrossEntropy_Taylor


def format_mb(memory_bytes):
    """Converts bytes to Megabytes for clean printing."""
    return memory_bytes / (1024**2)


def run_all_profiles():
    if not torch.cuda.is_available():
        print("FATAL: CUDA is not available. You must run this on the cluster node!")
        return

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda")

    config = configparser.ConfigParser()
    config.read("benchmarks/config/config.ini")
    seed = config.getint("config", "seed")
    T = config.getint("config", "n_timesteps")

    batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
    architectures = ["linear", "conv", "spiking-linear", "spiking-conv"]

    for model_name in architectures:
        print(f"\n{'=' * 50}")
        print(f"PROFILING ARCHITECTURE: {model_name.upper()}")
        print(f"{'=' * 50}")

        for batch_size in batch_sizes:
            print(f"\n*** BATCH SIZE: {batch_size} ***")

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            is_spiking = model_name.startswith("spiking")
            train_loader = get_dataset(
                batch_size=batch_size,
                n_batches=1,
                spiking=is_spiking,
                n_timesteps=T if is_spiking else None,
                seed=seed,
            )

            loss_fn = ADMM_CrossEntropy_Taylor() if is_spiking else ADMM_SSE()
            train_method = "decoupled-sequential" if is_spiking else "vectorized"

            model = get_model(
                model_name=model_name,
                init="zeros",
                train_method=train_method,
                layer_order="backwards",
                loss=loss_fn,
                z_first=False,
            )
            model.to(device)

            # Init memory
            torch.cuda.synchronize()
            mem_init = format_mb(torch.cuda.memory_allocated())
            print(f"  Init Allocated: {mem_init:.2f} MB")

            # 1 Epoch memory
            torch.cuda.reset_peak_memory_stats()
            model.fit(train_loader, warming=False)
            torch.cuda.synchronize()
            mem_peak_1 = format_mb(torch.cuda.max_memory_allocated())
            print(f"  1-Epoch Peak:   {mem_peak_1:.2f} MB")

            # 10 Epoch memory
            torch.cuda.reset_peak_memory_stats()
            for _ in range(9):  # 9 more iterations to reach 10 total
                model.fit(train_loader, warming=False)
            torch.cuda.synchronize()
            mem_peak_10 = format_mb(torch.cuda.max_memory_allocated())
            print(f"  10-Epoch Peak:  {mem_peak_10:.2f} MB")

            metrics = {
                "architecture": model_name,
                "batch_size": batch_size,
                "init_memory_mb": mem_init,
                "peak_1_epoch_mb": mem_peak_1,
                "peak_10_epoch_mb": mem_peak_10,
                "seed": seed,
            }

            metrics_filename = f"benchmarks/results/memory_benchmark/{model_name}/{batch_size}/results.json"
            os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

            with open(metrics_filename, "w") as f:
                json.dump(metrics, f, indent=4)

            # Cleanup before next iteration
            del model
            del train_loader
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_all_profiles()
