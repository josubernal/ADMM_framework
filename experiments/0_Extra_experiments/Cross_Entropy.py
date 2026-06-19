import json
import os
import time

import torch

# Import the loss functions directly from your framework
from src.admm.loss_functions import ADMM_CrossEntropy, ADMM_CrossEntropy_Taylor


class MockConfig:
    """A lightweight mock of ADMM_LayerConfig to provide the rho parameter."""

    def __init__(self, rho=1.0):
        self.rho = rho


def main():
    # Use GPU to properly measure the matrix inversion bottleneck
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on: {device.type.upper()}\n")

    # Benchmarking Parameters
    batch_size = 128
    iterations = 50  # How many times to run the update to get a stable average
    warmup_iters = 10  # Warmup to compile CUDA kernels before starting the timer

    # We scale the number of classes from NMNIST size (10) up to ImageNet size+ (2000)
    class_sizes = [10, 50, 100, 250, 500, 1000, 1500, 2000]

    # Initialize the loss operators and our mock config
    loss_exact = ADMM_CrossEntropy()
    loss_taylor = ADMM_CrossEntropy_Taylor()
    config = MockConfig(rho=1.0)

    # Storage for saving
    times_exact = []
    times_taylor = []

    for C in class_sizes:
        print(f"Benchmarking Class Size: C={C} ...")

        # 1. Generate randomized dummy tensors simulating the network state
        temporal_forward = torch.randn(batch_size, C, device=device)
        lambda_lagrange = torch.randn(batch_size, C, device=device)

        # Create valid one-hot labels
        labels = torch.zeros(batch_size, C, device=device)
        target_idx = torch.randint(0, C, (batch_size,), device=device)
        labels.scatter_(1, target_idx.unsqueeze(1), 1.0)

        # ==========================================
        # WARMUP RUNS (Ignore timing)
        # ==========================================
        for _ in range(warmup_iters):
            _ = loss_exact._loss_update(
                temporal_forward, labels, lambda_lagrange, config
            )
            _ = loss_taylor._loss_update(
                temporal_forward, labels, lambda_lagrange, config
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ==========================================
        # EXACT CROSS-ENTROPY (Newton-Raphson)
        # ==========================================
        start_exact = time.perf_counter()
        for _ in range(iterations):
            _ = loss_exact._loss_update(
                temporal_forward, labels, lambda_lagrange, config
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_exact = time.perf_counter()

        avg_exact_ms = ((end_exact - start_exact) / iterations) * 1000
        times_exact.append(avg_exact_ms)

        # ==========================================
        # TAYLOR APPROXIMATION CROSS-ENTROPY
        # ==========================================
        start_taylor = time.perf_counter()
        for _ in range(iterations):
            _ = loss_taylor._loss_update(
                temporal_forward, labels, lambda_lagrange, config
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_taylor = time.perf_counter()

        avg_taylor_ms = ((end_taylor - start_taylor) / iterations) * 1000
        times_taylor.append(avg_taylor_ms)

        print(f"  -> Exact (Newton): {avg_exact_ms:.4f} ms")
        print(f"  -> Taylor Approx:  {avg_taylor_ms:.4f} ms\n")

    # ==========================================
    # SAVE RESULTS TO JSON
    # ==========================================
    results = {
        "metadata": {
            "batch_size": batch_size,
            "iterations": iterations,
            "warmup_iters": warmup_iters,
            "device": device.type,
        },
        "class_sizes": class_sizes,
        "times_exact_ms": times_exact,
        "times_taylor_ms": times_taylor,
    }

    # Save to your existing results directory
    output_path = "experiments/0_Extra_experiments/results/Cross_Entropy.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Benchmark complete! Results saved to '{output_path}'")


if __name__ == "__main__":
    main()
