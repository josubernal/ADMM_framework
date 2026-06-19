import json
import os

import matplotlib.pyplot as plt


def plot_multibatch_benchmark():
    # The exact path where your Multibatch_Memory.py script saves the JSON
    results_path = (
        "experiments/0_Extra_experiments/results/Multibatch_Memory/results.json"
    )
    plot_filename = "experiments/0_Extra_experiments/results/Multibatch_Memory/multibatch_vs_full_plot.png"

    # Check if data exists
    if not os.path.exists(results_path):
        print(f"Error: Could not find '{results_path}'")
        print("Please ensure your benchmark script ran successfully.")
        return

    # Load data
    with open(results_path, "r") as f:
        data = json.load(f)

    sizes = data.get("total_sizes", [])
    metrics = data.get("metrics", {})
    chunk_size = data.get("chunk_size", 16)

    full_mem = metrics.get("Full_Batch", {}).get("memory_mb", [])
    full_time = metrics.get("Full_Batch", {}).get("time_s", [])

    multi_mem = metrics.get("Multi_Batch", {}).get("memory_mb", [])
    multi_time = metrics.get("Multi_Batch", {}).get("time_s", [])

    if not all([sizes, full_mem, full_time, multi_mem, multi_time]):
        print(
            "Error: The JSON file is missing required data keys or contains empty lists."
        )
        return

    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Peak Memory ---
    ax1.plot(
        sizes,
        full_mem,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Full Batch",
        color="#1f77b4",
    )
    ax1.plot(
        sizes,
        multi_mem,
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Multi-Batch (Chunk={chunk_size})",
        color="#ff7f0e",
    )

    ax1.set_xlabel("Total Samples Processed (N)", fontsize=12)
    ax1.set_ylabel("Peak VRAM Allocation [MB]", fontsize=12)
    ax1.set_title("Peak Memory vs. Dataset Size", fontsize=14)
    ax1.set_xticks(sizes)
    # Optional: Rotate x-ticks if they overlap
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Plot 2: Execution Time ---
    ax2.plot(
        sizes,
        full_time,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Full Batch",
        color="#1f77b4",
    )
    ax2.plot(
        sizes,
        multi_time,
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Multi-Batch (Chunk={chunk_size})",
        color="#ff7f0e",
    )

    ax2.set_xlabel("Total Samples Processed (N)", fontsize=12)
    ax2.set_ylabel("Execution Time [Seconds]", fontsize=12)
    ax2.set_title("Execution Time vs. Dataset Size", fontsize=14)
    ax2.set_xticks(sizes)
    # Optional: Rotate x-ticks if they overlap
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Finalize and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot successfully saved as '{plot_filename}'")

    # If running locally (not on a headless node), you can uncomment below to view it instantly
    # plt.show()


if __name__ == "__main__":
    plot_multibatch_benchmark()
