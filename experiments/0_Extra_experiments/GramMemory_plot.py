import json
import os

import matplotlib.pyplot as plt


def plot_covariance_peak_memory():
    # File paths matching the output of the new benchmark script
    results_path = "experiments/0_Extra_experiments/results/GramMemory/results.json"
    plot_filename = (
        "experiments/0_Extra_experiments/results/GramMemory/gram_peak_plot.png"
    )

    # Check if data exists
    if not os.path.exists(results_path):
        print(f"Error: Could not find the results file at '{results_path}'.")
        print("Please run the benchmark script first.")
        return

    # Load data
    with open(results_path, "r") as f:
        data = json.load(f)

    batch_sizes = data.get("batch_sizes", [])
    memory_mb = data.get("memory_mb", {})

    # Use the new dictionary keys from the Covariance Peak experiment
    ff_mem = memory_mb.get("FeedForward_Peak", [])
    conv_mem = memory_mb.get("Conv2d_Peak", [])
    spff_mem = memory_mb.get("Spiking_FeedForward_Peak", [])
    spconv_mem = memory_mb.get("Spiking_Conv2d_Peak", [])

    if not all([batch_sizes, ff_mem, conv_mem, spff_mem, spconv_mem]):
        print(
            "Error: The JSON file is missing required data keys or contains empty lists."
        )
        return

    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Static Networks (FeedForward vs Conv2d) ---
    ax1.plot(
        batch_sizes,
        ff_mem,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="FeedForward",
        color="#1f77b4",
    )
    ax1.plot(
        batch_sizes,
        conv_mem,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Conv2d (im2col)",
        color="#ff7f0e",
    )

    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Peak Transient Memory [MB]", fontsize=12)
    ax1.set_title("Static Networks: P^T P Peak Memory", fontsize=14)
    ax1.set_xticks(batch_sizes)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Subplot 2: Spiking Networks (SpFF vs SpConv) ---
    ax2.plot(
        batch_sizes,
        spff_mem,
        marker="^",
        linestyle="-",
        linewidth=2,
        label="Spiking FeedForward",
        color="#2ca02c",
    )
    ax2.plot(
        batch_sizes,
        spconv_mem,
        marker="v",
        linestyle="-",
        linewidth=2,
        label="Spiking Conv2d (im2col + Time)",
        color="#d62728",
    )

    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Peak Transient Memory [MB]", fontsize=12)
    ax2.set_title("Spiking Networks: P^T P Peak Memory", fontsize=14)
    ax2.set_xticks(batch_sizes)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Use a logarithmic scale for the Y-axis if the spike is massive
    # (Optional: uncomment these two lines if the Conv layer completely squashes the FF layer at the bottom of the graph)
    # ax1.set_yscale("log")
    # ax2.set_yscale("log")

    # Finalize and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot successfully saved as '{plot_filename}'")

    # plt.show()


if __name__ == "__main__":
    plot_covariance_peak_memory()
