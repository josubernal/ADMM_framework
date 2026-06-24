import json
import os

import matplotlib.pyplot as plt


def main():
    batch_sizes = [4, 8, 16, 32, 64, 128, 256]

    # Data storage
    old_mem = []
    old_time = []

    new_mem = []
    new_time = []

    # Read the JSON files
    for bs in batch_sizes:
        old_path = f"experiments/4_Performance_benchmark/results/Old/{bs}/results.json"
        new_path = f"experiments/4_Performance_benchmark/results/New/{bs}/results.json"

        # Parse OLD model results
        if os.path.exists(old_path):
            with open(old_path, "r") as f:
                data = json.load(f)
                old_mem.append(data.get("peak_10_epoch_mb", None))
                old_time.append(data.get("time", None))
        else:
            print(f"Warning: Missing data for Old model at batch size {bs}")
            old_mem.append(None)
            old_time.append(None)

        # Parse NEW model (Unrolled) results
        if os.path.exists(new_path):
            with open(new_path, "r") as f:
                data = json.load(f)
                new_mem.append(data.get("peak_10_epoch_mb", None))
                new_time.append(data.get("time", None))
        else:
            print(f"Warning: Missing data for New sequential model at batch size {bs}")
            new_mem.append(None)
            new_time.append(None)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Peak Memory ---
    ax1.plot(
        batch_sizes, old_mem, marker="o", linestyle="-", linewidth=2, label="Old Model"
    )
    ax1.plot(
        batch_sizes, new_mem, marker="s", linestyle="-", linewidth=2, label="New Model"
    )

    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Peak Memory (10 Epochs) [MB]", fontsize=12)
    ax1.set_title("Peak Memory vs Batch Size", fontsize=14)
    ax1.set_xticks(batch_sizes)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Plot 2: Execution Time ---
    ax2.plot(
        batch_sizes, old_time, marker="o", linestyle="-", linewidth=2, label="Old Model"
    )
    ax2.plot(
        batch_sizes, new_time, marker="s", linestyle="-", linewidth=2, label="New Model"
    )

    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Execution Time [Seconds]", fontsize=12)
    ax2.set_title("Execution Time vs Batch Size", fontsize=14)
    ax2.set_xticks(batch_sizes)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Finalize and save
    plt.tight_layout()
    plot_filename = "experiments/4_Performance_benchmark/results/plot.png"

    # Optional: ensure directory exists before saving (just in case the script is run from a fresh path)
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot successfully saved as '{plot_filename}'")


if __name__ == "__main__":
    main()
