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

    unrolled_nocache_mem = []
    unrolled_nocache_time = []

    dec_mem = []
    dec_time = []

    dec_nocache_mem = []
    dec_nocache_time = []

    # Read the JSON files
    for bs in batch_sizes:
        old_path = f"experiments/4_Memory_benchmark/results/Old/{bs}/results.json"
        new_path = f"experiments/4_Memory_benchmark/results/Unrolled/{bs}/results.json"
        unrolled_nocache_path = (
            f"experiments/4_Memory_benchmark/results/Unrolled_NoCache/{bs}/results.json"
        )
        decoupled_path = (
            f"experiments/4_Memory_benchmark/results/Decoupled/{bs}/results.json"
        )
        dec_nocache_path = f"experiments/4_Memory_benchmark/results/Decoupled_NoCache/{bs}/results.json"

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

        # Parse UNROLLED NO CACHE results
        if os.path.exists(unrolled_nocache_path):
            with open(unrolled_nocache_path, "r") as f:
                data = json.load(f)
                unrolled_nocache_mem.append(data.get("peak_10_epoch_mb", None))
                unrolled_nocache_time.append(data.get("time", None))
        else:
            print(
                f"Warning: Missing data for Unrolled No Cache model at batch size {bs}"
            )
            unrolled_nocache_mem.append(None)
            unrolled_nocache_time.append(None)

        # Parse DECOUPLED results
        if os.path.exists(decoupled_path):
            with open(decoupled_path, "r") as f:
                data = json.load(f)
                dec_mem.append(data.get("peak_10_epoch_mb", None))
                dec_time.append(data.get("time", None))
        else:
            print(f"Warning: Missing data for New decoupled model at batch size {bs}")
            dec_mem.append(None)
            dec_time.append(None)

        # Parse DECOUPLED NO CACHE results
        if os.path.exists(dec_nocache_path):
            with open(dec_nocache_path, "r") as f:
                data = json.load(f)
                dec_nocache_mem.append(data.get("peak_10_epoch_mb", None))
                dec_nocache_time.append(data.get("time", None))
        else:
            print(
                f"Warning: Missing data for Decoupled No Cache model at batch size {bs}"
            )
            dec_nocache_mem.append(None)
            dec_nocache_time.append(None)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Peak Memory ---
    ax1.plot(
        batch_sizes, old_mem, marker="o", linestyle="-", linewidth=2, label="Old Model"
    )
    ax1.plot(
        batch_sizes, new_mem, marker="s", linestyle="-", linewidth=2, label="Unrolled"
    )
    ax1.plot(
        batch_sizes,
        unrolled_nocache_mem,
        marker="^",
        linestyle="--",
        linewidth=2,
        label="Unrolled (No Cache)",
    )
    ax1.plot(
        batch_sizes, dec_mem, marker="s", linestyle="-", linewidth=2, label="Decoupled"
    )
    ax1.plot(
        batch_sizes,
        dec_nocache_mem,
        marker="v",
        linestyle="--",
        linewidth=2,
        label="Decoupled (No Cache)",
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
        batch_sizes, new_time, marker="s", linestyle="-", linewidth=2, label="Unrolled"
    )
    ax2.plot(
        batch_sizes,
        unrolled_nocache_time,
        marker="^",
        linestyle="--",
        linewidth=2,
        label="Unrolled (No Cache)",
    )
    ax2.plot(
        batch_sizes, dec_time, marker="s", linestyle="-", linewidth=2, label="Decoupled"
    )
    ax2.plot(
        batch_sizes,
        dec_nocache_time,
        marker="v",
        linestyle="--",
        linewidth=2,
        label="Decoupled (No Cache)",
    )

    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Execution Time [Seconds]", fontsize=12)
    ax2.set_title("Execution Time vs Batch Size", fontsize=14)
    ax2.set_xticks(batch_sizes)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Finalize and save
    plt.tight_layout()
    plot_filename = "experiments/4_Memory_benchmark/results/plot.png"

    # Optional: ensure directory exists before saving (just in case the script is run from a fresh path)
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot successfully saved as '{plot_filename}'")


if __name__ == "__main__":
    main()
