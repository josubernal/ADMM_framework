import json
import os

import matplotlib.pyplot as plt
import numpy as np

# IEEE publication standard settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (7.16, 3.0),
        "figure.dpi": 300,
        "hatch.linewidth": 0.5,
    }
)


def load_data(batch_sizes, base_path, model_type, n_repeats=5):
    """
    Load all repeated measurements for each batch size.

    Expected structure:
        base_path/model_type/{batch_size}_{repeat}/results.json
    """
    data = {}

    for bs in batch_sizes:
        data[bs] = []

        for repeat in range(n_repeats):
            filepath = os.path.join(
                base_path,
                model_type,
                f"{bs}_{repeat}",
                "results.json",
            )

            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    data[bs].append(json.load(f))
            else:
                print(f"Warning: Missing data: {filepath}")

    return data


def mean_std(values):
    """Return mean and sample standard deviation."""
    values = np.asarray(values, dtype=float)

    if len(values) == 1:
        return values[0], 0.0

    return np.mean(values), np.std(values, ddof=1)


def main():
    base_path = "paper/results/performance"
    batch_sizes = [4, 8, 16, 32, 64, 128, 256]
    n_repeats = 5

    # ============================================================
    # LOAD DATA
    # ============================================================

    old_data = load_data(
        batch_sizes,
        base_path,
        "perin_et_al",
        n_repeats,
    )

    new_data = load_data(
        batch_sizes,
        base_path,
        "framework",
        n_repeats,
    )

    # Only use batch sizes where all 5 trials exist
    valid_batches = [
        bs
        for bs in batch_sizes
        if len(old_data[bs]) == n_repeats and len(new_data[bs]) == n_repeats
    ]

    if not valid_batches:
        print("No valid overlapping data found to plot.")
        return

    print(f"Using batch sizes: {valid_batches}")
    print(f"Using {n_repeats} trials per batch size.")

    # ============================================================
    # CALCULATE MEANS AND STANDARD DEVIATIONS
    # ============================================================

    old_time_mean = []
    old_time_std = []

    new_time_mean = []
    new_time_std = []

    old_mem_1_mean = []
    old_mem_1_std = []

    new_mem_1_mean = []
    new_mem_1_std = []

    old_mem_10_mean = []
    old_mem_10_std = []

    new_mem_10_mean = []
    new_mem_10_std = []

    for bs in valid_batches:
        old_trials = old_data[bs]
        new_trials = new_data[bs]

        # --------------------------------------------------------
        # Execution time
        # --------------------------------------------------------

        old_times = [trial["time"] for trial in old_trials]

        new_times = [trial["time"] for trial in new_trials]

        mean, std = mean_std(old_times)
        old_time_mean.append(mean)
        old_time_std.append(std)

        mean, std = mean_std(new_times)
        new_time_mean.append(mean)
        new_time_std.append(std)

        # --------------------------------------------------------
        # Peak memory after first epoch
        # --------------------------------------------------------

        old_mem_1 = [trial["peak_1_epoch_mb"] for trial in old_trials]

        new_mem_1 = [trial["peak_1_epoch_mb"] for trial in new_trials]

        mean, std = mean_std(old_mem_1)
        old_mem_1_mean.append(mean)
        old_mem_1_std.append(std)

        mean, std = mean_std(new_mem_1)
        new_mem_1_mean.append(mean)
        new_mem_1_std.append(std)

        # --------------------------------------------------------
        # Peak memory after 10 epochs
        # --------------------------------------------------------

        old_mem_10 = [trial["peak_10_epoch_mb"] for trial in old_trials]

        new_mem_10 = [trial["peak_10_epoch_mb"] for trial in new_trials]

        mean, std = mean_std(old_mem_10)
        old_mem_10_mean.append(mean)
        old_mem_10_std.append(std)

        mean, std = mean_std(new_mem_10)
        new_mem_10_mean.append(mean)
        new_mem_10_std.append(std)

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print("\nResults:")
    print("-" * 70)

    for i, bs in enumerate(valid_batches):
        print(
            f"Batch {bs:3d} | "
            f"Time: "
            f"Old={old_time_mean[i]:.3f} ± {old_time_std[i]:.3f} s | "
            f"New={new_time_mean[i]:.3f} ± {new_time_std[i]:.3f} s"
        )

    # ============================================================
    # PLOTTING
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(1, 2)

    x = np.arange(len(valid_batches))

    # ------------------------------------------------------------
    # COLORS / HATCHES
    # ------------------------------------------------------------

    old_color = "dimgray"
    new_color = "white"
    edge_color = "black"

    # ============================================================
    # (A) EXECUTION TIME
    # ============================================================

    width = 0.35

    ax1.bar(
        x - width / 2,
        old_time_mean,
        width,
        yerr=old_time_std,
        capsize=3,
        error_kw={"elinewidth": 0.8},
        label="Perin et al.",
        color=old_color,
        edgecolor=edge_color,
    )

    ax1.bar(
        x + width / 2,
        new_time_mean,
        width,
        yerr=new_time_std,
        capsize=3,
        error_kw={"elinewidth": 0.8},
        label="Framework",
        color=new_color,
        edgecolor=edge_color,
        hatch="////",
    )

    ax1.set_ylabel("Execution Time (s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_title("(a) Execution Time", size=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_batches)

    ax1.grid(
        axis="y",
        linestyle=":",
        alpha=0.6,
        color="gray",
    )

    # ============================================================
    # (B) PEAK MEMORY
    # ============================================================

    width_mem = 0.18

    # Old - epoch 1
    ax2.bar(
        x - 1.5 * width_mem,
        old_mem_1_mean,
        width_mem,
        yerr=old_mem_1_std,
        capsize=3,
        error_kw={"elinewidth": 0.8},
        label="Perin et al.: Epoch 1",
        color=old_color,
        edgecolor=edge_color,
    )

    # Old - epoch 10
    ax2.bar(
        x - 0.5 * width_mem,
        old_mem_10_mean,
        width_mem,
        yerr=old_mem_10_std,
        capsize=3,
        error_kw={"elinewidth": 0.8},
        label="Perin et al.: Epoch 10",
        color="lightgray",
        edgecolor=edge_color,
    )

    # New - epoch 1
    ax2.bar(
        x + 0.5 * width_mem,
        new_mem_1_mean,
        width_mem,
        yerr=new_mem_1_std,
        capsize=3,
        error_kw={"elinewidth": 0.8},
        label="Framework: Epoch 1",
        color="white",
        edgecolor=edge_color,
    )

    # New - epoch 10
    ax2.bar(
        x + 1.5 * width_mem,
        new_mem_10_mean,
        width_mem,
        yerr=new_mem_10_std,
        capsize=3,
        error_kw={"elinewidth": 0.8},
        label="Framework: Epoch 10",
        color="white",
        edgecolor=edge_color,
        hatch="////",
    )

    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_xlabel("Batch Size")
    ax2.set_title("(b) Peak Memory", size=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_batches)

    ax2.grid(
        axis="y",
        linestyle=":",
        alpha=0.6,
        color="gray",
    )

    # ============================================================
    # CLEAN UP
    # ============================================================

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ============================================================
    # LEGEND
    # ============================================================

    handles1, labels1 = ax1.get_legend_handles_labels()

    fig.legend(
        handles1,
        labels1,
        loc="center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        columnspacing=3.0,
        frameon=False,
    )

    # ============================================================
    # LAYOUT
    # ============================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.15,
        top=0.75,
        wspace=0.3,
    )

    # ============================================================
    # SAVE
    # ============================================================

    plot_filepath = os.path.join(
        base_path,
        "loop_parts_benchmark_plot.pdf",
    )

    plt.savefig(
        plot_filepath,
        format="pdf",
    )

    print(f"\nPlot successfully saved to: {plot_filepath}")

    plt.show()


if __name__ == "__main__":
    main()
