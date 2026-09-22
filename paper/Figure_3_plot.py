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


def visible_std(mean, std, relative_threshold=0.01):
    if std / abs(mean) < relative_threshold:
        return np.nan
    return std


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

        if len(data[bs]) < n_repeats:
            print(f"Warning: Batch size {bs} has {len(data[bs])}/{n_repeats} trials.")

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
    # LOAD ALL TRIALS
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

    # Only keep batch sizes for which both methods have all trials
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
    # EXTRACT MEAN + STD FOR EACH BATCH SIZE
    # ============================================================

    old_time_params_mean = []
    old_time_params_std = []

    old_time_states_mean = []
    old_time_states_std = []

    old_time_total_mean = []
    old_time_total_std = []

    new_time_params_mean = []
    new_time_params_std = []

    new_time_states_mean = []
    new_time_states_std = []

    new_time_total_mean = []
    new_time_total_std = []

    old_mem_weights_mean = []
    old_mem_weights_std = []

    old_mem_states_mean = []
    old_mem_states_std = []

    new_mem_cov_weights_mean = []
    new_mem_cov_weights_std = []

    new_mem_states_mean = []
    new_mem_states_std = []

    for bs in valid_batches:
        old_trials = old_data[bs]
        new_trials = new_data[bs]

        # --------------------------------------------------------
        # Execution time
        # --------------------------------------------------------

        old_params = [d["detailed_parts"]["total_weight_time"] for d in old_trials]

        old_states = [
            d["detailed_parts"]["total_act_time"] + d["detailed_parts"]["total_z_time"]
            for d in old_trials
        ]

        old_total = [p + s for p, s in zip(old_params, old_states)]

        new_params = [
            d["detailed_parts"]["total_phase1_cov_time"]
            + d["detailed_parts"]["total_phase2_weight_time"]
            for d in new_trials
        ]

        new_states = [
            d["detailed_parts"]["total_phase3_state_time"] for d in new_trials
        ]

        new_total = [p + s for p, s in zip(new_params, new_states)]

        mean, std = mean_std(old_params)
        old_time_params_mean.append(mean)
        old_time_params_std.append(std)

        mean, std = mean_std(old_states)
        old_time_states_mean.append(mean)
        old_time_states_std.append(std)

        mean, std = mean_std(old_total)
        old_time_total_mean.append(mean)
        old_time_total_std.append(std)

        mean, std = mean_std(new_params)
        new_time_params_mean.append(mean)
        new_time_params_std.append(std)

        mean, std = mean_std(new_states)
        new_time_states_mean.append(mean)
        new_time_states_std.append(std)

        mean, std = mean_std(new_total)
        new_time_total_mean.append(mean)
        new_time_total_std.append(std)

        # --------------------------------------------------------
        # Peak memory
        # --------------------------------------------------------

        old_mem_weights = [
            d["detailed_parts"]["peak_weight_mem_mb"] for d in old_trials
        ]

        old_mem_states = [
            max(
                d["detailed_parts"]["peak_act_mem_mb"],
                d["detailed_parts"]["peak_z_mem_mb"],
            )
            for d in old_trials
        ]

        new_mem_cov_weights = [
            max(
                d["detailed_parts"]["peak_phase1_cov_mem_mb"],
                d["detailed_parts"]["peak_phase2_weight_mem_mb"],
            )
            for d in new_trials
        ]

        new_mem_states = [
            d["detailed_parts"]["peak_phase3_state_mem_mb"] for d in new_trials
        ]

        mean, std = mean_std(old_mem_weights)
        old_mem_weights_mean.append(mean)
        old_mem_weights_std.append(std)

        mean, std = mean_std(old_mem_states)
        old_mem_states_mean.append(mean)
        old_mem_states_std.append(std)

        mean, std = mean_std(new_mem_cov_weights)
        new_mem_cov_weights_mean.append(mean)
        new_mem_cov_weights_std.append(std)

        mean, std = mean_std(new_mem_states)
        new_mem_states_mean.append(mean)
        new_mem_states_std.append(std)

    old_time_total_std_plot = [
        visible_std(mean, std)
        for mean, std in zip(old_time_total_mean, old_time_total_std)
    ]

    new_time_total_std_plot = [
        visible_std(mean, std)
        for mean, std in zip(new_time_total_mean, new_time_total_std)
    ]

    old_memory_weights_std_plot = [
        visible_std(mean, std)
        for mean, std in zip(old_mem_weights_mean, old_mem_weights_std)
    ]
    old_memory_states_std_plot = [
        visible_std(mean, std)
        for mean, std in zip(old_mem_states_mean, old_mem_states_std)
    ]

    new_memory_weights_std_plot = [
        visible_std(mean, std)
        for mean, std in zip(new_mem_cov_weights_mean, new_mem_cov_weights_std)
    ]
    new_memory_states_std_plot = [
        visible_std(mean, std)
        for mean, std in zip(new_mem_states_mean, new_mem_states_std)
    ]

    print("old weights:", old_memory_weights_std_plot)
    print("old states:", old_memory_states_std_plot)
    print("new weights:", new_memory_weights_std_plot)
    print("new states:", new_memory_states_std_plot)
    # ============================================================
    # PLOTTING
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(1, 2)

    x = np.arange(len(valid_batches))
    width = 0.35

    # IEEE grayscale / hatch styling
    c_sota_param = "dimgray"
    c_sota_state = "lightgray"
    c_new_param = "black"
    c_new_state = "white"
    edge_color = "black"

    # ============================================================
    # 1. EXECUTION TIME
    # ============================================================

    # Perin et al. -- Parameters
    ax1.bar(
        x - width / 2,
        old_time_params_mean,
        width,
        label="Perin et al.: Parameters",
        color=c_sota_param,
        edgecolor=edge_color,
    )

    # Perin et al. -- States
    # Error bar is placed on the TOTAL height of the stack.
    ax1.bar(
        x - width / 2,
        old_time_states_mean,
        width,
        bottom=old_time_params_mean,
        label="Perin et al.: States",
        color=c_sota_state,
        edgecolor=edge_color,
        yerr=old_time_total_std_plot,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    # Framework -- Parameters
    ax1.bar(
        x + width / 2,
        new_time_params_mean,
        width,
        label="Framework: Parameters",
        color=c_new_param,
        edgecolor=edge_color,
    )

    # Framework -- States
    # Error bar is placed on the TOTAL height of the stack.
    ax1.bar(
        x + width / 2,
        new_time_states_mean,
        width,
        bottom=new_time_params_mean,
        label="Framework: States",
        color=c_new_state,
        edgecolor=edge_color,
        hatch="////",
        yerr=new_time_total_std_plot,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    ax1.set_ylabel("Execution Time (s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_title("(a) Execution Time Breakdown", size=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_batches)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, color="gray")

    # ============================================================
    # 2. PEAK MEMORY
    # ============================================================

    width_mem = 0.2

    # Perin et al. -- Parameter memory
    ax2.bar(
        x - width_mem * 1.5,
        old_mem_weights_mean,
        width_mem,
        label="SOTA: Parameter Peak Mem",
        color=c_sota_param,
        edgecolor=edge_color,
        yerr=old_memory_weights_std_plot,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    # Perin et al. -- State memory
    ax2.bar(
        x - width_mem * 0.5,
        old_mem_states_mean,
        width_mem,
        label="SOTA: State Peak Mem",
        color=c_sota_state,
        edgecolor=edge_color,
        yerr=old_memory_states_std_plot,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    # Framework -- Parameter memory
    ax2.bar(
        x + width_mem * 0.5,
        new_mem_cov_weights_mean,
        width_mem,
        label="Framework: Parameter Peak Mem",
        color=c_new_param,
        edgecolor=edge_color,
        yerr=new_memory_weights_std_plot,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    # Framework -- State memory
    ax2.bar(
        x + width_mem * 1.5,
        new_mem_states_mean,
        width_mem,
        label="Framework: State Peak Mem",
        color=c_new_state,
        edgecolor=edge_color,
        hatch="////",
        yerr=new_memory_states_std_plot,
        capsize=3,
        error_kw={"elinewidth": 0.8},
    )

    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_xlabel("Batch Size")
    ax2.set_title("(b) Peak Memory Comparison", size=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_batches)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, color="gray")

    # ============================================================
    # CLEAN UP
    # ============================================================

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ============================================================
    # SHARED LEGEND
    # ============================================================

    handles, _ = ax1.get_legend_handles_labels()

    shared_labels = [
        "Perin et al.: Parameters",
        "Perin et al.: States",
        "Framework: Parameters",
        "Framework: States",
    ]

    fig.legend(
        handles,
        shared_labels,
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

    plot_filepath = "paper/results/Figure_3.pdf"

    plt.savefig(plot_filepath, format="pdf")

    print(f"Plot successfully saved to: {plot_filepath}")

    plt.show()


if __name__ == "__main__":
    main()
