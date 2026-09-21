import json
import os

import matplotlib.pyplot as plt
import numpy as np

# IEEE publication standard settings
plt.rcParams.update(
    {
        "font.family": "serif",  # IEEE uses serif fonts (usually Times)
        "font.size": 8,  # Standard text size for legends/ticks
        "axes.labelsize": 8,  # Axis labels slightly larger
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (
            7.16,
            3.0,
        ),  # 7.16 inches fits exactly across a two-column IEEE page
        "figure.dpi": 300,  # High resolution for print
        "hatch.linewidth": 0.5,  # Keep hatch lines thin and clean
    }
)


def load_data(batch_sizes, base_path, model_type):
    """Loads metrics from JSON files for a given model type."""
    data = []
    for bs in batch_sizes:
        filepath = os.path.join(base_path, f"{model_type}", str(bs), "results.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data.append((bs, json.load(f)))
        else:
            print(f"Warning: Missing data for {model_type} at batch size {bs}")
    return data


def main():
    # Setup paths and batch sizes
    base_path = "experiments/4_Performance_benchmark/results/detailed"
    batch_sizes = [4, 8, 16, 32, 64, 128, 256]

    # Load data
    old_data = load_data(batch_sizes, base_path, "Old")
    new_data = load_data(batch_sizes, base_path, "New")

    # Ensure we only plot batch sizes where both models have data
    valid_batches = [bs for bs, _ in old_data if any(b == bs for b, _ in new_data)]

    if not valid_batches:
        print("No valid overlapping data found to plot.")
        return

    # ------------------------------------------
    # Extract Time Data
    # ------------------------------------------
    old_time_weights = [
        d["detailed_parts"]["total_weight_time"]
        for bs, d in old_data
        if bs in valid_batches
    ]
    old_time_act = [
        d["detailed_parts"]["total_act_time"]
        for bs, d in old_data
        if bs in valid_batches
    ]
    old_time_z = [
        d["detailed_parts"]["total_z_time"] for bs, d in old_data if bs in valid_batches
    ]

    new_time_cov = [
        d["detailed_parts"]["total_phase1_cov_time"]
        for bs, d in new_data
        if bs in valid_batches
    ]
    new_time_weights = [
        d["detailed_parts"]["total_phase2_weight_time"]
        for bs, d in new_data
        if bs in valid_batches
    ]
    new_time_states = [
        d["detailed_parts"]["total_phase3_state_time"]
        for bs, d in new_data
        if bs in valid_batches
    ]

    # Aggregate Times into 1:1 Logical Blocks
    old_time_params = old_time_weights
    old_time_states_combined = np.add(old_time_act, old_time_z)

    new_time_params = np.add(new_time_cov, new_time_weights)
    new_time_states_combined = new_time_states

    # ------------------------------------------
    # Extract Memory Data
    # ------------------------------------------
    old_mem_weights = [
        d["detailed_parts"]["peak_weight_mem_mb"]
        for bs, d in old_data
        if bs in valid_batches
    ]
    old_mem_states = [
        max(
            d["detailed_parts"]["peak_act_mem_mb"], d["detailed_parts"]["peak_z_mem_mb"]
        )
        for bs, d in old_data
        if bs in valid_batches
    ]

    new_mem_cov_weights = [
        max(
            d["detailed_parts"]["peak_phase1_cov_mem_mb"],
            d["detailed_parts"]["peak_phase2_weight_mem_mb"],
        )
        for bs, d in new_data
        if bs in valid_batches
    ]
    new_mem_states = [
        d["detailed_parts"]["peak_phase3_state_mem_mb"]
        for bs, d in new_data
        if bs in valid_batches
    ]

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2)
    x = np.arange(len(valid_batches))
    width = 0.35

    # IEEE Grayscale/Hatch Styling Palette
    c_sota_param = "dimgray"
    c_sota_state = "lightgray"
    c_new_param = "black"
    c_new_state = "white"
    edge_color = "black"

    # ------------------------------------------
    # 1. Execution Time (Stacked Bar Chart)
    # ------------------------------------------

    # SOTA Stack
    ax1.bar(
        x - width / 2,
        old_time_params,
        width,
        label="Perin et al.: Parameters",
        color=c_sota_param,
        edgecolor=edge_color,
    )
    ax1.bar(
        x - width / 2,
        old_time_states_combined,
        width,
        bottom=old_time_params,
        label="Perin et al.: States",
        color=c_sota_state,
        edgecolor=edge_color,
    )

    # Modular Framework Stack
    ax1.bar(
        x + width / 2,
        new_time_params,
        width,
        label="Framework: Parameters",
        color=c_new_param,
        edgecolor=edge_color,
    )
    ax1.bar(
        x + width / 2,
        new_time_states_combined,
        width,
        bottom=new_time_params,
        label="Framework: States",
        color=c_new_state,
        edgecolor=edge_color,
        hatch="////",  # Hatched for B&W contrast
    )

    ax1.set_ylabel("Execution Time (s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_title("(a) Execution Time Breakdown", size=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_batches)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, color="gray")

    # ------------------------------------------
    # 2. Peak Memory (Grouped Bar Chart)
    # ------------------------------------------

    width_mem = 0.2

    # SOTA Memory
    ax2.bar(
        x - width_mem * 1.5,
        old_mem_weights,
        width_mem,
        label="SOTA: Parameter Peak Mem",
        color=c_sota_param,
        edgecolor=edge_color,
    )
    ax2.bar(
        x - width_mem * 0.5,
        old_mem_states,
        width_mem,
        label="SOTA: State Peak Mem",
        color=c_sota_state,
        edgecolor=edge_color,
    )

    # Modular Framework Memory
    ax2.bar(
        x + width_mem * 0.5,
        new_mem_cov_weights,
        width_mem,
        label="Modular: Parameter Peak Mem",
        color=c_new_param,
        edgecolor=edge_color,
    )
    ax2.bar(
        x + width_mem * 1.5,
        new_mem_states,
        width_mem,
        label="Modular: State Peak Mem",
        color=c_new_state,
        edgecolor=edge_color,
        hatch="////",
    )

    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_xlabel("Batch Size")
    ax2.set_title("(b) Peak Memory Comparison", size=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_batches)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, color="gray")

    # Clean up IEEE aesthetics
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # REMOVED: ax.legend() so the local legends disappear

    # ==========================================
    # LAYOUT & LEGEND FIXES
    # ==========================================

    handles, _ = ax1.get_legend_handles_labels()
    shared_labels = [
        "Perin et al.: Parameters",
        "Perin et al.: States",
        "Framework: Parameters",
        "Framework: States",
    ]

    # 1. Place the legend in the top margin space
    leg = fig.legend(
        handles,
        shared_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.92),  # Centered vertically in the top 25% of the figure
        ncol=2,
        columnspacing=3.0,
        frameon=False,
    )

    # 2. Use manual margins INSTEAD of tight_layout
    # This guarantees your text will not be cut off while keeping the 7.16" width.
    fig.subplots_adjust(
        left=0.09,  # Reserves 9% of the width (~0.64 inches) for the y-axis label
        right=0.98,  # 2% margin on the right side
        bottom=0.15,  # 15% margin at the bottom for x-axis labels
        top=0.75,  # Caps the plot at 75% height, leaving the top 25% for the legend
        wspace=0.3,  # Spacing between ax1 and ax2
    )

    plot_filepath = os.path.join(base_path, "loop_parts_benchmark_plot.pdf")

    # Save exactly to the 7.16" canvas without bbox_inches changing the dimensions
    plt.savefig(plot_filepath, format="pdf")
    print(f"Plot successfully saved to: {plot_filepath}")

    plt.show()


if __name__ == "__main__":
    main()
