import json
import os

import matplotlib.pyplot as plt
import numpy as np


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    x = np.arange(len(valid_batches))
    width = 0.35

    # ------------------------------------------
    # 1. Execution Time (Stacked Bar Chart)
    # ------------------------------------------

    # Old Model Stack
    ax1.bar(
        x - width / 2,
        old_time_params,
        width,
        label="Old: Parameter Updates",
        color="#1f77b4",  # Dark Blue
        edgecolor="black",
    )
    ax1.bar(
        x - width / 2,
        old_time_states_combined,
        width,
        bottom=old_time_params,
        label="Old: State Updates (a, z)",
        color="#aec7e8",  # Light Blue
        edgecolor="black",
    )

    # New Model Stack
    ax1.bar(
        x + width / 2,
        new_time_params,
        width,
        label="New: Parameter Updates",
        color="#2ca02c",  # Dark Green
        edgecolor="black",
    )
    ax1.bar(
        x + width / 2,
        new_time_states_combined,
        width,
        bottom=new_time_params,
        label="New: State Updates (a, z)",
        color="#98df8a",  # Light Green
        edgecolor="black",
    )

    ax1.set_ylabel("Execution Time (Seconds)")
    ax1.set_xlabel("Batch Size")
    ax1.set_title("Execution Time Breakdown per Logical Block")
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_batches)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # ------------------------------------------
    # 2. Peak Memory (Grouped Bar Chart)
    # ------------------------------------------

    width_mem = 0.2
    # Old model memory lines
    ax2.bar(
        x - width_mem * 1.5,
        old_mem_weights,
        width_mem,
        label="Old: Parameter Peak Mem",
        color="#1f77b4",  # Dark Blue
    )
    ax2.bar(
        x - width_mem * 0.5,
        old_mem_states,
        width_mem,
        label="Old: State Peak Mem",
        color="#aec7e8",  # Light Blue
    )

    # New model memory lines
    ax2.bar(
        x + width_mem * 0.5,
        new_mem_cov_weights,
        width_mem,
        label="New: Parameter Peak Mem",
        color="#2ca02c",  # Dark Green
    )
    ax2.bar(
        x + width_mem * 1.5,
        new_mem_states,
        width_mem,
        label="New: State Peak Mem",
        color="#98df8a",  # Light Green
    )

    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_xlabel("Batch Size")
    ax2.set_title("Peak Memory Comparison per Logical Block")
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_batches)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Layout adjustment and save
    plt.tight_layout()
    plot_filepath = os.path.join(base_path, "loop_parts_benchmark_plot.png")
    plt.savefig(plot_filepath, dpi=300)
    print(f"Plot successfully saved to: {plot_filepath}")
    plt.show()


if __name__ == "__main__":
    main()
