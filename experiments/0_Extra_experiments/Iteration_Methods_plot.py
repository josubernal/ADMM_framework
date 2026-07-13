import json
import os

import matplotlib.pyplot as plt


def plot_learning_curves_and_costs():
    methods = ["unrolled", "decoupled", "vectorized"]
    base_dir = "experiments/0_Extra_experiments/results/Iteration_Methods"
    plot_filename = os.path.join(base_dir, "iteration_methods_learning_curves.png")

    times = []
    mems = []
    acc_lists = []
    valid_methods = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    valid_colors = []

    # 1. Parse the JSON files
    for i, method in enumerate(methods):
        filepath = os.path.join(base_dir, method, "results.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)

                # Extract isolated metrics
                t = data.get("total_execution_time_s", 0)
                m = data.get("isolated_peak_computational_memory_mb", 0)

                # Search for the accuracy metric list
                acc = data.get(
                    "Train Accuracy",
                    data.get(
                        "test_accuracy", data.get("accuracy", data.get("Accuracy", []))
                    ),
                )

                # Ensure it's a list for the line plot
                if not isinstance(acc, list):
                    acc = [acc]

                times.append(t)
                mems.append(m)
                acc_lists.append(acc)
                valid_methods.append(method.capitalize())
                valid_colors.append(colors[i])
        else:
            print(f"Warning: Could not find results for {method} at '{filepath}'")

    if not valid_methods:
        print(
            "Error: No data found to plot. Ensure the benchmark script ran successfully."
        )
        return

    # 2. Create the Figure (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Accuracy vs. Epoch (Line Chart) ---
    for i, method in enumerate(valid_methods):
        epochs = range(len(acc_lists[i]))
        ax1.plot(
            epochs, acc_lists[i], label=method, color=valid_colors[i], linewidth=2.5
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Train Accuracy (%)", fontsize=12)
    ax1.set_title("Learning Curve: Accuracy vs. Epoch", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # --- Panel 2: Total Execution Time (Bar Chart) ---
    bars_time = ax2.bar(valid_methods, times, color=valid_colors)
    ax2.set_ylabel("Execution Time [Seconds]", fontsize=12)
    ax2.set_title("Total Training Time", fontsize=14)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add text labels on top of bars
    for bar in bars_time:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(times) * 0.01),
            f"{yval:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # --- Panel 3: Peak Memory (Bar Chart) ---
    bars_mem = ax3.bar(valid_methods, mems, color=valid_colors)
    ax3.set_ylabel("Computational Memory [MB]", fontsize=12)
    ax3.set_title("Peak Computational VRAM", fontsize=14)
    ax3.grid(axis="y", linestyle="--", alpha=0.7)

    # Add text labels on top of bars
    for bar in bars_mem:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(mems) * 0.01),
            f"{yval:.1f}MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Finalize and Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"\nPlot successfully saved as '{plot_filename}'")


if __name__ == "__main__":
    plot_learning_curves_and_costs()
