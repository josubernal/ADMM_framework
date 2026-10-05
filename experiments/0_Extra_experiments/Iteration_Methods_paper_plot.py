import json
import os

import matplotlib.pyplot as plt


def plot_learning_costs_ieee():
    # --- IEEE Publication Standard Configurations ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
            ],  # Fallback if Times isn't installed
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,  # IEEE REQUIREMENT: Embed TrueType fonts in PDF
            "ps.fonttype": 42,  # Embed fonts in PostScript formats as well
        }
    )

    methods = ["unrolled", "decoupled", "vectorized"]

    # Mapping internal method names to formal publication names
    display_names = {
        "unrolled": "Gauss-Seidel",
        "decoupled": "Hybrid",
        "vectorized": "Jacobi",
    }

    base_dir = "experiments/0_Extra_experiments/results/Iteration_Methods"
    plot_filename = os.path.join(base_dir, "iteration_methods_costs.pdf")

    times = []
    mems = []
    valid_methods = []

    # IEEE Grayscale Palette & Hatches
    gray_colors = ["#FFFFFF", "#DDDDDD", "#888888"]
    hatch_patterns = ["///", "...", "xxx"]

    valid_colors = []
    valid_hatches = []

    # 1. Parse the JSON files
    for i, method in enumerate(methods):
        filepath = os.path.join(base_dir, method, "results.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract isolated metrics
            t = data.get("total_execution_time_s", 0)
            m = data.get("isolated_peak_computational_memory_mb", 0)

            times.append(t)
            mems.append(m)

            # Use the formal mapping instead of just capitalizing
            valid_methods.append(display_names[method])

            valid_colors.append(gray_colors[i])
            valid_hatches.append(hatch_patterns[i])
        else:
            print(f"Warning: Could not find results for {method} at '{filepath}'")

    if not valid_methods:
        print(
            "Error: No data found to plot. Ensure the benchmark script ran successfully."
        )
        return

    # 2. Create the Figure (1 row, 2 columns) - 7 inches fits IEEE double-column width
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

    # --- Panel 1: Total Execution Time (Bar Chart) ---
    bars_time = ax1.bar(
        valid_methods,
        times,
        color=valid_colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.6,
    )

    # Apply hatch patterns to the bars
    for bar, hatch in zip(bars_time, valid_hatches):
        bar.set_hatch(hatch)

    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("(a) Total Training Time")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Add 20% headroom so text labels don't hit the top spine
    ax1.set_ylim(0, max(times) * 1.2)

    # Add text labels on top of bars
    for bar in bars_time:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(times) * 0.03),
            f"{yval:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # --- Panel 2: Peak Memory (Bar Chart) ---
    bars_mem = ax2.bar(
        valid_methods,
        mems,
        color=valid_colors,
        edgecolor="black",
        linewidth=1.2,
        width=0.6,
    )

    # Apply hatch patterns to the bars
    for bar, hatch in zip(bars_mem, valid_hatches):
        bar.set_hatch(hatch)

    ax2.set_ylabel("Computational Memory (MB)")
    ax2.set_title("(b) Peak Computational VRAM")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add 20% headroom for text
    ax2.set_ylim(0, max(mems) * 1.2)

    # Add text labels on top of bars
    for bar in bars_mem:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(mems) * 0.03),
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # Finalize and Save
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    # Export strictly as PDF
    plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
    print(f"\nPlot successfully saved as vector PDF at '{plot_filename}'")


if __name__ == "__main__":
    plot_learning_costs_ieee()
