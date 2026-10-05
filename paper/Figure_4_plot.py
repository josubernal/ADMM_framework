import argparse
import json
import os

import matplotlib.pyplot as plt

# IEEE publication standard settings
plt.rcParams.update(
    {
        "font.family": "serif",  # IEEE uses serif fonts (usually Times)
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (7.0, 3.0),  # Wider to accommodate side-by-side plots
        "figure.dpi": 300,
        "lines.linewidth": 1.2,
    }
)


def plot_accuracy_and_time(paths):
    # Create 1 row, 2 columns. The line plot gets more width (2:1 ratio)
    fig, (ax_line, ax_bar) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1.2]})

    # Define simple greyscale styles for the 3 new methods
    methods = [
        ("Hybrid", paths["hybrid"], "black", "-"),
        ("Jacobi", paths["jacobi"], "darkgray", "--"),
        ("Gauss-Seidel", paths["gauss_seidel"], "gray", "-."),
    ]

    lines = []
    labels = []
    times = []
    bar_colors = []

    # Load and plot each available file gracefully
    for label, path, color, ls in methods:
        if not os.path.exists(path):
            print(f"Warning: File not found, skipping '{label}': {path}")
            continue

        with open(path, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON, skipping '{label}': {path}")
                continue

        accuracy = metrics.get("accuracy", [])
        if not accuracy:
            print(f"Warning: No 'accuracy' data found, skipping '{label}': {path}")
            continue

        # Plot Line (Accuracy)
        epochs = [i * 5 for i in range(1, len(accuracy) + 1)]
        (line,) = ax_line.plot(
            epochs,
            accuracy,
            label=label,
            color=color,
            linestyle=ls,
        )
        lines.append(line)
        labels.append(label)

        # Extract Time - grabbing ONLY the last time from the list
        t = metrics.get("admm_time", 0.0)
        if isinstance(t, list) and len(t) > 0:
            t = t[-1]  # [-1] gets the last element instead of [:-1]
        elif isinstance(t, list):
            t = 0.0

        times.append(t)
        bar_colors.append(color)

    # --- Setup Line Plot (ax_line) ---
    ax_line.set_xlabel("Epoch")
    ax_line.set_ylabel("Train Accuracy")
    ax_line.grid(True, linestyle=":", alpha=0.6, color="gray")
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)

    # --- Setup Bar Plot (ax_bar) ---
    if times and any(t > 0 for t in times):
        ax_bar.bar(labels, times, color=bar_colors, edgecolor="black", alpha=0.8)
        ax_bar.set_ylabel("Total Time (s)")
        ax_bar.grid(True, axis="y", linestyle=":", alpha=0.6, color="gray")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
    else:
        print("Warning: No valid time data found to plot in the bar chart.")

    # Shared Legend on top - Adjusted to be closer to the plots
    if lines:
        fig.legend(
            handles=lines,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.95),  # Lowered from 1.05 to hug the plot closer
            ncol=3,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.5,
            handletextpad=0.5,
        )

    # Adjust layout leaving empty space at the top for the legend
    # Changed rect from 0.90 to 0.95 to reduce the empty white space at the top
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save as PDF
    save_path = "paper/results/Figure_1.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot as {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Train Accuracy curves and Total Time in IEEE greyscale format."
    )

    # Updated arguments to match the 3 requested methods
    parser.add_argument(
        "--hybrid",
        type=str,
        default="paper/results/admm_spiking_feedforward_cross_entropy_taylor/results.json",
    )
    parser.add_argument(
        "--jacobi",
        type=str,
        default="paper/results/admm_spiking_feedforward_jacobi/results.json",
    )
    parser.add_argument(
        "--gauss_seidel",
        type=str,
        default="paper/results/admm_spiking_feedforward_gauss_seidel/results.json",
    )

    args = parser.parse_args()

    paths_dict = {
        "hybrid": args.hybrid,
        "jacobi": args.jacobi,
        "gauss_seidel": args.gauss_seidel,
    }

    plot_accuracy_and_time(paths_dict)
