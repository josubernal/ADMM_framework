import argparse
import json

import matplotlib.pyplot as plt

# IEEE publication standard settings
plt.rcParams.update(
    {
        "font.family": "serif",  # IEEE uses serif fonts (usually Times)
        "font.size": 8,  # Standard text size for legends/ticks
        "axes.labelsize": 9,  # Axis labels slightly larger
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (3.5, 2.5),  # 3.5 inches fits exactly in one IEEE column
        "figure.dpi": 300,  # High resolution for print
        "lines.linewidth": 1.2,  # Clean, legible line thickness
    }
)


def plot_accuracy(sota_path, new_path):
    # Load JSON data
    with open(sota_path, "r") as f:
        sota_metrics = json.load(f)
    with open(new_path, "r") as f:
        new_metrics = json.load(f)

    fig, ax = plt.subplots()

    # Plot SOTA using solid black line
    ax.plot(
        sota_metrics.get("accuracy", []),
        label="Perin et al.",
        color="black",
        linestyle="-",
    )

    # Plot NEW framework using dashed black line
    ax.plot(
        new_metrics.get("accuracy", []),
        label="Framework",
        color="black",
        linestyle="--",
    )

    # Set labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy")

    # IEEE aesthetic cleanups: Light dotted grid, remove top/right bounding boxes
    ax.grid(True, linestyle=":", alpha=0.6, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Borderless legend to reduce visual clutter
    ax.legend(frameon=False)

    plt.tight_layout()

    # Save as vector graphic for LaTeX
    plt.savefig("accuracy_comparison.pdf", format="pdf", bbox_inches="tight")
    print("Saved plot as accuracy_comparison.pdf")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot SOTA vs NEW framework Train Accuracy in IEEE black-only format."
    )
    parser.add_argument(
        "--sota",
        type=str,
        default="experiments/1_SOTA_evaluation/results/SOTA/results.json",
        help="Path to the SOTA metrics.json file",
    )
    parser.add_argument(
        "--new",
        type=str,
        default="experiments/1_SOTA_evaluation/results/NEW/results.json",
        help="Path to the NEW metrics.json file",
    )

    args = parser.parse_args()

    plot_accuracy(args.sota, args.new)
