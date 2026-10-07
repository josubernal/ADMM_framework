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
        "figure.figsize": (3.5, 3.0),  # Slightly taller to accommodate the top legend
        "figure.dpi": 300,
        "lines.linewidth": 1.2,
    }
)


def plot_accuracy(paths):
    fig, ax = plt.subplots()

    # Define simple greyscale styles instead of many line types
    methods = [
        ("Perin et al.", paths["perin"], "dimgray", "dotted"),
        ("SSE", paths["sse"], "darkgray", "-"),
        ("CE", paths["ce"], "black", "-"),
        ("Hinge", paths["hinge"], "darkgray", "--"),
        ("CE Taylor", paths["ce_taylor"], "black", "--"),
    ]

    lines = []
    labels = []

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
        if label == "Perin et al.":
            accuracy = [acc * 100 for acc in accuracy]
        epochs = [i * 5 for i in range(1, len(accuracy) + 1)]
        (line,) = ax.plot(
            epochs,
            accuracy,
            label=label,
            color=color,
            linestyle=ls,
        )
        lines.append(line)
        labels.append(label)

    # Set labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Accuracy")

    # IEEE aesthetic cleanups
    ax.grid(True, linestyle=":", alpha=0.6, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shared Legend on top (matching your reference file structure)
    if lines:
        fig.legend(
            handles=lines,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=3,  # 3 columns fit nicely inside a 3.5-inch width
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0,
            handletextpad=0.5,
        )

    # Adjust layout leaving empty space at the top (0 to 0.82) for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.82])

    # Save as PDF
    save_path = "paper/results/Figure_1.pdf"
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot as {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Train Accuracy curves in IEEE greyscale format."
    )

    parser.add_argument(
        "--perin",
        type=str,
        default="paper/results/admm_spiking_perin_et_al/results.json",
    )
    parser.add_argument(
        "--sse",
        type=str,
        default="paper/results/admm_spiking_feedforward_sse/results.json",
    )
    parser.add_argument(
        "--ce",
        type=str,
        default="paper/results/admm_spiking_feedforward_cross_entropy/results.json",
    )
    parser.add_argument(
        "--hinge",
        type=str,
        default="paper/results/admm_spiking_feedforward_hinge/results.json",
    )
    parser.add_argument(
        "--ce_taylor",
        type=str,
        default="paper/results/admm_spiking_feedforward_cross_entropy_taylor/results.json",
    )

    args = parser.parse_args()

    paths_dict = {
        "perin": args.perin,
        "sse": args.sse,
        "ce": args.ce,
        "hinge": args.hinge,
        "ce_taylor": args.ce_taylor,
    }

    plot_accuracy(paths_dict)
