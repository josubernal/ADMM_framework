import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    base_dir = "experiments/5_Non-spiking_Depth/results"
    max_layers = 5

    # Prepare the figure: 1x3 grid for Accuracy, F1 Score, and Loss
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))
    fig.suptitle(
        "Impact of Network Depth on ADMM Training (MNIST)",
        fontsize=16,
        fontweight="bold",
    )

    ax_acc = axes[0]
    ax_f1 = axes[1]
    ax_loss = axes[2]

    # Colors for different depths to make a nice gradient/distinct lines
    colors = plt.cm.viridis(np.linspace(0, 0.9, max_layers))

    # Loop through the results for each hidden layer count
    for layers in range(1, max_layers + 1):
        json_path = os.path.join(base_dir, str(layers), "results.json")

        if not os.path.exists(json_path):
            print(f"Warning: File not found -> {json_path}")
            continue

        with open(json_path, "r") as f:
            metrics = json.load(f)

        # Extract metrics
        loss = metrics.get("loss", [])
        acc = metrics.get("accuracy", [])
        f1 = metrics.get("f1", [])

        epochs_x = range(1, len(acc) + 1)

        # Plot Accuracy
        if acc:
            ax_acc.plot(
                epochs_x,
                acc,
                label=f"{layers} Hidden Layer{'s' if layers > 1 else ''}",
                color=colors[layers - 1],
                linewidth=2,
            )

        # Plot F1 Score
        if f1:
            ax_f1.plot(
                epochs_x,
                f1,
                label=f"{layers} Hidden Layer{'s' if layers > 1 else ''}",
                color=colors[layers - 1],
                linewidth=2,
            )
        if loss:
            ax_loss.plot(
                epochs_x,
                loss,
                label=f"{layers} Hidden Layer{'s' if layers > 1 else ''}",
                color=colors[layers - 1],
                linewidth=2,
            )

    # Format Accuracy Plot
    ax_acc.set_title("Accuracy vs Epochs")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    ax_acc.legend()

    # Format F1 Plot
    ax_f1.set_title("F1 Score vs Epochs")
    ax_f1.set_xlabel("Epochs")
    ax_f1.set_ylabel("F1 Score")
    ax_f1.grid(True, linestyle="--", alpha=0.6)
    ax_f1.legend()

    # Format Loss Plot
    ax_loss.set_title("Internal Loss (SSE) vs Epochs")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("SSE")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend()

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(base_dir, "depth_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved depth comparison plot to: {save_path}")


if __name__ == "__main__":
    main()
