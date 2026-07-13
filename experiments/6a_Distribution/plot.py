import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():

    # The different distribution strategies to compare
    strategies = ["baseline", "intranode_2w", "internode"]
    base_dir = "results"

    # Setup the plot: 1 row, 2 columns (Epochs & Time)
    fig, (ax_epoch, ax_time) = plt.subplots(1, 2, figsize=(14, 6))
    has_data = False

    # Define some distinct colors and markers for clarity
    styles = {
        "baseline": {
            "color": "#1f77b4",
            "marker": "o",
            "label": "Baseline (Single Node)",
        },
        "intranode_2w": {
            "color": "#ff7f0e",
            "marker": "s",
            "label": "Intranode (Multi-Node)",
        },
        "internode": {
            "color": "#2ca02c",
            "marker": "^",
            "label": "Internode (Multi-GPU)",
        },
    }

    for strategy in strategies:
        # Standard path based on main.py output: results/<strategy>/results.json
        json_path = os.path.join(
            "experiments", "6a_Distribution", base_dir, strategy, "results.json"
        )

        if not os.path.exists(json_path):
            print(f"  [!] Missing data for {strategy}: {json_path}")
            continue

        with open(json_path, "r") as f:
            metrics = json.load(f)

        has_data = True

        # Extract metrics
        acc = metrics.get("accuracy", [])
        # Handle potential key variations for time
        time_arr = metrics.get("admm_time", metrics.get("times", []))
        epochs = np.arange(1, len(acc) + 1)

        style = styles[strategy]

        # 1. Plot Accuracy vs Epoch
        if acc:
            ax_epoch.plot(
                epochs,
                acc,
                label=style["label"],
                color=style["color"],
                linewidth=1,
                marker=style["marker"],
                markersize=1,
                alpha=0.8,
            )

        # 2. Plot Accuracy vs Time
        if acc and time_arr:
            # Ensure array lengths match cleanly before plotting
            min_len = min(len(acc), len(time_arr))
            ax_time.plot(
                time_arr[:min_len],
                acc[:min_len],
                label=style["label"],
                color=style["color"],
                linewidth=1,
                marker=style["marker"],
                markersize=1,
                alpha=0.8,
            )

    if not has_data:
        print(f"Error: No data found in {base_dir}/")
        plt.close(fig)
        return

    # Formatting: Accuracy vs Epochs
    ax_epoch.set_title("Accuracy vs Epochs")
    ax_epoch.set_xlabel("Epochs")
    ax_epoch.set_ylabel("Accuracy (%)")
    ax_epoch.grid(True, linestyle="--", alpha=0.6)
    ax_epoch.legend()

    # Formatting: Accuracy vs Time
    ax_time.set_title("Accuracy vs Time")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("Accuracy (%)")
    ax_time.grid(True, linestyle="--", alpha=0.6)
    ax_time.legend()

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(
        "experiments", "6a_Distribution", base_dir, "distribution_comparison.png"
    )
    plt.savefig(save_path, dpi=300)
    print(f"Success! Saved plot to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
