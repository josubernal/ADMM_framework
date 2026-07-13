import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Define the worker configurations to compare
    configs = {
        "adam": {
            "folder": "adam",
            "color": "#f509e5",
            "marker": "o",
            "label": "Adam (Baseline)",
        },
        "baseline": {
            "folder": "baseline",
            "color": "#1f77b4",
            "marker": "o",
            "label": "1 Worker (Baseline)",
        },
        "intranode_2w": {
            "folder": "intranode_2w",
            "color": "#ff7f0e",
            "marker": "s",
            "label": "2 Workers (Intranode)",
        },
        "intranode_3w": {
            "folder": "intranode_3w",
            "color": "#2ca02c",
            "marker": "^",
            "label": "3 Workers (Intranode)",
        },
        "intranode_4w": {
            "folder": "intranode_4w",
            "color": "#d62728",
            "marker": "D",
            "label": "4 Workers (Intranode)",
        },
        "cpu_dist_32w": {
            "folder": "cpu_dist_32w",
            "color": "#020202",
            "marker": "^",
            "label": "32 Workers (CPU)",
        },
        "cpu_dist_64w": {
            "folder": "cpu_dist_64w",
            "color": "#00EEFF",
            "marker": "D",
            "label": "64 Workers (CPU)",
        },
    }

    base_dir = "results"

    # Setup the plot: 2 rows, 2 columns
    fig, ((ax_epoch, ax_time), (ax_total_bar, ax_75_bar)) = plt.subplots(
        2, 2, figsize=(16, 12)
    )
    has_data = False

    # Data tracking for Total Time Bar Plot
    bar_total_labels = []
    bar_total_times = []
    bar_total_colors = []

    # Data tracking for Time to 75% Bar Plot
    bar_75_labels = []
    bar_75_times = []
    bar_75_colors = []

    for config_id, style in configs.items():
        # Standard path: results/<folder_name>/<model_name>/results.json
        json_path = os.path.join(
            "experiments", "6a_Distribution", base_dir, style["folder"], "results.json"
        )

        if not os.path.exists(json_path):
            print(f"  [!] Missing data for {style['label']}: {json_path}")
            continue

        with open(json_path, "r") as f:
            metrics = json.load(f)

        has_data = True

        # Extract metrics
        acc = metrics.get("accuracy", [])
        time_arr = metrics.get("admm_time", metrics.get("times", []))
        epochs = np.arange(1, len(acc) + 1)

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

        # 2. Plot Accuracy vs Time & Collect Bar Data
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

            # --- Collect Total Time ---
            bar_total_labels.append(style["label"])
            bar_total_times.append(time_arr[-1])
            bar_total_colors.append(style["color"])

            # --- Collect Time to 75% ---
            # Change 75.0 to 0.75 if your accuracy is on a 0.0-1.0 scale
            target_acc = 75.0
            achieved_idx = next((i for i, a in enumerate(acc) if a >= target_acc), None)

            # Only add to the 75% plot if it actually reached the threshold
            if achieved_idx is not None:
                bar_75_labels.append(style["label"])
                bar_75_times.append(time_arr[achieved_idx])
                bar_75_colors.append(style["color"])
            else:
                print(f"  [-] {style['label']} did not reach {target_acc}% accuracy.")

    if not has_data:
        print(f"Error: No data found in {base_dir}/")
        plt.close(fig)
        return

    # Formatting: Accuracy vs Epochs (Top Left)
    ax_epoch.set_title("Scaling Accuracy vs Epochs")
    ax_epoch.set_xlabel("Epochs")
    ax_epoch.set_ylabel("Accuracy (%)")
    ax_epoch.grid(True, linestyle="--", alpha=0.6)
    ax_epoch.legend()

    # Formatting: Accuracy vs Time (Top Right)
    ax_time.set_title("Scaling Accuracy vs Time")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("Accuracy (%)")
    ax_time.grid(True, linestyle="--", alpha=0.6)
    ax_time.legend()

    # Formatting: Total Running Time Bar Plot (Bottom Left)
    if bar_total_times:
        x_positions = np.arange(len(bar_total_labels))
        ax_total_bar.bar(
            x_positions, bar_total_times, color=bar_total_colors, alpha=0.8
        )
        ax_total_bar.set_title("Total Running Time")
        ax_total_bar.set_ylabel("Time (seconds)")
        ax_total_bar.set_xticks(x_positions)
        ax_total_bar.set_xticklabels(bar_total_labels, rotation=45, ha="right")
        ax_total_bar.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Formatting: Time to 75% Accuracy Bar Plot (Bottom Right)
    if bar_75_times:
        x_positions_75 = np.arange(len(bar_75_labels))
        ax_75_bar.bar(x_positions_75, bar_75_times, color=bar_75_colors, alpha=0.8)
        ax_75_bar.set_title("Time to 75% Accuracy")
        ax_75_bar.set_ylabel("Time (seconds)")
        ax_75_bar.set_xticks(x_positions_75)
        ax_75_bar.set_xticklabels(bar_75_labels, rotation=45, ha="right")
        ax_75_bar.grid(True, axis="y", linestyle="--", alpha=0.6)
    else:
        ax_75_bar.set_title("Time to 75% Accuracy")
        ax_75_bar.text(
            0.5,
            0.5,
            "No configurations reached 75%",
            ha="center",
            va="center",
            transform=ax_75_bar.transAxes,
        )

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(
        "experiments", "6a_Distribution", base_dir, "intranode_scaling_comparison.png"
    )
    plt.savefig(save_path, dpi=300)
    print(f"Success! Saved plot to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
