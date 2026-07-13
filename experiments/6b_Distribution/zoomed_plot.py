import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Define the worker configurations to compare (CPU configs removed)
    configs = {
        "adam": {
            "folder": "adam",
            "color": "#f509e5",
            "label": "Adam (Baseline)",
        },
        "baseline": {
            "folder": "baseline",
            "color": "#1f77b4",
            "label": "1 Worker (Baseline)",
        },
        "intranode_2w": {
            "folder": "intranode_2w",
            "color": "#ff7f0e",
            "label": "2 Workers (Intranode)",
        },
        "intranode_3w": {
            "folder": "intranode_3w",
            "color": "#2ca02c",
            "label": "3 Workers (Intranode)",
        },
        "intranode_4w": {
            "folder": "intranode_4w",
            "color": "#d62728",
            "label": "4 Workers (Intranode)",
        },
    }

    base_dir = "results"

    # Setup the plot: single axis for the bar plot
    fig, ax_total_bar = plt.subplots(figsize=(10, 6))
    has_data = False

    # Data tracking for Total Time Bar Plot
    bar_total_labels = []
    bar_total_times = []
    bar_total_colors = []

    for config_id, style in configs.items():
        # Standard path: experiments/6b_Distribution/results/<folder_name>/results.json
        json_path = os.path.join(
            "experiments", "6b_Distribution", base_dir, style["folder"], "results.json"
        )

        if not os.path.exists(json_path):
            print(f"  [!] Missing data for {style['label']}: {json_path}")
            continue

        with open(json_path, "r") as f:
            metrics = json.load(f)

        has_data = True

        # Extract metrics
        time_arr = metrics.get("admm_time", metrics.get("times", []))

        # --- Collect Total Time ---
        if time_arr:
            bar_total_labels.append(style["label"])
            bar_total_times.append(time_arr[-1])
            bar_total_colors.append(style["color"])
        else:
            print(f"  [-] {style['label']} is missing time array data.")

    if not has_data or not bar_total_times:
        print(f"Error: No time data found in {base_dir}/")
        plt.close(fig)
        return

    # Formatting: Total Running Time Bar Plot
    x_positions = np.arange(len(bar_total_labels))
    ax_total_bar.bar(x_positions, bar_total_times, color=bar_total_colors, alpha=0.8)

    # Add values on top of bars for clarity
    for i, v in enumerate(bar_total_times):
        ax_total_bar.text(
            i,
            v + (max(bar_total_times) * 0.01),
            f"{v:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_total_bar.set_title("Total Running Time", fontsize=14, pad=15)
    ax_total_bar.set_ylabel("Time (seconds)", fontsize=12)
    ax_total_bar.set_xticks(x_positions)
    ax_total_bar.set_xticklabels(bar_total_labels, rotation=35, ha="right", fontsize=10)
    ax_total_bar.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(
        "experiments", "6b_Distribution", base_dir, "intranode_total_time_bar.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Success! Saved plot to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
