import json
import os

import matplotlib.pyplot as plt
import numpy as np

# IEEE publication standard settings for a single-column figure
plt.rcParams.update(
    {
        "font.family": "serif",  # IEEE uses serif fonts (usually Times)
        "font.size": 8,  # Standard text size for ticks/labels
        "axes.labelsize": 9,  # Axis labels slightly larger
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (
            3.5,
            3.0,
        ),  # 3.5 inches wide (1 column). 3.0 height allows room for rotated labels.
        "figure.dpi": 300,  # High resolution for print
        "hatch.linewidth": 0.5,  # Keep hatch lines thin and clean
    }
)


def main():
    # Define the worker configurations with IEEE-compliant grayscale & hatch styling
    configs = {
        "adam": {
            "folder": "adam",
            "color": "black",
            "hatch": "",
            "label": "Adam (GPU)",
        },
        "baseline": {
            "folder": "baseline",
            "color": "dimgray",
            "hatch": "",
            "label": "1 Worker (GPU)",
        },
        "intranode_2w": {
            "folder": "intranode_2w",
            "color": "darkgray",
            "hatch": "////",
            "label": "2 Workers (GPU)",
        },
        "intranode_3w": {
            "folder": "intranode_3w",
            "color": "lightgray",
            "hatch": "\\\\\\\\",
            "label": "3 Workers (GPU)",
        },
        "intranode_4w": {
            "folder": "intranode_4w",
            "color": "white",
            "hatch": "....",
            "label": "4 Workers (GPU)",
        },
        # "cpu_dist_32w": {
        #     "folder": "cpu_dist_32w",
        #     "color": "darkgray",
        #     "hatch": "xxxx",
        #     "label": "32 Workers (CPU)",
        # },
        # "cpu_dist_64w": {
        #     "folder": "cpu_dist_64w",
        #     "color": "white",
        #     "hatch": "----",
        #     "label": "64 Workers (CPU)",
        # },
    }

    base_dir = "results"
    extracted_data = []

    # 1. Extract the data
    for config_id, style in configs.items():
        json_path = os.path.join(
            "experiments", "6b_Distribution", base_dir, style["folder"], "results.json"
        )

        if not os.path.exists(json_path):
            print(f"  [!] Missing data for {style['label']}: {json_path}")
            continue

        with open(json_path, "r") as f:
            metrics = json.load(f)

        time_arr = metrics.get("admm_time", metrics.get("times", []))

        if time_arr:
            total_time = time_arr[-1]
            extracted_data.append(
                {
                    "label": style["label"],
                    "time": total_time,
                    "color": style["color"],
                    "hatch": style["hatch"],
                }
            )

    if not extracted_data:
        print(f"Error: No valid time data found in {base_dir}/")
        return

    # 2. Sort the data from lowest time to highest time
    extracted_data.sort(key=lambda x: x["time"])

    # Unpack sorted data for plotting
    sorted_labels = [item["label"] for item in extracted_data]
    sorted_times = [item["time"] for item in extracted_data]
    sorted_colors = [item["color"] for item in extracted_data]
    sorted_hatches = [item["hatch"] for item in extracted_data]

    # 3. Create the IEEE-styled plot
    fig, ax = plt.subplots()

    x_positions = np.arange(len(sorted_labels))
    bars = ax.bar(
        x_positions,
        sorted_times,
        color=sorted_colors,
        edgecolor="black",
        width=0.6,
        zorder=3,  # Ensures bars are drawn on top of the grid
    )

    # Apply the distinct hatch patterns to each bar
    for bar, hatch_pattern in zip(bars, sorted_hatches):
        bar.set_hatch(hatch_pattern)

    # Formatting axes and labels
    ax.set_ylabel("Total Running Time (s)")
    # ax.set_xlabel("Configuration") # Omitted because x-tick labels make it obvious

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right")

    # IEEE aesthetic cleanups
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="gray", zorder=0)

    plt.tight_layout()

    # 4. Save the figure
    save_path = os.path.join(
        "experiments", "6b_Distribution", base_dir, "total_running_time_ieee.pdf"
    )
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Success! Saved plot to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
