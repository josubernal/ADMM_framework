import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Apply IEEE standard formatting (Serif fonts, specific sizing)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (7.16, 5.5),  # 7.16 inches = max double-column width for IEEE
        "lines.linewidth": 1.5,
    }
)

# Define a Black & White / Grayscale friendly palette with distinct linestyles
STYLES = {
    "ADMM": {"label": "ADMM", "color": "black", "ls": "-"},
    "Full_Adam": {"label": "Full-Batch Adam", "color": "black", "ls": "--"},
    "Mini_Adam": {"label": "Mini-Batch Adam", "color": "dimgray", "ls": "-."},
    "Full_SGD": {"label": "Full-Batch SGD", "color": "black", "ls": ":"},
    "Mini_SGD": {
        "label": "Mini-Batch SGD",
        "color": "darkgray",
        "ls": (0, (3, 1, 1, 1, 1, 1)),
    },  # dash-dot-dot
}


def match_lengths(x_arr, y_arr):
    min_len = min(len(x_arr), len(y_arr))
    return x_arr[:min_len], y_arr[:min_len]


def plot_model_row(axes_row, metrics, title_prefix):
    # --- Extract Time Arrays ---
    times = {
        "Full_Adam": metrics.get("gd_time", metrics.get("adam_time", [])),
        "Mini_Adam": metrics.get("mini_time", metrics.get("mini_adam_time", [])),
        "Full_SGD": metrics.get("sgd_time", []),
        "Mini_SGD": metrics.get("mini_sgd_time", []),
        "ADMM": metrics.get("admm_time", []),
    }

    # --- Extract Accuracy Arrays ---
    accs = {
        "Full_Adam": metrics.get("gd_accuracy", metrics.get("adam_accuracy", [])),
        "Mini_Adam": metrics.get(
            "mini_accuracy", metrics.get("mini_adam_accuracy", [])
        ),
        "Full_SGD": metrics.get("sgd_accuracy", []),
        "Mini_SGD": metrics.get("mini_sgd_accuracy", []),
        "ADMM": metrics.get("accuracy", []),
    }

    ax_ep, ax_t = axes_row

    # --- Plot vs Epochs ---
    for key in ["Full_Adam", "Mini_Adam", "Full_SGD", "Mini_SGD", "ADMM"]:
        y_data = accs[key]
        if y_data:
            epochs = np.arange(1, len(y_data) + 1)
            ax_ep.plot(
                epochs,
                y_data,
                label=STYLES[key]["label"],
                color=STYLES[key]["color"],
                ls=STYLES[key]["ls"],
            )

    ax_ep.set_title(f"{title_prefix}: Train Accuracy vs Epochs")
    ax_ep.set_xlabel("Epochs")
    ax_ep.set_ylabel("Train Accuracy (%)")
    ax_ep.grid(True, linestyle=":", alpha=0.7)

    # --- Plot vs Time ---
    for key in ["Full_Adam", "Mini_Adam", "Full_SGD", "Mini_SGD", "ADMM"]:
        y_data = accs[key]
        t_data = times[key]
        if y_data and t_data:
            t, y = match_lengths(t_data, y_data)
            ax_t.plot(
                t,
                y,
                label=STYLES[key]["label"],
                color=STYLES[key]["color"],
                ls=STYLES[key]["ls"],
            )

    ax_t.set_title(f"{title_prefix}: Train Accuracy vs Time")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Train Accuracy (%)")
    ax_t.grid(True, linestyle=":", alpha=0.7)


def main():
    base_dir = "experiments/3_SGD_evaluation/results"

    # Load CNN data
    cnn_path = os.path.join(base_dir, "conv", "results.json")
    if os.path.exists(cnn_path):
        with open(cnn_path, "r") as f:
            cnn_metrics = json.load(f)
    else:
        print(f"File not found: {cnn_path}")
        cnn_metrics = {}

    # Load CSNN data
    csnn_path = os.path.join(base_dir, "spiking-conv", "results.json")
    if os.path.exists(csnn_path):
        with open(csnn_path, "r") as f:
            csnn_metrics = json.load(f)
    else:
        print(f"File not found: {csnn_path}")
        csnn_metrics = {}

    # Create 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2)

    # Plot top row (CNN) and bottom row (CSNN)
    if cnn_metrics:
        plot_model_row(axes[0], cnn_metrics, "CNN")
    if csnn_metrics:
        plot_model_row(axes[1], csnn_metrics, "Spiking CNN")

    # Add a single shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=5,
            frameon=False,
            handlelength=2.5,
        )

    fig.tight_layout()

    # Save the consolidated figure as PDF for IEEE/LaTeX
    save_path = os.path.join(base_dir, "convolutional_comparison_combined.pdf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300, format="pdf")
    print(f"Saved IEEE-formatted PDF plot to: {save_path}")


if __name__ == "__main__":
    main()
