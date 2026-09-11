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
        "figure.figsize": (
            7.16,
            5.5,
        ),  # 7.16 inches = max double-column width, 5.5 height for 2 rows
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


def load_metrics(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        print(f"File not found: {path}")
        return {}


def create_and_save_figure(
    metrics_top, title_top, metrics_bottom, title_bottom, save_path, is_zoom=False
):
    # Create 2x2 grid for this specific category
    fig, axes = plt.subplots(nrows=2, ncols=2)

    # Plot top and bottom rows
    if metrics_top:
        plot_model_row(axes[0], metrics_top, title_top)
    if metrics_bottom:
        plot_model_row(axes[1], metrics_bottom, title_bottom)

    # Apply zoom logic based on reference
    if is_zoom:
        for ax in axes.flat:
            ax.set_xlim(-5, 205)
            # Both columns plot accuracy, so the y-limits apply universally
            ax.set_ylim(58, 102)

    # Add a single shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Fallback to other axes if the first one is empty
    if not handles:
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            frameon=False,
            handlelength=2.5,
        )

    fig.tight_layout()

    # Save and close to free memory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300, format="pdf")

    plot_type = "Zoomed" if is_zoom else "Standard"
    print(f"Saved {plot_type} IEEE-formatted PDF plot to: {save_path}")
    plt.close(fig)


def main():
    base_dir = "experiments/3_SGD_evaluation/results"

    # --- Load Data for 4 Models ---
    ff_metrics = load_metrics(os.path.join(base_dir, "feedforward", "results.json"))
    cnn_metrics = load_metrics(os.path.join(base_dir, "conv", "results.json"))

    sff_metrics = load_metrics(
        os.path.join(base_dir, "spiking-feedforward", "results.json")
    )
    csnn_metrics = load_metrics(os.path.join(base_dir, "spiking-conv", "results.json"))

    # ==========================================
    # 1. Non-Spiking (FF-ANN and CNN) - STANDARD
    # ==========================================
    create_and_save_figure(
        metrics_top=ff_metrics,
        title_top="FF-ANN",
        metrics_bottom=cnn_metrics,
        title_bottom="CNN",
        save_path=os.path.join(base_dir, "non_spiking_comparison_combined.pdf"),
        is_zoom=False,
    )

    # ==========================================
    # 2. Non-Spiking (FF-ANN and CNN) - ZOOMED
    # ==========================================
    create_and_save_figure(
        metrics_top=ff_metrics,
        title_top="FF-ANN",
        metrics_bottom=cnn_metrics,
        title_bottom="CNN",
        save_path=os.path.join(base_dir, "non_spiking_comparison_combined_zoomed.pdf"),
        is_zoom=True,
    )

    # ==========================================
    # 3. Spiking (Spiking FF and Spiking CNN) - STANDARD
    # ==========================================
    create_and_save_figure(
        metrics_top=sff_metrics,
        title_top="Spiking FF",
        metrics_bottom=csnn_metrics,
        title_bottom="Spiking CNN",
        save_path=os.path.join(base_dir, "spiking_comparison_combined.pdf"),
        is_zoom=False,
    )

    # ==========================================
    # 4. Spiking (Spiking FF and Spiking CNN) - ZOOMED
    # ==========================================
    create_and_save_figure(
        metrics_top=sff_metrics,
        title_top="Spiking FF",
        metrics_bottom=csnn_metrics,
        title_bottom="Spiking CNN",
        save_path=os.path.join(base_dir, "spiking_comparison_combined_zoomed.pdf"),
        is_zoom=True,
    )


if __name__ == "__main__":
    main()
