import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# IEEE FORMATTING
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (7.16, 5.5),
        "lines.linewidth": 1.5,
    }
)


# ============================================================
# STYLES
# ============================================================

STYLES = {
    "ADMM": {
        "label": "ADMM",
        "color": "black",
        "ls": "-",
    },
    "Full_Adam": {
        "label": "Full-Batch Adam",
        "color": "dimgray",
        "ls": ":",
    },
    "Mini_Adam": {
        "label": "Mini-Batch Adam",
        "color": "dimgray",
        "ls": "--",
    },
    "Full_SGD": {
        "label": "Full-Batch SGD",
        "color": "lightgray",
        "ls": ":",
    },
    "Mini_SGD": {
        "label": "Mini-Batch SGD",
        "color": "lightgray",
        "ls": "--",
    },
}


METHODS = [
    "Full_Adam",
    "Mini_Adam",
    "Full_SGD",
    "Mini_SGD",
    "ADMM",
]


# ============================================================
# LOAD JSON
# ============================================================


def load_results(base_dir, folder_name):

    path = os.path.join(
        base_dir,
        folder_name,
        "results.json",
    )

    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return {}

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# GET METHOD RESULTS
# ============================================================


def get_method_results(base_dir, folder_name, accuracy_key, time_key):

    metrics = load_results(
        base_dir,
        folder_name,
    )

    if not metrics:
        return [], []

    accuracy = metrics.get(
        accuracy_key,
        [],
    )

    time = metrics.get(
        time_key,
        [],
    )

    if not accuracy:
        print(f"Warning: '{accuracy_key}' not found in {folder_name}/results.json")

    if not time:
        print(f"Warning: '{time_key}' not found in {folder_name}/results.json")

    return accuracy, time


# ============================================================
# PLOT ONE METHOD
# ============================================================


def plot_method_epoch(
    ax,
    accuracy,
    method,
):

    if len(accuracy) == 0:
        return

    epochs = np.arange(
        1,
        len(accuracy) + 1,
    )

    ax.plot(
        epochs,
        accuracy,
        label=STYLES[method]["label"],
        color=STYLES[method]["color"],
        ls=STYLES[method]["ls"],
    )


def plot_method_time(
    ax,
    accuracy,
    time,
    method,
):

    if len(accuracy) == 0 or len(time) == 0:
        return

    min_len = min(
        len(accuracy),
        len(time),
    )

    accuracy = accuracy[:min_len]
    time = time[:min_len]

    ax.plot(
        time,
        accuracy,
        label=STYLES[method]["label"],
        color=STYLES[method]["color"],
        ls=STYLES[method]["ls"],
    )


# ============================================================
# PLOT ONE ARCHITECTURE
# ============================================================


def plot_architecture(
    ax_epoch,
    ax_time,
    base_dir,
    architecture,
    methods,
):

    for method, config in methods.items():
        folder = config["folder"]
        accuracy_key = config["accuracy"]
        time_key = config["time"]

        accuracy, time = get_method_results(
            base_dir,
            folder,
            accuracy_key,
            time_key,
        )

        plot_method_epoch(
            ax_epoch,
            accuracy,
            method,
        )

        plot_method_time(
            ax_time,
            accuracy,
            time,
            method,
        )

    ax_epoch.set_title(architecture)

    ax_epoch.set_xlabel("Epochs")

    ax_time.set_xlabel("Time (s)")

    ax_epoch.grid(
        True,
        linestyle=":",
        alpha=0.7,
    )

    ax_time.grid(
        True,
        linestyle=":",
        alpha=0.7,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    base_dir = "paper/results"

    # ========================================================
    # RESULT FOLDER / JSON KEY STRUCTURE
    # ========================================================

    architectures = {
        "Feedforward": {
            "ADMM": {
                "folder": "admm_feedforward",
                "accuracy": "accuracy",
                "time": "admm_time",
            },
            "Full_Adam": {
                "folder": "full_batch_feedforward",
                "accuracy": "gd_accuracy",
                "time": "gd_time",
            },
            "Mini_Adam": {
                "folder": "mini_batch_feedforward",
                "accuracy": "mini_adam_accuracy",
                "time": "mini_adam_time",
            },
            "Full_SGD": {
                "folder": "full_batch_feedforward",
                "accuracy": "sgd_accuracy",
                "time": "sgd_time",
            },
            "Mini_SGD": {
                "folder": "mini_batch_feedforward",
                "accuracy": "mini_sgd_accuracy",
                "time": "mini_sgd_time",
            },
        },
        "Spiking Feedforward": {
            "ADMM": {
                "folder": "admm_spiking_feedforward",
                "accuracy": "accuracy",
                "time": "admm_time",
            },
            "Full_Adam": {
                "folder": "full_batch_spiking_feedforward",
                "accuracy": "gd_accuracy",
                "time": "gd_time",
            },
            "Mini_Adam": {
                "folder": "mini_batch_spiking_feedforward",
                "accuracy": "mini_adam_accuracy",
                "time": "mini_adam_time",
            },
            "Full_SGD": {
                "folder": "full_batch_spiking_feedforward",
                "accuracy": "sgd_accuracy",
                "time": "sgd_time",
            },
            "Mini_SGD": {
                "folder": "mini_batch_spiking_feedforward",
                "accuracy": "mini_sgd_accuracy",
                "time": "mini_sgd_time",
            },
        },
        "Convolutional": {
            "ADMM": {
                "folder": "admm_conv",
                "accuracy": "accuracy",
                "time": "admm_time",
            },
            "Full_Adam": {
                "folder": "full_batch_conv",
                "accuracy": "gd_accuracy",
                "time": "gd_time",
            },
            "Mini_Adam": {
                "folder": "mini_batch_conv",
                "accuracy": "mini_adam_accuracy",
                "time": "mini_adam_time",
            },
            "Full_SGD": {
                "folder": "full_batch_conv",
                "accuracy": "sgd_accuracy",
                "time": "sgd_time",
            },
            "Mini_SGD": {
                "folder": "mini_batch_conv",
                "accuracy": "mini_sgd_accuracy",
                "time": "mini_sgd_time",
            },
        },
        "Spiking Convolutional": {
            "ADMM": {
                "folder": "admm_spiking_conv",
                "accuracy": "accuracy",
                "time": "admm_time",
            },
            "Full_Adam": {
                "folder": "full_batch_spiking_conv",
                "accuracy": "gd_accuracy",
                "time": "gd_time",
            },
            "Mini_Adam": {
                "folder": "mini_batch_spiking_conv",
                "accuracy": "mini_adam_accuracy",
                "time": "mini_adam_time",
            },
            "Full_SGD": {
                "folder": "full_batch_spiking_conv",
                "accuracy": "sgd_accuracy",
                "time": "sgd_time",
            },
            "Mini_SGD": {
                "folder": "mini_batch_spiking_conv",
                "accuracy": "mini_sgd_accuracy",
                "time": "mini_sgd_time",
            },
        },
    }

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(7.16, 4.8),
    )

    # ========================================================
    # PLOT ALL ARCHITECTURES
    # ========================================================

    for col, (
        architecture,
        methods,
    ) in enumerate(architectures.items()):
        plot_architecture(
            ax_epoch=axes[0, col],
            ax_time=axes[1, col],
            base_dir=base_dir,
            architecture=architecture,
            methods=methods,
        )

    # ========================================================
    # AXIS LABELS
    # ========================================================

    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")

    # ========================================================
    # SHARED LEGEND — SINGLE HORIZONTAL ROW
    # ========================================================

    legend_handles = []

    for method in METHODS:
        handle = plt.Line2D(
            [],
            [],
            color=STYLES[method]["color"],
            linestyle=STYLES[method]["ls"],
            linewidth=1.5,
            label=STYLES[method]["label"],
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=5,  # <-- all in one row
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.tight_layout(
        rect=[0, 0, 1, 0.93],  # <-- leave space for legend
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_path = os.path.join(
        base_dir,
        "Figure_2.pdf",
    )

    fig.savefig(
        save_path,
        bbox_inches="tight",
        dpi=300,
        format="pdf",
    )

    print(f"Saved combined figure to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
