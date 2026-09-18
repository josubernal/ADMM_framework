import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Define the models and the path to the results
    model_types = ["feedforward", "conv", "spiking-feedforward", "spiking-conv"]
    base_dir = "experiments/3_SGD_evaluation/results"

    # Helper function to safely align array lengths before plotting
    def match_lengths(x_arr, y_arr):
        min_len = min(len(x_arr), len(y_arr))
        return x_arr[:min_len], y_arr[:min_len]

    for model_name in model_types:
        json_path = os.path.join(base_dir, model_name, "results.json")

        if not os.path.exists(json_path):
            print(f"Warning: File not found -> {json_path}")
            continue

        with open(json_path, "r") as f:
            metrics = json.load(f)

        # ---------------------------------------------------------
        # 1. EXTRACT DATA
        # ---------------------------------------------------------
        # Time arrays (in seconds)
        gd_time = metrics.get("gd_time", metrics.get("adam_time", []))
        mini_time = metrics.get("mini_time", metrics.get("mini_adam_time", []))
        sgd_time = metrics.get("sgd_time", [])
        mini_sgd_time = metrics.get("mini_sgd_time", [])
        admm_time = metrics.get("admm_time", [])

        # Full-Batch Adam Metrics
        gd_acc = metrics.get("gd_accuracy", metrics.get("adam_accuracy", []))
        gd_f1 = metrics.get("gd_f1", metrics.get("adam_f1", []))
        raw_gd_fr = metrics.get("gd_firing_rate", metrics.get("adam_firing_rate", []))
        gd_fr = [fr[0] if isinstance(fr, list) else fr for fr in raw_gd_fr]

        # Mini-Batch Adam Metrics
        mini_acc = metrics.get("mini_accuracy", metrics.get("mini_adam_accuracy", []))
        mini_f1 = metrics.get("mini_f1", metrics.get("mini_adam_f1", []))
        raw_mini_fr = metrics.get(
            "mini_firing_rate", metrics.get("mini_adam_firing_rate", [])
        )
        mini_fr = [fr[0] if isinstance(fr, list) else fr for fr in raw_mini_fr]

        # Full-Batch Bare SGD Metrics
        sgd_acc = metrics.get("sgd_accuracy", [])
        sgd_f1 = metrics.get("sgd_f1", [])
        raw_sgd_fr = metrics.get("sgd_firing_rate", [])
        sgd_fr = [fr[0] if isinstance(fr, list) else fr for fr in raw_sgd_fr]

        # Mini-Batch Bare SGD Metrics
        mini_sgd_acc = metrics.get("mini_sgd_accuracy", [])
        mini_sgd_f1 = metrics.get("mini_sgd_f1", [])
        raw_mini_sgd_fr = metrics.get("mini_sgd_firing_rate", [])
        mini_sgd_fr = [fr[0] if isinstance(fr, list) else fr for fr in raw_mini_sgd_fr]

        # ADMM Metrics
        admm_acc = metrics.get("accuracy", [])
        admm_f1 = metrics.get("f1", [])
        raw_admm_fr = metrics.get("firing_rate", [])
        admm_fr = [fr[0] if isinstance(fr, list) else fr for fr in raw_admm_fr]

        # Epoch arrays
        epochs_gd = np.arange(1, len(gd_acc) + 1) if gd_acc else []
        epochs_mini = np.arange(1, len(mini_acc) + 1) if mini_acc else []
        epochs_sgd = np.arange(1, len(sgd_acc) + 1) if sgd_acc else []
        epochs_mini_sgd = np.arange(1, len(mini_sgd_acc) + 1) if mini_sgd_acc else []
        epochs_admm = np.arange(1, len(admm_acc) + 1) if admm_acc else []

        is_spiking = "spiking" in model_name

        # =========================================================
        # FIGURE 1 & 2: ACCURACY & F1 (Standard & Zoomed)
        # =========================================================
        fig_main, axes_main = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        fig_zoom, axes_zoom = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

        plot_configs = [
            (fig_main, axes_main, False, f"{model_name}_comparison.png"),
            (fig_zoom, axes_zoom, True, f"{model_name}_comparison_zoomed.png"),
        ]

        # Loop through both standard and zoomed grids to avoid code duplication
        for fig, axes, is_zoom, save_filename in plot_configs:
            ax_acc_ep = axes[0, 0]
            ax_f1_ep = axes[0, 1]
            ax_acc_t = axes[1, 0]
            ax_f1_t = axes[1, 1]

            # ---------------------------------------------------------
            # ROW 1: METRICS vs EPOCHS
            # ---------------------------------------------------------
            if gd_acc:
                ax_acc_ep.plot(
                    epochs_gd,
                    gd_acc,
                    label="Full-Batch Adam",
                    ls="--",
                    color="#1f77b4",
                    linewidth=2,
                )
            if mini_acc:
                ax_acc_ep.plot(
                    epochs_mini,
                    mini_acc,
                    ls="-.",
                    label="Mini-Batch Adam",
                    color="#2ca02c",
                    linewidth=2,
                )
            if sgd_acc:
                ax_acc_ep.plot(
                    epochs_sgd,
                    sgd_acc,
                    label="Full-Batch GD",
                    ls=":",
                    color="#d62728",
                    linewidth=2,
                )
            if mini_sgd_acc:
                ax_acc_ep.plot(
                    epochs_mini_sgd,
                    mini_sgd_acc,
                    label="Mini-Batch SGD",
                    ls=":",
                    color="#9467bd",
                    linewidth=2,
                )
            if admm_acc:
                ax_acc_ep.plot(
                    epochs_admm, admm_acc, label="ADMM", color="#ff7f0e", linewidth=2
                )

            ax_acc_ep.set_title("Accuracy vs Epochs")
            ax_acc_ep.set_xlabel("Epochs")
            ax_acc_ep.set_ylabel("Accuracy (%)")
            ax_acc_ep.grid(True, linestyle="--", alpha=0.6)

            if gd_f1:
                ax_f1_ep.plot(
                    epochs_gd,
                    gd_f1,
                    label="Full-Batch Adam",
                    ls="--",
                    color="#1f77b4",
                    linewidth=2,
                )
            if mini_f1:
                ax_f1_ep.plot(
                    epochs_mini,
                    mini_f1,
                    label="Mini-Batch Adam",
                    ls="-.",
                    color="#2ca02c",
                    linewidth=2,
                )
            if sgd_f1:
                ax_f1_ep.plot(
                    epochs_sgd,
                    sgd_f1,
                    label="Full-Batch GD",
                    ls=":",
                    color="#d62728",
                    linewidth=2,
                )
            if mini_sgd_f1:
                ax_f1_ep.plot(
                    epochs_mini_sgd,
                    mini_sgd_f1,
                    label="Mini-Batch SGD",
                    ls=":",
                    color="#9467bd",
                    linewidth=2,
                )
            if admm_f1:
                ax_f1_ep.plot(
                    epochs_admm, admm_f1, label="ADMM", color="#ff7f0e", linewidth=2
                )

            ax_f1_ep.set_title("F1 Score vs Epochs")
            ax_f1_ep.set_xlabel("Epochs")
            ax_f1_ep.set_ylabel("F1 Score")
            ax_f1_ep.grid(True, linestyle="--", alpha=0.6)

            # ---------------------------------------------------------
            # ROW 2: METRICS vs TIME
            # ---------------------------------------------------------
            if gd_acc and gd_time:
                t, y = match_lengths(gd_time, gd_acc)
                ax_acc_t.plot(
                    t, y, label="Full-Batch Adam", ls="--", color="#1f77b4", linewidth=2
                )
            if mini_acc and mini_time:
                t, y = match_lengths(mini_time, mini_acc)
                ax_acc_t.plot(
                    t, y, label="Mini-Batch Adam", ls="-.", color="#2ca02c", linewidth=2
                )
            if sgd_acc and sgd_time:
                t, y = match_lengths(sgd_time, sgd_acc)
                ax_acc_t.plot(
                    t, y, label="Full-Batch GD", ls=":", color="#d62728", linewidth=2
                )
            if mini_sgd_acc and mini_sgd_time:
                t, y = match_lengths(mini_sgd_time, mini_sgd_acc)
                ax_acc_t.plot(
                    t, y, label="Mini-Batch SGD", ls=":", color="#9467bd", linewidth=2
                )
            if admm_acc and admm_time:
                t, y = match_lengths(admm_time, admm_acc)
                ax_acc_t.plot(t, y, label="ADMM", color="#ff7f0e", linewidth=2)

            ax_acc_t.set_title("Accuracy vs Time")
            ax_acc_t.set_xlabel("Time (seconds)")
            ax_acc_t.set_ylabel("Accuracy (%)")
            ax_acc_t.grid(True, linestyle="--", alpha=0.6)

            if gd_f1 and gd_time:
                t, y = match_lengths(gd_time, gd_f1)
                ax_f1_t.plot(
                    t, y, label="Full-Batch Adam", ls="--", color="#1f77b4", linewidth=2
                )
            if mini_f1 and mini_time:
                t, y = match_lengths(mini_time, mini_f1)
                ax_f1_t.plot(
                    t, y, label="Mini-Batch Adam", ls="-.", color="#2ca02c", linewidth=2
                )
            if sgd_f1 and sgd_time:
                t, y = match_lengths(sgd_time, sgd_f1)
                ax_f1_t.plot(
                    t, y, label="Full-Batch GD", ls=":", color="#d62728", linewidth=2
                )
            if mini_sgd_f1 and mini_sgd_time:
                t, y = match_lengths(mini_sgd_time, mini_sgd_f1)
                ax_f1_t.plot(
                    t, y, label="Mini-Batch SGD", ls=":", color="#9467bd", linewidth=2
                )
            if admm_f1 and admm_time:
                t, y = match_lengths(admm_time, admm_f1)
                ax_f1_t.plot(t, y, label="ADMM", color="#ff7f0e", linewidth=2)

            ax_f1_t.set_title("F1 Score vs Time")
            ax_f1_t.set_xlabel("Time (seconds)")
            ax_f1_t.set_ylabel("F1 Score")
            ax_f1_t.grid(True, linestyle="--", alpha=0.6)

            # ---------------------------------------------------------
            # ZOOM BOUNDS APPLICATION
            # ---------------------------------------------------------
            if is_zoom:
                for ax in [ax_acc_ep, ax_f1_ep, ax_acc_t, ax_f1_t]:
                    ax.set_xlim(-5, 205)

                # Accuracy is in % (e.g., 60 to 100), plus padding (58 to 102)
                for ax in [ax_acc_ep, ax_acc_t]:
                    ax.set_ylim(58, 102)

                # F1 score is 0 to 1 (e.g., 0.6 to 1.0), plus padding (0.58 to 1.02)
                for ax in [ax_f1_ep, ax_f1_t]:
                    ax.set_ylim(0.58, 1.02)

            # ---------------------------------------------------------
            # FIGURE-LEVEL LEGEND
            # ---------------------------------------------------------
            handles, labels = ax_acc_ep.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=5,
                    frameon=False,
                )

            # Adjust layout and save figure
            fig.tight_layout()
            save_path = os.path.join(base_dir, model_name, save_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

            plot_type = "zoomed" if is_zoom else "main"
            print(f"Saved {plot_type} plot for {model_name} to: {save_path}")

            plt.close(fig)

        # =========================================================
        # FIGURE 3: FIRING RATE (Only for Spiking Models)
        # =========================================================
        if is_spiking:
            fig_fr, ax_fr = plt.subplots(figsize=(8, 5))

            if gd_fr and not all(np.isnan(x) for x in gd_fr):
                ax_fr.plot(
                    epochs_gd,
                    gd_fr,
                    label="Full-Batch Adam",
                    ls="--",
                    color="#1f77b4",
                    linewidth=2,
                )
            if mini_fr and not all(np.isnan(x) for x in mini_fr):
                ax_fr.plot(
                    epochs_mini,
                    mini_fr,
                    label="Mini-Batch Adam",
                    ls="-.",
                    color="#2ca02c",
                    linewidth=2,
                )
            if sgd_fr and not all(np.isnan(x) for x in sgd_fr):
                ax_fr.plot(
                    epochs_sgd,
                    sgd_fr,
                    label="Full-Batch GD",
                    ls=":",
                    color="#d62728",
                    linewidth=2,
                )
            if mini_sgd_fr and not all(np.isnan(x) for x in mini_sgd_fr):
                ax_fr.plot(
                    epochs_mini_sgd,
                    mini_sgd_fr,
                    label="Mini-Batch SGD",
                    ls=":",
                    color="#9467bd",
                    linewidth=2,
                )
            if admm_fr and not all(np.isnan(x) for x in admm_fr if x is not None):
                ax_fr.plot(
                    epochs_admm, admm_fr, label="ADMM", color="#ff7f0e", linewidth=2
                )

            ax_fr.set_title("Mean Firing Rate vs Epochs")
            ax_fr.set_xlabel("Epochs")
            ax_fr.set_ylabel("Mean Firing Rate")
            ax_fr.grid(True, linestyle="--", alpha=0.6)

            if (
                (gd_fr and not all(np.isnan(x) for x in gd_fr))
                or (admm_fr and not all(np.isnan(x) for x in admm_fr if x is not None))
                or (mini_fr and not all(np.isnan(x) for x in mini_fr))
                or (sgd_fr and not all(np.isnan(x) for x in sgd_fr))
                or (mini_sgd_fr and not all(np.isnan(x) for x in mini_sgd_fr))
            ):
                ax_fr.legend()

            fig_fr.tight_layout()

            save_path_fr = os.path.join(
                base_dir, model_name, f"{model_name}_firing_rate.png"
            )
            fig_fr.savefig(save_path_fr, bbox_inches="tight", dpi=300)
            print(f"Saved firing rate plot for {model_name} to: {save_path_fr}")

            plt.close(fig_fr)


if __name__ == "__main__":
    main()
