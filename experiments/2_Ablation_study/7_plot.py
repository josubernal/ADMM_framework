import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.labelsize": 8,  # X-axis number size (default is usually 10.0)
        "ytick.labelsize": 8,  # Y-axis number size
        "axes.titlesize": 12,  # Title size (keeps titles distinct if you shrink ticks)
        "legend.fontsize": 12,  # Legend font size
    }
)


def main():
    # ==========================================
    # 1. SETUP: CHANGE THIS NAME FOR OTHER FILES
    # ==========================================
    experiment_name = "lagrange"

    # ==========================================
    # 2. DATA LOADING
    # ==========================================
    category_dir = Path(f"experiments/2_Ablation_study/results/{experiment_name}")

    if not category_dir.exists():
        print(f"Directory {category_dir} not found. Run the training script first.")
        sys.exit(1)

    metrics_data = {}
    for method_dir in category_dir.iterdir():
        if method_dir.is_dir():
            json_path = method_dir / "results.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    metrics_data[method_dir.name] = json.load(f)

    if not metrics_data:
        print(f"No results found in {category_dir}")
        sys.exit(1)

    print(f"\nGenerating plot for: {experiment_name}")

    # ==========================================
    # 3. PLOTTING ENGINE (CUSTOMIZE HERE)
    # ==========================================
    fig, ax = plt.subplots(2, 3, figsize=(30, 10))

    colors = plt.cm.tab10.colors

    for i, (method, metrics) in enumerate(metrics_data.items()):
        c = colors[i % len(colors)]

        # [0, 0] - Lagrangian
        ax[0, 0].semilogy(
            metrics.get("lagrangian", []), label=method, color=c, linewidth=2
        )

        # [0, 1] - Firing Rate
        ax[0, 1].plot(metrics.get("firing_rate", []), color=c, alpha=0.7)
        ax[0, 1].plot([], [], color=c, label=method, linewidth=2)

        # [0, 2] - Preactivation Constraint
        ax[0, 2].semilogy(
            metrics.get("preactivation_constraint_sum", []), color=c, alpha=0.5
        )
        ax[0, 2].plot([], [], color=c, label=method, linewidth=2)

        # [1, 0] - Activation Constraint
        ax[1, 0].semilogy(
            metrics.get("activation_constraint_sum", []), color=c, alpha=0.5
        )
        ax[1, 0].plot([], [], color=c, label=method, linewidth=2)

        # [1, 1] - Loss
        ax[1, 1].semilogy(metrics.get("loss", []), label=method, color=c, linewidth=2)

        # [1, 2] - Train Accuracy
        ax[1, 2].plot(metrics.get("accuracy", []), label=method, color=c, linewidth=2)

    # Clean up formatting for all 6 subplots
    titles = [
        "Lagrangian",
        "Firing Rate",
        "Preactivation Constraint (||z - F(a)||^2)",
        "Activation Constraint (||a - h(z)||^)",
        "Loss",
        "Train Accuracy",
    ]

    axes_flat = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]

    for j, axis in enumerate(axes_flat):
        axis.set_title(titles[j], fontsize=14)
        axis.set_xlabel("Epochs")
        axis.legend()
        axis.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout(w_pad=5.0, h_pad=2.0, rect=[0.01, 0, 1, 1])

    # ==========================================
    # 4. SAVE AND DISPLAY
    # ==========================================
    save_path = Path(
        f"experiments/2_Ablation_study/results/{experiment_name}_comparison.png"
    )
    plt.savefig(save_path)
    print(f"Saved '{experiment_name}' plot to: {save_path}")

    # Show the interactive window
    plt.show()


if __name__ == "__main__":
    main()
