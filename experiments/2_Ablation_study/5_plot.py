import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
    }
)


def main():
    # ==========================================
    # 1. SETUP: CHANGE THIS NAME FOR OTHER FILES
    # ==========================================
    experiment_name = "loss_functions"

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

    print(f"\nGenerating plots for: {experiment_name}")

    # ==========================================
    # 3. PLOTTING ENGINE: FIGURE 1 (METRICS)
    # ==========================================
    fig1, ax = plt.subplots(2, 3, figsize=(30, 10))
    fig1.suptitle(
        f"Ablation Study: {experiment_name.replace('_', ' ').title()}",
        fontsize=20,
        fontweight="bold",
    )

    colors = plt.cm.tab10.colors

    # Storage for the separate bar chart
    methods_list = []
    runtimes = []
    bar_colors = []

    for i, (method, metrics) in enumerate(metrics_data.items()):
        c = colors[i % len(colors)]
        methods_list.append(method)
        bar_colors.append(c)

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

        # [1, 1] - F1
        ax[1, 1].plot(metrics.get("f1", []), label=method, color=c, linewidth=2)

        # [1, 2] - Train Accuracy
        ax[1, 2].plot(metrics.get("accuracy", []), label=method, color=c, linewidth=2)

        # Extract runtime robustly
        rt = metrics.get(
            "running_time", metrics.get("time", metrics.get("admm_time", 0))
        )
        if isinstance(rt, list):
            rt = rt[-1] if len(rt) > 0 else 0
        runtimes.append(rt)

    # Clean up formatting for the 2x3 grid
    titles = [
        "Lagrangian",
        "Firing Rate",
        "Preactivation Constraint (||z - F(a)||^2)",
        "Activation Constraint (||a - h(z)||^)",
        "Train F1 Score",
        "Train Accuracy",
    ]

    axes_flat = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]

    for j, axis in enumerate(axes_flat):
        axis.set_title(titles[j], fontsize=14)
        axis.set_xlabel("Epochs")
        axis.legend()
        axis.grid(True, linestyle=":", alpha=0.7)

    fig1.tight_layout(w_pad=5.0, h_pad=2.0)

    # Save Figure 1
    save_path_metrics = Path(
        f"experiments/2_Ablation_study/results/{experiment_name}_comparison.png"
    )
    fig1.savefig(save_path_metrics)
    print(f"Saved metrics plot to: {save_path_metrics}")

    # ==========================================
    # 4. PLOTTING ENGINE: FIGURE 2 (RUNTIME)
    # ==========================================
    # fig2, ax_bar = plt.subplots(figsize=(10, 6))

    # # Plot the bar chart
    # ax_bar.bar(methods_list, runtimes, color=bar_colors, alpha=0.8)

    # # Formatting
    # ax_bar.set_title(
    #     f"Total Run Time: {experiment_name.replace('_', ' ').title()}", fontsize=16
    # )
    # ax_bar.set_ylabel("Seconds", fontsize=12)
    # ax_bar.tick_params(axis="x", rotation=15, labelsize=10)
    # ax_bar.grid(axis="y", linestyle=":", alpha=0.7)

    # fig2.tight_layout()

    # # Save Figure 2
    # save_path_runtime = Path(
    #     f"experiments/2_Ablation_study/results/{experiment_name}_runtime.png"
    # )
    # fig2.savefig(save_path_runtime)
    # print(f"Saved runtime plot to: {save_path_runtime}")

    # ==========================================
    # 5. DISPLAY
    # ==========================================
    # Calling plt.show() will now open both figures in separate windows simultaneously.
    plt.show()


if __name__ == "__main__":
    main()
