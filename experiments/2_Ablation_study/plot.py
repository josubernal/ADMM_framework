import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_ablation_category(experiment_name, method_json_paths):
    """
    Reads multiple JSON result files for a specific ablation experiment
    and plots them on a single 2x3 grid.
    """
    # Load all metrics for the current experiment
    metrics_data = {}
    for method_name, path in method_json_paths.items():
        with open(path, "r") as f:
            metrics_data[method_name] = json.load(f)

    fig, ax = plt.subplots(2, 3, figsize=(30, 10))
    fig.suptitle(
        f"Ablation Study: {experiment_name.replace('_', ' ').title()}",
        fontsize=20,
        fontweight="bold",
    )

    # Use a distinct colormap that easily scales up to 10 different methods
    colors = plt.cm.tab10.colors

    for i, (method, metrics) in enumerate(metrics_data.items()):
        c = colors[i % len(colors)]

        # [0, 0] - Lagrangian
        ax[0, 0].semilogy(
            metrics.get("lagrangian", []), label=method, color=c, linewidth=2
        )

        # [0, 1] - Firing Rate (Replaces Primal Residual Norm)
        # Plots all layer firing rates for this method with the same color
        ax[0, 1].plot(metrics.get("firing_rate", []), color=c, alpha=0.7)
        ax[0, 1].plot(
            [], [], color=c, label=method, linewidth=2
        )  # Proxy artist for legend

        # [0, 2] - Preactivation Constraint
        # Plots all layer residuals for this method with the same color
        ax[0, 2].semilogy(
            metrics.get("preactivation_constraint_sum", []), color=c, alpha=0.5
        )
        ax[0, 2].plot(
            [], [], color=c, label=method, linewidth=2
        )  # Proxy artist for legend

        # [1, 0] - Activation Constraint
        # Plots all layer residuals for this method with the same color
        ax[1, 0].semilogy(
            metrics.get("activation_constraint_sum", []), color=c, alpha=0.5
        )
        ax[1, 0].plot(
            [], [], color=c, label=method, linewidth=2
        )  # Proxy artist for legend

        # [1, 1] - Loss
        ax[1, 1].semilogy(metrics.get("loss", []), label=method, color=c, linewidth=2)

        # [1, 2] - Train Accuracy
        ax[1, 2].plot(metrics.get("accuracy", []), label=method, color=c, linewidth=2)

    # Clean up formatting for all 6 subplots
    titles = [
        "Lagrangian",
        "Firing Rate",
        "Preactivation Constraint (||z - F(a)||)",
        "Activation Constraint (||a - h(z)||)",
        "Loss",
        "Train Accuracy",
    ]

    axes_flat = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]

    for j, axis in enumerate(axes_flat):
        axis.set_title(titles[j], fontsize=14)
        axis.set_xlabel("Epochs")
        axis.legend()
        axis.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()

    # Save the figure alongside displaying it so you have a hard copy
    save_path = Path(
        f"experiments/2_Ablation_study/results/{experiment_name}_comparison.png"
    )
    plt.savefig(save_path)
    print(f"Saved '{experiment_name}' plot to: {save_path}")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Define the root results directory
    base_dir = Path("experiments/2_Ablation_study/results")

    if not base_dir.exists():
        print(
            f"Directory {base_dir} not found. Make sure you run this from the project root."
        )
    else:
        # 1. Iterate through each experiment category (e.g., 'block_methods', 'loss_functions')
        for category_dir in base_dir.iterdir():
            if category_dir.is_dir():
                method_paths = {}

                # 2. Inside each category, find the specific methods (e.g., 'two-block', 'multi-block')
                for method_dir in category_dir.iterdir():
                    if method_dir.is_dir():
                        json_path = method_dir / "results.json"

                        # 3. If a result file exists, map its method name to its file path
                        if json_path.exists():
                            method_paths[method_dir.name] = json_path

                # 4. If we found data, generate the 2x3 plot for this whole experiment
                if method_paths:
                    print(f"\nGenerating plot for: {category_dir.name}")
                    plot_ablation_category(category_dir.name, method_paths)
