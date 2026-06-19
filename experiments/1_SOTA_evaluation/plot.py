import argparse
import json

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.labelsize": 8,  # X-axis number size (default is usually 10.0)
        "ytick.labelsize": 8,  # Y-axis number size
        "axes.titlesize": 12,  # Title size (keeps titles distinct if you shrink ticks)
        "legend.fontsize": 12,  # Legend font size
    }
)


def plot_grid_comparison(sota_path, new_path):
    # Load JSON data
    with open(sota_path, "r") as f:
        sota_metrics = json.load(f)
    with open(new_path, "r") as f:
        new_metrics = json.load(f)

    # Reverted back to the standard 2x3 grid
    fig, ax = plt.subplots(2, 3, figsize=(30, 10))

    # [0, 0] - Lagrangian (semilogy)
    ax[0, 0].semilogy(
        sota_metrics.get("lagrangian", []), label="SOTA", color="blue", linestyle="-"
    )
    ax[0, 0].semilogy(
        new_metrics.get("lagrangian", []),
        label="Modular framework",
        color="orange",
        linestyle="--",
    )
    ax[0, 0].set_title("Lagrangian")
    ax[0, 0].legend()
    ax[0, 0].grid(True, linestyle=":", alpha=0.7)

    # [0, 1] - Firing Rate (linear plot, replaces Primal Residual Norm)
    ax[0, 1].plot(
        sota_metrics.get("firing_rate", []),
        color="blue",
        linestyle="-",
        alpha=0.7,
    )
    ax[0, 1].plot(
        new_metrics.get("firing_rate", []),
        color="orange",
        linestyle="--",
        alpha=0.7,
    )

    # Proxy artists to keep the legend clean (since firing rates might be multi-dimensional)
    ax[0, 1].plot([], [], color="blue", label="SOTA")
    ax[0, 1].plot([], [], color="orange", linestyle="--", label="Modular framework")
    ax[0, 1].set_title("Firing Rate")
    ax[0, 1].legend()
    ax[0, 1].grid(True, linestyle=":", alpha=0.7)

    # [0, 2] - Preactivation Constraint (semilogy)
    ax[0, 2].semilogy(
        sota_metrics.get("preactivation_constraint_sum", []),
        color="blue",
        linestyle="-",
        alpha=0.7,
    )
    ax[0, 2].semilogy(
        new_metrics.get("preactivation_constraint_sum", []),
        color="orange",
        linestyle="--",
        alpha=0.7,
    )
    ax[0, 2].plot([], [], color="blue", label="SOTA")
    ax[0, 2].plot([], [], color="orange", linestyle="--", label="Modular framework")
    ax[0, 2].set_title("Preactivation Constraint (||z - F(a)||^2)")
    ax[0, 2].legend()
    ax[0, 2].grid(True, linestyle=":", alpha=0.7)

    # [1, 0] - Activation Constraint (semilogy)
    ax[1, 0].semilogy(
        sota_metrics.get("activation_constraint_sum", []),
        color="blue",
        linestyle="-",
        alpha=0.7,
    )
    ax[1, 0].semilogy(
        new_metrics.get("activation_constraint_sum", []),
        color="orange",
        linestyle="--",
        alpha=0.7,
    )
    ax[1, 0].plot([], [], color="blue", label="SOTA")
    ax[1, 0].plot([], [], color="orange", linestyle="--", label="Modular framework")
    ax[1, 0].set_title("Activation Constraint (||a - h(z)||^2)")
    ax[1, 0].legend()
    ax[1, 0].grid(True, linestyle=":", alpha=0.7)

    # [1, 1] - Loss (semilogy)
    # ax[1, 1].semilogy(
    #     sota_metrics.get("loss", []), label="SOTA", color="blue", linestyle="-"
    # )
    # ax[1, 1].semilogy(
    #     new_metrics.get("loss", []), label="NEW", color="orange", linestyle="--"
    # )

    # ax[1, 1].set_title("Loss")
    # ax[1, 1].legend()
    # ax[1, 1].grid(True, linestyle=":", alpha=0.7)

    # [1, 1] - Primal Residual Norm (semilogy)
    # ax[1, 1].semilogy(
    #     sota_metrics.get("primal_residual_norm", []),
    #     label="SOTA",
    #     color="blue",
    #     linestyle="-",
    # )
    # ax[1, 1].semilogy(
    #     new_metrics.get("primal_residual_norm", []),
    #     label="NEW",
    #     color="orange",
    #     linestyle="--",
    # )

    # ax[1, 1].set_title("Primal Residual Norm")
    # ax[1, 1].legend()
    # ax[1, 1].grid(True, linestyle=":", alpha=0.7)

    # [1, 1] - F1 (semilogy)
    ax[1, 1].plot(
        sota_metrics.get("f1", []),
        label="SOTA",
        color="blue",
        linestyle="-",
    )
    ax[1, 1].plot(
        new_metrics.get("f1", []),
        label="Modular framework",
        color="orange",
        linestyle="--",
    )

    ax[1, 1].set_title("F1 Score")
    ax[1, 1].legend()
    ax[1, 1].grid(True, linestyle=":", alpha=0.7)

    # [1, 2] - Train Accuracy (linear plot)
    ax[1, 2].plot(
        sota_metrics.get("accuracy", []), label="SOTA", color="blue", linestyle="-"
    )
    ax[1, 2].plot(
        new_metrics.get("accuracy", []),
        label="Modular framework",
        color="orange",
        linestyle="--",
    )

    ax[1, 2].set_title("Train Accuracy")
    ax[1, 2].legend()
    ax[1, 2].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout(w_pad=5.0, h_pad=2.0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot SOTA vs Modular framework ADMM metrics in a 2x3 grid."
    )
    parser.add_argument(
        "--sota",
        type=str,
        default="experiments/1_SOTA_evaluation/results/SOTA/results.json",
        help="Path to the SOTA metrics.json file",
    )
    parser.add_argument(
        "--new",
        type=str,
        default="experiments/1_SOTA_evaluation/results/NEW/results.json",
        help="Path to the NEW metrics.json file",
    )

    args = parser.parse_args()

    plot_grid_comparison(args.sota, args.new)
