import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Academic formatting to match your thesis style
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
    # 1. SETUP PATHS & MODELS
    # ==========================================
    models = ["feedforward", "conv", "spiking-feedforward", "spiking-conv"]
    blocks = ["two-block", "multi-block"]
    base_dir = Path("experiments/0_Extra_experiments/results/Block_Methods")

    if not base_dir.exists():
        print(f"❌ Error: Could not find {base_dir}. Run the benchmark script first.")
        sys.exit(1)

    print("Generating 6-stat block divergence plots...\n")

    # ==========================================
    # 2. PLOTTING ENGINE
    # ==========================================
    for model_name in models:
        fig, ax = plt.subplots(2, 3, figsize=(30, 10))
        fig.suptitle(
            f"Block Method Divergence: {model_name.replace('-', ' ').title()}",
            fontweight="bold",
            fontsize=20,
        )

        colors = {"two-block": "blue", "multi-block": "red"}
        data_found = False

        for block in blocks:
            json_path = base_dir / model_name / block / "results.json"

            if not json_path.exists():
                print(f"  ⚠️ Warning: Missing data for {model_name} - {block}")
                continue

            with open(json_path, "r") as f:
                metrics = json.load(f)

            # Skip if there's no real data inside
            if "accuracy" not in metrics or not metrics["accuracy"]:
                continue

            data_found = True
            c = colors[block]

            # [0, 0] - Lagrangian
            ax[0, 0].semilogy(
                metrics.get("lagrangian", []), label=block, color=c, linewidth=2
            )

            # [0, 1] - Firing Rate
            ax[0, 1].plot(metrics.get("firing_rate", []), color=c, alpha=0.7)
            ax[0, 1].plot([], [], color=c, label=block, linewidth=2)

            # [0, 2] - Preactivation Constraint
            ax[0, 2].semilogy(
                metrics.get("preactivation_constraint_sum", []), color=c, alpha=0.5
            )
            ax[0, 2].plot([], [], color=c, label=block, linewidth=2)

            # [1, 0] - Activation Constraint
            ax[1, 0].semilogy(
                metrics.get("activation_constraint_sum", []), color=c, alpha=0.5
            )
            ax[1, 0].plot([], [], color=c, label=block, linewidth=2)

            # [1, 1] - Loss
            ax[1, 1].semilogy(
                metrics.get("loss", []), label=block, color=c, linewidth=2
            )

            # [1, 2] - Train Accuracy
            ax[1, 2].plot(
                metrics.get("accuracy", []), label=block, color=c, linewidth=2
            )

        if data_found:
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

            plt.tight_layout(w_pad=5.0, h_pad=2.0, rect=[0.01, 0, 1, 1])

            # Save the figure locally within the results folder
            save_path = base_dir / f"{model_name}_block_divergence.png"
            plt.savefig(save_path)
            print(f"✅ Saved 6-stat plot for '{model_name}' to: {save_path}")
        else:
            print(f"⏭️  Skipped plotting for '{model_name}' due to missing data.")

        # Close the figure to free up memory before the next loop iteration
        plt.close(fig)

    print("\n🎉 All 6-stat plots generated successfully!")


if __name__ == "__main__":
    main()
