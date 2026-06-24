import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Apply academic formatting for the plot
plt.rcParams.update(
    {
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def main():
    # ==========================================
    # 1. SETUP PATHS
    # ==========================================
    results_dir = Path("experiments/0_Extra_experiments/results")
    json_path = results_dir / "Cross_Entropy.json"

    if not json_path.exists():
        print(f"❌ Error: Could not find {json_path}. Run the benchmark script first.")
        sys.exit(1)

    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    with open(json_path, "r") as f:
        data = json.load(f)

    class_sizes = data.get("class_sizes", [])
    times_exact = data.get("times_exact_ms", [])
    times_taylor = data.get("times_taylor_ms", [])

    if not class_sizes or not times_exact or not times_taylor:
        print("❌ Error: JSON file is missing required data keys.")
        sys.exit(1)

    print(f"Generating scalability plot from {json_path}...")

    # ==========================================
    # 3. PLOTTING ENGINE
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both lines with distinct markers and colors
    ax.plot(
        class_sizes,
        times_exact,
        marker="o",
        color="red",
        linewidth=2.5,
        label="Exact CE (Newton-Raphson)",
    )

    ax.plot(
        class_sizes,
        times_taylor,
        marker="s",
        color="blue",
        linewidth=2.5,
        label="Taylor Approx CE",
    )

    # Formatting
    ax.set_xlabel("Number of Classes ($C$)")
    ax.set_ylabel("Time per Output Layer Update (Milliseconds)")

    # Add a subtle grid
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left")

    plt.tight_layout()

    # ==========================================
    # 4. SAVE AND DISPLAY
    # ==========================================
    save_path = results_dir / "Cross_Entropy_scaling.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved plot to: {save_path}")

    # Show the interactive window
    plt.show()


if __name__ == "__main__":
    main()
