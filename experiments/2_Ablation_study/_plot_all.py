import subprocess
import sys


def main():
    # List of all your independent plot scripts
    scripts = [
        # "experiments/2_Ablation_study/1_plot.py",
        # "experiments/2_Ablation_study/2_plot.py",
        # "experiments/2_Ablation_study/3_plot.py",
        "experiments/2_Ablation_study/4_plot.py",
        "experiments/2_Ablation_study/5_plot.py",
        "experiments/2_Ablation_study/6_plot.py",
        "experiments/2_Ablation_study/7_plot.py",
        "experiments/2_Ablation_study/8_plot.py",
    ]

    for script in scripts:
        print(f"\n{'=' * 50}")
        print(f"RUNNING: {script}")
        print(f"{'=' * 50}")

        try:
            # sys.executable ensures it uses your .venv Python path
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"❌ Error executing {script}. Process exited with code {e.returncode}."
            )
        except FileNotFoundError:
            print(f"❌ Could not find '{script}'. Ensure the filename matches exactly.")

    print("\n✅ All plotting scripts executed successfully!")


if __name__ == "__main__":
    main()
