import subprocess
from pathlib import Path


def run_scripts_in_order():
    folder_path = Path(__file__).resolve().parent

    project_root = folder_path.parent.parent

    files = sorted(folder_path.glob("*.py"))

    print(f"Found {len(files)} Python scripts to run in '{folder_path}':")

    for file in files:
        if file.name == Path(__file__).name:
            continue
        if file.name.startswith("_"):
            continue
        if file.name.endswith("plot.py"):
            continue

        print(f"\n{'=' * 40}")
        print(f"🚀 Running: {file.name}")
        print(f"{'=' * 40}")

        rel_path = file.relative_to(project_root).with_suffix("")
        module_name = ".".join(rel_path.parts)

        try:
            subprocess.run(
                ["uv", "run", "python", "-m", module_name], cwd=project_root, check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error: {file.name} failed with exit code {e.returncode}.")
            print("Stopping execution.")
            break


if __name__ == "__main__":
    run_scripts_in_order()
