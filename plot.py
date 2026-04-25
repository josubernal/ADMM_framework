import argparse
from admm import (
    ADMM_Metrics
)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ADMM training metrics.")
    parser.add_argument("filepath", type=str, help="Path to the metrics.json file")
    args = parser.parse_args()
    metrics = ADMM_Metrics()
    metrics.load(args.filepath)
    metrics.plot()