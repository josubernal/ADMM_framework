import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(filepath):
    """Loads the JSON metrics file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found at: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

 
def plot_metrics(filepath, target_metrics=None, save_path=None):
    """Generates plots for the requested metrics."""
    metrics = load_metrics(filepath)
    
    fig, ax = plt.subplots(2, 3, figsize=(30, 5))
    ax[0, 0].semilogy(metrics["lagrangians"])
    ax[0, 0].set_title("Lagrangian")
    ax[0, 1].semilogy(metrics["lambdas"])
    ax[0, 1].set_title("Primal Residual Norm")
    ax[0, 2].semilogy(metrics["soft_constraints"]["a"])
    ax[0, 2].set_title("Activation Constraint (||a - h(z)||)")
    ax[1, 0].semilogy(metrics["soft_constraints"]["z"])
    ax[1, 0].set_title("Preactivation Constraint") 
    ax[1, 1].semilogy(metrics["losses"])
    ax[1, 1].set_title("Loss")
    ax[1, 2].plot(metrics["accuracy_list"])
    ax[1, 2].set_title("Train Accuracy")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ADMM training metrics.")
    parser.add_argument("filepath", type=str, help="Path to the metrics.json file")
    args = parser.parse_args()
    plot_metrics(args.filepath)