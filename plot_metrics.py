import json
import os
import argparse
import matplotlib.pyplot as plt

def load_metrics(filepath):
    """Loads the JSON metrics file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found at: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def get_filepaths_from_file(list_filepath):
    """Reads a text file and returns a list of file paths."""
    filepaths = []
    if not os.path.exists(list_filepath):
        raise FileNotFoundError(f"List file not found at: {list_filepath}")
        
    with open(list_filepath, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith('#'):
                filepaths.append(clean_line)
                
    if not filepaths:
        raise ValueError(f"No valid file paths found in {list_filepath}")
        
    return filepaths

def generate_labels(filepaths):
    """Finds differing directory parts, splits by '-', and keeps unique sub-parts."""
    if len(filepaths) <= 1:
        return [os.path.basename(os.path.dirname(filepaths[0]))] if filepaths else []

    # Split paths into individual directories/files
    split_paths = [p.replace('\\', '/').split('/') for p in filepaths]
    
    # Identify which folder/file levels are different across the paths
    max_len = max(len(p) for p in split_paths)
    diff_dir_indices = []
    for i in range(max_len):
        parts_at_i = [p[i] if i < len(p) else None for p in split_paths]
        if len(set(parts_at_i)) > 1:
            diff_dir_indices.append(i)
            
    labels = []
    for p in split_paths:
        diff_parts = []
        for i in diff_dir_indices:
            if i < len(p):
                dir_part = p[i]
                # Split the differing directory by '-'
                sub_parts = dir_part.split('-')
                
                unique_sub_parts = []
                for sub_i, sub_part in enumerate(sub_parts):
                    is_different = False
                    for other_p in split_paths:
                        if i < len(other_p):
                            other_sub_parts = other_p[i].split('-')
                            # If the other path lacks this sub-part or it's different
                            if sub_i >= len(other_sub_parts) or other_sub_parts[sub_i] != sub_part:
                                is_different = True
                                break
                        else:
                            is_different = True
                            break
                            
                    if is_different:
                        unique_sub_parts.append(sub_part)
                
                # Rejoin only the unique hyphenated parts
                if unique_sub_parts:
                    diff_parts.append("-".join(unique_sub_parts))
                else:
                    diff_parts.append(dir_part)
                    
        if not diff_parts:
            labels.append(p[-2] if len(p) > 1 else p[-1])
        else:
            labels.append("/".join(diff_parts))
            
    return labels

def plot_metrics(filepaths, target_metrics=None, save_path=None):
    """Generates plots comparing the requested metrics across multiple files."""
    # Expanded to 3x3 to fit all 9 plots
    fig, ax = plt.subplots(3, 3, figsize=(30, 16))
    
    labels = generate_labels(filepaths)
    running_times = []
    final_accs = []
    max_accs = []
    
    for filepath, label in zip(filepaths, labels):
        metrics = load_metrics(filepath)
        
        # Tracking metrics for the custom plots
        time = metrics.get("running_time", 0.0)
        running_times.append(time)
        
        acc_list = metrics.get("accuracy_list", [])
        final_accs.append(acc_list[-1] if acc_list else 0.0)
        max_accs.append(max(acc_list) if acc_list else 0.0)
        
        ax[0, 0].semilogy(metrics["lagrangians"], label=label)
        ax[0, 1].semilogy(metrics["lambdas"], label=label)
        ax[0, 2].semilogy(metrics["soft_constraints"]["a"], label=label)
        ax[1, 0].semilogy(metrics["soft_constraints"]["z"], label=label)
        ax[1, 1].semilogy(metrics["losses"], label=label)
        ax[1, 2].plot(acc_list, label=label)

    # Standard Line Plots Formatting
    ax[0, 0].set_title("Lagrangian")
    ax[0, 0].legend()
    
    ax[0, 1].set_title("Primal Residual Norm")
    ax[0, 1].legend()
    
    ax[0, 2].set_title("Activation Constraint (||a - h(z)||)")
    ax[0, 2].legend()
    
    ax[1, 0].set_title("Preactivation Constraint") 
    ax[1, 0].legend()
    
    ax[1, 1].set_title("Loss")
    ax[1, 1].legend()
    
    ax[1, 2].set_title("Train Accuracy")
    ax[1, 2].legend()
    
    # --- Plot 7: Running Time (Bar) ---
    ax[2, 0].bar(labels, running_times, color='skyblue', edgecolor='black')
    ax[2, 0].set_title("Running Time")
    ax[2, 0].set_ylabel("Seconds")
    ax[2, 0].set_xticks(range(len(labels)))
    ax[2, 0].set_xticklabels(labels, rotation=25, ha="right")
    
    # --- Plot 8: Time vs Final Accuracy (Scatter) ---
    for time, acc, label in zip(running_times, final_accs, labels):
        ax[2, 1].scatter(time, acc, label=label, s=120)
    ax[2, 1].set_title("Running Time vs Final Accuracy")
    ax[2, 1].set_xlabel("Running Time (Seconds)")
    ax[2, 1].set_ylabel("Final Accuracy")
    ax[2, 1].legend()

    # --- Plot 9: Time vs Max Accuracy (Scatter) ---
    for time, acc, label in zip(running_times, max_accs, labels):
        ax[2, 2].scatter(time, acc, label=label, s=120)
    ax[2, 2].set_title("Running Time vs Max Accuracy")
    ax[2, 2].set_xlabel("Running Time (Seconds)")
    ax[2, 2].set_ylabel("Max Accuracy")
    ax[2, 2].legend()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare ADMM training metrics from a list file.")
    
    parser.add_argument(
        "list_file", 
        type=str, 
        nargs='?', 
        default="plot_runs.txt", 
        help="Path to a text file containing paths to metrics.json files (defaults to plot_runs.txt)"
    )
    
    args = parser.parse_args()
    paths_to_plot = get_filepaths_from_file(args.list_file)
    plot_metrics(paths_to_plot)