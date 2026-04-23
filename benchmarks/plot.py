import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 1. Handle Command Line Arguments
    if len(sys.argv) < 2:
        print("Usage: python plot.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    # Adjusting folder path based on your structure
    folder_path = os.path.join("benchmarks", "results", experiment_name)
    
    if not os.path.exists(folder_path):
        # Fallback to current directory if benchmarks/results/ doesn't exist
        folder_path = experiment_name
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{experiment_name}' not found.")
            sys.exit(1)

    # ==========================================
    # PLOT 1: Accuracy (4 Models in One Figure)
    # ==========================================
    if "gd_comparation" in experiment_name:
        json_paths = sorted(list(Path(folder_path).rglob("results.json")))

        if not json_paths:
            print(f"Skipping Accuracy plots: No 'results.json' found in {folder_path}")
        else:
            num_models = len(json_paths)
            # Determine grid size (e.g., 2x2 for 4 models)
            cols = 2 if num_models > 1 else 1
            rows = (num_models + cols - 1) // cols
            
            fig_acc, axes_acc = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
            fig_acc.suptitle(f"Accuracy Comparison: GD vs ADMM", fontsize=16, fontweight='bold')
            axes_flat = axes_acc.flatten()

            for i, json_path in enumerate(json_paths):
                with open(json_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                ax = axes_flat[i]
                gd_accs = data.get('gd_accuracy', [])
                admm_accs = data.get('accuracy_list', [])
                
                # Generate steps if not present
                gd_steps = list(range(1, len(gd_accs) + 1))
                admm_steps = list(range(1, len(admm_accs) + 1))
                
                warming_stop = data.get('warming_stop', None) 
                model_name = data.get('model_name', 'Model')
                batch_size = data.get('batch_size', 'N/A')

                # Plotting
                ax.plot(gd_steps, gd_accs, label='Gradient Descent', color='#1f77b4', linewidth=2)
                ax.plot(admm_steps, admm_accs, label='ADMM', color='#ff7f0e', linewidth=2)

                if warming_stop is not None:
                    ax.axvline(x=warming_stop, color='red', linestyle='--', alpha=0.6, label=f'Warming Stop ({warming_stop})')

                ax.set_title(f"Model: {model_name} (Batch: {batch_size})", fontsize=12)
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy (%)')
                ax.legend(fontsize='small')
                ax.grid(True, linestyle='--', alpha=0.5)

            # Remove any empty subplots if num_models is odd
            for j in range(i + 1, len(axes_flat)):
                fig_acc.delaxes(axes_flat[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ==========================================
    # PLOT 2: Memory Benchmark (3 metrics in one)
    # ==========================================
    if "memory_benchmark" in experiment_name:
        memory_csv_path = os.path.join(folder_path, "benchmark_memory_metrics.csv")
        
        if os.path.exists(memory_csv_path):
            df = pd.read_csv(memory_csv_path)
            plt.style.use('ggplot')
            
            metrics = ['Init Memory (MB)', 'Peak 1 Epoch (MB)', 'Peak 10 Epochs (MB)']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            markers = ['o', 's', '^', 'D', 'v']

            if 'Architecture' in df.columns and 'Batch Size' in df.columns:
                fig_mem, axes_mem = plt.subplots(1, 3, figsize=(18, 6))
                fig_mem.suptitle(f"Memory Usage vs Batch Size", fontsize=16, fontweight='bold')
                
                architectures = df['Architecture'].unique()
                
                for ax_idx, metric in enumerate(metrics):
                    ax = axes_mem[ax_idx]
                    for i, arch in enumerate(architectures):
                        arch_data = df[df['Architecture'] == arch].sort_values('Batch Size')
                        ax.plot(arch_data['Batch Size'], arch_data[metric], 
                                marker=markers[i % len(markers)], 
                                color=colors[i % len(colors)], 
                                label=arch, linewidth=2)
                    
                    ax.set_title(metric)
                    ax.set_xlabel("Batch Size")
                    ax.set_ylabel("VRAM (MB)")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            print(f"Skipping Memory plot: {memory_csv_path} not found.")

        
    
    if "iteration_methods" in experiment_name:
        all_jsons = list(Path(folder_path).rglob("*.json"))
        iter_data = []
        for jp in all_jsons:
            try:
                with open(jp, 'r') as f:
                    d = json.load(f)
                    # Only keep JSONs that have the specific keys from iteration_methods.py
                    if "method" in d and "architecture" in d and "accuracy_list" in d:
                        iter_data.append(d)
            except:
                continue
        if iter_data:
            # Find unique architectures (should be 'linear' and 'conv')
            archs = list(set([d["architecture"] for d in iter_data]))
            archs.sort(reverse=True) # Usually puts 'linear' first, then 'conv'
            
            cols = len(archs)
            if cols > 0:
                fig_iter, axes_iter = plt.subplots(1, cols, figsize=(7 * cols, 6), squeeze=False)
                fig_iter.suptitle("Iteration Methods Comparison: Accuracy over Epochs", fontsize=16, fontweight='bold')
                
                for i, arch in enumerate(archs):
                    ax = axes_iter[0, i]
                    # Get all runs for this specific architecture
                    arch_specific_data = [d for d in iter_data if d["architecture"] == arch]
                    
                    # Sort alphabetically by method so colors/legend stay consistent
                    arch_specific_data.sort(key=lambda x: x["method"])
                    
                    for d in arch_specific_data:
                        acc = d["accuracy_list"]
                        method_name = d["method"]
                        epochs_list = list(range(len(acc)))
                        
                        # Plot the accuracy curve for this method
                        ax.plot(epochs_list, acc, label=method_name, linewidth=2, marker='o', markersize=4)
                        
                    ax.set_title(f"Architecture: {arch.upper()}", fontsize=14)
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Accuracy (%)")
                    ax.legend(fontsize='small', loc='lower right')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            print(f"Skipping Iteration Methods plot: No compatible JSONs found in {folder_path}")
            
    
    # ==========================================
    # PLOT 4: Depth Benchmark (Number of Layers)
    # ==========================================
    if "static_breakpoint" in experiment_name:
        all_jsons = list(Path(folder_path).rglob("*.json"))
        static_data = []
        for jp in all_jsons:
            try:
                with open(jp, 'r') as f:
                    d = json.load(f)
                    # Check for the keys saved by your static breakpoint script
                    if "architecture" in d and "layers" in d and "accuracy_list" in d:
                        static_data.append(d)
            except:
                continue
                
        if static_data:
            # Find unique architectures (should be 'linear' and 'conv')
            archs = list(set([d["architecture"] for d in static_data]))
            archs.sort(reverse=True) # Usually puts 'linear' first, then 'conv'
            
            cols = len(archs)
            if cols > 0:
                fig_static, axes_static = plt.subplots(1, cols, figsize=(7 * cols, 6), squeeze=False)
                fig_static.suptitle("Depth Comparison: Accuracy vs. Number of Hidden Layers", fontsize=16, fontweight='bold')
                
                for i, arch in enumerate(archs):
                    ax = axes_static[0, i]
                    # Get all runs for this specific architecture
                    arch_specific_data = [d for d in static_data if d["architecture"] == arch]
                    
                    # Sort NUMERICALLY by layers so the legend goes 1, 2, 3, 4
                    arch_specific_data.sort(key=lambda x: int(x["layers"]))
                    
                    for d in arch_specific_data:
                        acc = d["accuracy_list"]
                        num_layers = d["layers"]
                        epochs_list = list(range(1, len(acc) + 1)) # Start at Epoch 1
                        
                        # Plot the accuracy curve for this layer count
                        ax.plot(epochs_list, acc, label=f"{num_layers} Hidden Layer(s)", linewidth=2, marker='o', markersize=4)
                        
                    ax.set_title(f"Architecture: {arch.upper()}", fontsize=14)
                    ax.set_xlabel("Epochs")
                    ax.set_ylabel("Accuracy (%)")
                    ax.legend(title="Network Depth", fontsize='small', loc='lower right')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            print(f"Skipping Depth plot: No compatible JSONs found in {folder_path}")

    plt.show()

if __name__ == "__main__":
    main()