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
                model_name = data.get('architecture', None)
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

    # ==========================================
    # PLOT 3: Iteration 
    # ==========================================
    
    if "iteration_methods" in experiment_name:
        all_jsons = list(Path(folder_path).rglob("*.json"))
        iter_data = []
        for jp in all_jsons:
            try:
                with open(jp, 'r') as f:
                    d = json.load(f)
                    if all(k in d for k in ["method", "architecture", "accuracy_list", "running_time"]):
                        iter_data.append(d)
            except:
                continue

        if iter_data:
            archs = sorted(list(set([d["architecture"] for d in iter_data])), reverse=True) # ['linear', 'conv']
            
            # Create a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"ADMM Iteration Methods Benchmark: {experiment_name.replace('_', ' ').upper()}", 
                         fontsize=20, fontweight='bold', y=0.98)

            # Define a consistent color map for methods across all plots
            all_methods = sorted(list(set([d["method"] for d in iter_data])))
            cmap = plt.get_cmap('tab10')
            method_colors = {method: cmap(i) for i, method in enumerate(all_methods)}

            for i, arch in enumerate(archs):
                # Filter data for this architecture
                arch_data = [d for d in iter_data if d["architecture"] == arch]
                arch_data.sort(key=lambda x: x["method"])

                # --- TOP ROW: Accuracy vs. Epochs ---
                ax_top = axes[0, i]
                for d in arch_data:
                    method = d["method"]
                    ax_top.plot(range(len(d["accuracy_list"])), d["accuracy_list"], 
                                label=method, color=method_colors[method], 
                                linewidth=2.5, marker='o', markersize=4, alpha=0.8)
                
                ax_top.set_title(f"Convergence: {arch.upper()}", fontsize=15, pad=10)
                ax_top.set_ylabel("Accuracy (%)", fontsize=10)
                ax_top.set_xlabel("Epochs", fontsize=10)
                ax_top.grid(True, linestyle='--', alpha=0.6)
                ax_top.legend(fontsize='small', loc='lower right', frameon=True)

                # --- BOTTOM ROW: Time vs. Max Accuracy ---
                ax_bottom = axes[1, i]
                for d in arch_data:
                    method = d["method"]
                    max_acc = max(d["accuracy_list"])
                    total_time = d["running_time"]
                    
                    # Plotting points individually to link them to the legend
                    ax_bottom.scatter(total_time, max_acc, color=method_colors[method], 
                                      s=150, label=method, edgecolors='black', zorder=3)
  
                ax_bottom.set_ylabel("Peak Accuracy (%)", fontsize=10)
                ax_bottom.set_xlabel("Total Training Time (seconds)", fontsize=10)
                ax_bottom.grid(True, linestyle=':', alpha=0.6)
                
                # Add legend only if there's data
                if arch_data:
                    ax_bottom.legend(fontsize='small', title="Methods", loc='best')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Optional: Save the figure automatically
            save_path = os.path.join(folder_path, "iteration_methods_comparison.png")
            plt.savefig(save_path, dpi=300)
            print(f"Combined plot saved to: {save_path}")
    
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

    # ==========================================
    # PLOT: Loss Function Comparison (2x2 Grid)
    # ==========================================
    if "loss_function" in experiment_name:
        all_jsons = list(Path(folder_path).rglob("*.json"))
        loss_data = []
        for jp in all_jsons:
            try:
                with open(jp, 'r') as f:
                    d = json.load(f)
                    if "architecture" in d and "accuracy_list" in d:
                        # Extract loss function name (fallback to folder name if key is missing)
                        loss_name = d.get("loss_function", jp.parent.name)
                        d["loss_function"] = loss_name
                        loss_data.append(d)
            except Exception as e:
                continue

        if loss_data:
            # Standardize architecture order for a nice 2x2 layout
            archs = list(set([d["architecture"] for d in loss_data]))
            arch_order = {"linear": 0, "conv": 1, "spiking-linear": 2, "spiking-conv": 3}
            archs.sort(key=lambda x: arch_order.get(x, 10))
            
            # Setup Grid (e.g., 2x2 for 4 models)
            cols = 2 if len(archs) > 1 else 1
            rows = (len(archs) + cols - 1) // cols
            
            fig_loss, axes_loss = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
            fig_loss.suptitle("Loss Function Comparison: ADMM Convergence", fontsize=16, fontweight='bold')
            axes_flat = axes_loss.flatten()

            # Define a consistent color map for the loss functions
            unique_losses = list(set([str(d["loss_function"]) for d in loss_data]))
            cmap = plt.get_cmap('Set1')
            loss_colors = {loss: cmap(i) for i, loss in enumerate(unique_losses)}

            for i, arch in enumerate(archs):
                ax = axes_flat[i]
                arch_data = [d for d in loss_data if d["architecture"] == arch]
                
                # Sort alphabetically by loss so the legend is consistent
                arch_data.sort(key=lambda x: str(x["loss_function"]))
                
                for d in arch_data:
                    acc = d["accuracy_list"]
                    loss_name = str(d["loss_function"])
                    epochs_list = list(range(len(acc)))
                    
                    ax.plot(epochs_list, acc, label=loss_name, color=loss_colors[loss_name], 
                            linewidth=2.5, marker='o', markersize=5, alpha=0.8)
                
                # Formatting the subplot
                ax.set_title(f"Architecture: {arch.upper()}", fontsize=14)
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Only add legend if data exists for this subplot
                if arch_data:
                    ax.legend(title="Loss Objective", fontsize='small', loc='lower right')

            # Clean up any empty subplots if you test < 4 architectures
            for j in range(len(archs), len(axes_flat)):
                fig_loss.delaxes(axes_flat[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Optional: Save it
            save_path = os.path.join(folder_path, "loss_function_comparison.png")
            plt.savefig(save_path, dpi=300)
            print(f"Loss comparison plot saved to: {save_path}")
            
        else:
            print(f"Skipping Loss Function plot: No compatible JSONs found in {folder_path}")
    # ==========================================
    # PLOT 5: Initialization Comparison (2x2 Grid)
    # ==========================================
    if "initialization_parameters" in experiment_name:
        all_jsons = list(Path(folder_path).rglob("*.json"))
        init_data = []
        for jp in all_jsons:
            try:
                with open(jp, 'r') as f:
                    d = json.load(f)
                    if "architecture" in d and "accuracy_list" in d:
                        # Extract initialization name (fallback to folder name if key is missing)
                        init_name = d.get("initialization", jp.parent.name)
                        d["initialization"] = init_name
                        init_data.append(d)
            except Exception as e:
                continue

        if init_data:
            # Standardize architecture order for a nice 2x2 layout
            archs = list(set([d["architecture"] for d in init_data]))
            arch_order = {"linear": 0, "conv": 1, "spiking-linear": 2, "spiking-conv": 3}
            archs.sort(key=lambda x: arch_order.get(x, 10))
            
            # Setup Grid (e.g., 2x2 for 4 models)
            cols = 2 if len(archs) > 1 else 1
            rows = (len(archs) + cols - 1) // cols
            
            fig_init, axes_init = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
            fig_init.suptitle("Weight Initialization Comparison: ADMM Convergence", fontsize=16, fontweight='bold')
            axes_flat = axes_init.flatten()

            # Define a consistent color map for the initializations (tab10 is great for distinct categories)
            unique_inits = list(set([str(d["initialization"]) for d in init_data]))
            cmap = plt.get_cmap('tab10')
            init_colors = {init: cmap(i % 10) for i, init in enumerate(unique_inits)}

            for i, arch in enumerate(archs):
                ax = axes_flat[i]
                arch_data = [d for d in init_data if d["architecture"] == arch]
                
                # Sort alphabetically by initialization so the legend is consistent
                arch_data.sort(key=lambda x: str(x["initialization"]))
                
                for d in arch_data:
                    acc = d["accuracy_list"]
                    init_name = str(d["initialization"])
                    epochs_list = list(range(len(acc)))
                    
                    ax.plot(epochs_list, acc, label=init_name, color=init_colors[init_name], 
                            linewidth=2.5, marker='o', markersize=4, alpha=0.8)
                
                # Formatting the subplot
                ax.set_title(f"Architecture: {arch.upper()}", fontsize=14)
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Only add legend if data exists for this subplot
                if arch_data:
                    ax.legend(title="Initialization Method", fontsize='small', loc='lower right')

            # Clean up any empty subplots if you test < 4 architectures
            for j in range(len(archs), len(axes_flat)):
                fig_init.delaxes(axes_flat[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure automatically
            save_path = os.path.join(folder_path, "initialization_comparison.png")
            plt.savefig(save_path, dpi=300)
            print(f"Initialization comparison plot saved to: {save_path}")
            
        else:
            print(f"Skipping Initialization plot: No compatible JSONs found in {folder_path}")
    
    if "initialization_states" in experiment_name:
        all_jsons = list(Path(folder_path).rglob("*.json"))
        init_data = []
        for jp in all_jsons:
            try:
                with open(jp, 'r') as f:
                    d = json.load(f)
                    if "architecture" in d and "accuracy_list" in d:
                        # Extract initialization name (fallback to folder name if key is missing)
                        init_name = d.get("initialization", jp.parent.name)
                        d["initialization"] = init_name
                        init_data.append(d)
            except Exception as e:
                continue

        if init_data:
            # Standardize architecture order for a nice 2x2 layout
            archs = list(set([d["architecture"] for d in init_data]))
            arch_order = {"linear": 0, "conv": 1, "spiking-linear": 2, "spiking-conv": 3}
            archs.sort(key=lambda x: arch_order.get(x, 10))
            
            # Setup Grid (e.g., 2x2 for 4 models)
            cols = 2 if len(archs) > 1 else 1
            rows = (len(archs) + cols - 1) // cols
            
            fig_init, axes_init = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)
            fig_init.suptitle("Weight Initialization Comparison: ADMM Convergence", fontsize=16, fontweight='bold')
            axes_flat = axes_init.flatten()

            # Define a consistent color map for the initializations (tab10 is great for distinct categories)
            unique_inits = list(set([str(d["initialization"]) for d in init_data]))
            cmap = plt.get_cmap('tab10')
            init_colors = {init: cmap(i % 10) for i, init in enumerate(unique_inits)}

            for i, arch in enumerate(archs):
                ax = axes_flat[i]
                arch_data = [d for d in init_data if d["architecture"] == arch]
                
                # Sort alphabetically by initialization so the legend is consistent
                arch_data.sort(key=lambda x: str(x["initialization"]))
                
                for d in arch_data:
                    acc = d["accuracy_list"]
                    init_name = str(d["initialization"])
                    epochs_list = list(range(len(acc)))
                    
                    ax.plot(epochs_list, acc, label=init_name, color=init_colors[init_name], 
                            linewidth=2.5, marker='o', markersize=4, alpha=0.8)
                
                # Formatting the subplot
                ax.set_title(f"Architecture: {arch.upper()}", fontsize=14)
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # Only add legend if data exists for this subplot
                if arch_data:
                    ax.legend(title="Initialization Method", fontsize='small', loc='lower right')

            # Clean up any empty subplots if you test < 4 architectures
            for j in range(len(archs), len(axes_flat)):
                fig_init.delaxes(axes_flat[j])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure automatically
            save_path = os.path.join(folder_path, "initialization_comparison.png")
            plt.savefig(save_path, dpi=300)
            print(f"Initialization comparison plot saved to: {save_path}")
            
        else:
            print(f"Skipping Initialization plot: No compatible JSONs found in {folder_path}")
            
    plt.show()

if __name__ == "__main__":
    main()