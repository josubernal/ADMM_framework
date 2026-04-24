import torch
import configparser
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import itertools
import time
from utils.dataset import get_data
import torch.nn.functional as F

from admm import (
    ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpatialPool, ADMM_SpikingConv2d, 
    ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_Heaviside, ADMM_ReLU, ADMM_Metrics, ADMM_GAP
)

def is_valid_combination(params: dict) -> bool:    
    is_spiking = params.get('model').split("-")[0]=="spiking"
    print(is_spiking)
    train_method = params.get('train_method', 'vectorized')
    if not is_spiking and train_method != 'vectorized':
        return False    
    return True

def parse_value(v):
    """Smartly parse strings from the INI file into their correct Python types."""
    v = v.strip()
    if v.lower() == 'true': return True
    if v.lower() == 'false': return False
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v  

def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1

if __name__ == "__main__":
    # --- Base Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    
    # --- Smart Config Parsing ---
    grid_params = {}
    if config.has_section('config'):
        for key, val in config.items('config'):
            # Split by comma to support both static (length 1) and grid (length > 1)
            grid_params[key] = [parse_value(v) for v in val.split(',')]
    else:
        raise ValueError("Config file must contain a [config] section.")
            
    # Generate all combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combinations = [combo for combo in combinations if is_valid_combination(combo)]
    
    is_static_run = (len(combinations) == 1)
    
    print(f"Total configurations to evaluate: {len(combinations)}")
    if not is_static_run:
        print(f"Grid Parameters detected.")
    
    # To store metrics for plotting if it's a static run
    final_metrics = {}
    dynamic_keys = [k for k, v in grid_params.items() if len(v) > 1]
    # ==========================================
    # EXECUTION LOOP
    # ==========================================
    for idx, cfg in enumerate(combinations):
        print(f"\n=======================================================")
        if is_static_run:
            print(f"Running Static Config: {cfg['model']} | Train Method: {cfg['train_method']}")
        else:
            print(f"[{idx+1}/{len(combinations)}] Running Grid Config: {[f"{k}:{cfg[k]}" for k in dynamic_keys]}")
        print(f"=======================================================")

        # Ensure perfect reproducibility for each run
        seed = cfg.get('seed', 8281003564)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
        # Extract variables
        use_double = cfg.get('use_double', False)
        model_name = cfg.get('model', 'linear')
        epochs = cfg.get('epochs', 20)
        warming_iters = cfg.get('warming_iters', 6)
        batch_size = cfg.get('batch_size', 50)
        init = cfg.get('init', 'zeros')
        bias = cfg.get('bias', False)
        train_method = cfg.get('train_method', 'unrolled-sequential')
        rho = cfg.get('rho', 1.0)
        beta = cfg.get('beta', 0.1)
        thetas = cfg.get('thetas', 0.3)
        deltas = cfg.get('deltas', 0.95)
        n_timesteps = cfg.get('n_timesteps', 150)
        num_layers = cfg.get('layers', 2)
        use_fft = cfg.get('use_fft', True)
        padding_mode = cfg.get('padding_mode', 'circular')

        if use_double:
            torch.set_default_dtype(torch.float64)
            current_dtype = torch.float64
        else:
            torch.set_default_dtype(torch.float32)
            current_dtype = torch.float32

        # ---------------------------------------------------------
        # 1. LOAD DATASET
        # ---------------------------------------------------------

        if model_name in ["linear", "conv"]:
            data, targets= get_data(batch_size, spiking=False, device=device, seed=seed)
        else:
            data, targets = get_data(batch_size, spiking=True, device=device, n_timesteps=n_timesteps, seed=seed)
            
        targets = F.one_hot(targets.long(), num_classes=10).float()
        # ---------------------------------------------------------
        # 2. INITIALIZE ARCHITECTURE
        # ---------------------------------------------------------
        hidden_dims = int(cfg.get('hidden_dims', 100))
        hidden_channels = int(cfg.get('hidden_channels', 4))
        
        match model_name:
            case "spiking-linear":
                if num_layers == 2:
                    layers = nn.ModuleList([
                        ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_dims, h=ADMM_Heaviside(thetas=thetas), init=init, bias=bias),
                        ADMM_SpikingLinear(in_f=hidden_dims, out_f=10, h=None, init=init, bias=bias)
                    ])
                elif num_layers == 3:
                    mid_dims = hidden_dims // 2
                    layers = nn.ModuleList([
                        ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_dims, h=ADMM_Heaviside(thetas=thetas), init=init, bias=bias),
                        ADMM_SpikingLinear(in_f=hidden_dims, out_f=mid_dims, h=ADMM_Heaviside(thetas=thetas), init=init, bias=bias),
                        ADMM_SpikingLinear(in_f=mid_dims, out_f=10, h=None, init=init, bias=bias)
                    ])

            case "spiking-conv":
                k = cfg.get('kernel_size', 5)
                p = cfg.get('padding', 2)
                s = cfg.get('stride', 1) # FFTs require stride=1
                pool_h, pool_w = 8, 8
                # N-MNIST is 34x34. Calculate the exact output size of the Conv layer
                if num_layers == 2:
                    # Dynamically calculate the 18x18 output
                    spatial_out = int(calc_spatial_out(34, k, p, s)) 
                    # 2 * 18 * 18 = 648
                    lin_in = hidden_channels * spatial_out * spatial_out 
                    
                    layers = nn.ModuleList([
                        ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels, k=k, p=p, s=s, 
                                           h=ADMM_Heaviside(thetas=thetas), init=init, bias=bias, 
                                           use_fft=False,  padding_mode="zeros"),
                                           
                        # Passing 648 to in_f and using Flatten
                        ADMM_SpikingLinear(in_f=lin_in, out_f=10, init=init, h=None, 
                                           pool_op=ADMM_Flatten(), bias=bias)
                    ])
                elif num_layers == 3:
                    mid_p = k // 2 
                    mid_c = hidden_channels // 2
                    lin_in = mid_c * pool_h * pool_w
                    layers = nn.ModuleList([
                        ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels, k=k, p=p, s=1, h=ADMM_Heaviside(thetas=thetas), init=init, bias=bias, use_fft=True,  padding_mode="circular"),
                        ADMM_SpikingConv2d(in_c=hidden_channels, out_c=mid_c, k=k, p=mid_p, s=s, h=ADMM_Heaviside(thetas=thetas), init=init, bias=bias, use_fft=False, padding_mode="zeros"),
                        ADMM_SpikingLinear(in_f=lin_in, out_f=10, init=init, h=None, pool_op=ADMM_SpatialPool((pool_h, pool_w)), bias=bias)
                    ])

            case "linear": 
                layer_list = []
                current_in = 28 * 28

                # 1. Build all hidden layers (constant width)
                for _ in range(num_layers - 1):
                    layer_list.append(
                        ADMM_Linear(in_f=current_in, out_f=hidden_dims, h=ADMM_ReLU(), init=init, bias=bias)
                    )
                    # For all subsequent layers, the input is now hidden_dims
                    current_in = hidden_dims

                # 2. Build the final classification layer
                layer_list.append(
                    ADMM_Linear(in_f=current_in, out_f=10, h=ADMM_ReLU(), init=init, bias=bias)
                )

                # 3. Convert to PyTorch ModuleList
                layers = nn.ModuleList(layer_list)
        
            case "conv":
                k = cfg.get('kernel_size', 5)
                p = cfg.get('padding', 2)
                s = cfg.get('stride', 1)
                
                layer_list = []
                
                # Initial Image dimensions (1 channel, 28x28)
                current_in_c = 1
                current_spatial = 28
                
                # 1. Build N-2 Convolutional Layers (High-Res Feature Extractors)
                for _ in range(num_layers - 2):
                    layer_list.append(
                        ADMM_Conv2d(in_c=current_in_c, out_c=hidden_channels, k=k, p=p, s=1, 
                                    h=ADMM_ReLU(), init=init, bias=bias, 
                                    use_fft=True, padding_mode="circular") # <-- Changed to True for speed!
                    )
                    
                    # Update spatial size mathematically (using stride 1)
                    current_spatial = int(calc_spatial_out(current_spatial, k, p, 1))
                    current_in_c = hidden_channels
                
                # 2. Build the final Convolutional Layer (The Bottleneck)
                layer_list.append(
                    ADMM_Conv2d(in_c=current_in_c, out_c=hidden_channels, k=k, p=p, s=s, 
                                h=ADMM_ReLU(), init=init, bias=bias, 
                                use_fft=use_fft, padding_mode=padding_mode)
                )
                
                # CRITICAL FIX: Update the spatial size one last time based on stride `s`!
                current_spatial = int(calc_spatial_out(current_spatial, k, p, s))
                
                # 3. Build the final Linear Classification Layer
                lin_in = hidden_channels * current_spatial * current_spatial
                
                layer_list.append(
                    ADMM_Linear(in_f=lin_in, out_f=10, h=ADMM_ReLU(), init=init, 
                                pool_op=ADMM_Flatten(), bias=bias)
                )
                
                # 4. Convert to PyTorch ModuleList
                layers = nn.ModuleList(layer_list)
        # ---------------------------------------------------------
        # 3. INITIALIZE MODEL & METRICS
        # ---------------------------------------------------------
        if model_name.split('-')[0] == "spiking":
            model = ADMM(layers, T=n_timesteps, rho=rho, thetas=thetas, deltas=deltas, beta=beta, init=init, bias=bias, train_method=train_method).to(device)
        else: 
            model = ADMM(layers, rho=rho, beta=beta, init=init, bias=bias, train_method=train_method).to(device)
        
        m = ADMM_Metrics(model)
        model._init_states(data)
        
        if is_static_run:
            print(json.dumps(m.network_size_statistics(), indent=4))

        
        keys_to_exclude = {'model', 'batch_size', 'seed'} 
        filtered_keys = [k for k in dynamic_keys if k not in keys_to_exclude]

        # 3. Build the combo string using the filtered list
        if is_static_run or not filtered_keys:
            combo_str = "baseline"
        else:
            combo_str = "_".join([f"{k}-{cfg[k]}" for k in filtered_keys])
            
        metrics_path = f'metrics_test/{model_name}/{batch_size}/{combo_str}/{seed}'
        os.makedirs(metrics_path, exist_ok=True)

        print(f"Training...")

        accuracy_list = []
        firing_rate_list = []
        
        # ---------------------------------------------------------
        # 4. TRAINING LOOP
        # ---------------------------------------------------------
        start_time = time.time()
        for epoch in range(epochs + 1):
            model.fit(data, targets, warming=epoch < warming_iters)             
            if epoch % 1 == 0:
                with torch.no_grad():
                    raw_outputs, firing_rates = model.forward_model(data)
                    flat_outputs = raw_outputs.view(batch_size, -1)
                
                    _, preds = flat_outputs.max(dim=1)
                    accuracy = 100. * (preds == targets.argmax(dim=1)).sum().item() / batch_size

                    m.save_metrics(data, targets)

                    print(f"Epoch [{epoch:3d}/{epochs}] | Acc: {accuracy:6.2f}%| Firing rate: {[f'{v:.4f}' for v in firing_rates]} | {m}")
                
                    accuracy_list.append(accuracy)
                    firing_rate_list.append([f'{v:.4f}' for v in firing_rates])

        #########################################
        # SAVING RESULTS AND PLOTTING

        end_time = time.time()    
        running_time = end_time - start_time
        print(f"Model finished in {end_time - start_time:.2f} seconds.")       
        metrics=m.get_dic()
        metrics["accuracy_list"] = accuracy_list
        metrics["firing_rate"] = firing_rate_list
        metrics["running_time" ]= running_time
        
        # Store for plotting if static
        if is_static_run:
            final_metrics = metrics

        with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    # ==========================================
    # VISUALIZATION (Static Run Only)
    # ==========================================
    if is_static_run:
        print("\nRendering training plots for static run...")
        fig, ax = plt.subplots(2, 3, figsize=(30, 5))
        ax[0, 0].semilogy(metrics["lagrangian_cost"])
        ax[0, 0].set_title("Lagrangian")
        ax[0, 1].semilogy(metrics["primal_residual"])
        ax[0, 1].set_title("Primal Residual Norm")
        ax[0, 2].semilogy(metrics["activation_constraint_sum"])
        ax[0, 2].set_title("Activation Constraint (||a - h(z)||)")
        ax[1, 0].semilogy(metrics["preactivation_constraint_sum"])
        ax[1, 0].set_title("Preactivation Constraint") 
        ax[1, 1].semilogy(metrics["loss"])
        ax[1, 1].set_title("Loss")
        ax[1, 2].plot(metrics["accuracy_list"])
        ax[1, 2].set_title("Train Accuracy")
        
        plt.tight_layout()
        plt.show()
