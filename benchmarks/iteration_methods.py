import torch
import configparser
import torch.nn as nn
import time
import os       
import json
from utils.dataset import get_data
from admm import (
    ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpikingConv2d, 
    ADMM, ADMM_Heaviside, ADMM_Metrics, ADMM_Hinge
)

def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read('benchmarks/config/config.ini')
    # ==========================================
    # 1. GLOBAL SETUP & SEEDING
    # ==========================================
    seed = config.getint('config','seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    batch_size = config.getint('config', 'batch_size_spiking')
    epochs = config.getint('config', 'epochs')
    hidden_size_spiking = config.getint('config', 'hidden_size_spiking')
    hidden_channels_spiking =  config.getint('config', 'hidden_channels_spiking')
    k = config.getint('config', 'k')
    p = config.getint('config', 'p')
    s = config.getint('config', 's')
    n_timesteps = config.getint('config', 'n_timesteps')
    accuracy_threshold = config.getint('config', 'acc_threshold')
    min_warming_iters  = config.getint('config', 'min_warming_iters')
    max_warming_iters  = config.getint('config', 'max_warming_iters')
    primal_delta_limit =  config.getfloat('config', 'primal_threshold') 
     
    prev_primal_residual = float('inf')
    
    # Standard Scalar Hyperparameters
    linear_rho = config.getfloat('config', 'splinear_rho')
    linear_beta = config.getfloat('config', 'splinear_beta')
    conv_rho = config.getfloat('config', 'spconv_rho')
    conv_beta = config.getfloat('config', 'spconv_beta')
    deltas = config.getfloat('config', 'deltas')
    thetas = config.getfloat('config', 'thetas')
    
    # ==========================================
    # 2. DATASET LOADING (LOADED ONCE)
    # ==========================================

    raw_data, targets = get_data(batch_size, spiking=True, device=device, n_timesteps=n_timesteps)
    targets = torch.nn.functional.one_hot(targets.long(), num_classes=10).float()
    # ==========================================
    # 3. BENCHMARK LOOPS
    # ==========================================
    architectures = ["linear", "conv"]
    methods = [
        "unrolled-sequential", 
        "unrolled-random", 
        "unrolled-backwards",
        "decoupled-sequential", 
        "decoupled-random",
        "decoupled-backwards",
        "vectorized"
    ]
    
    print("\n" + "="*80)
    print(f"  ADMM 5-METHOD ARCHITECTURE BENCHMARK (Device: {device.type.upper()})")
    print("="*80)

    for arch in architectures:
        print(f"\n{'#'*60}")
        print(f"### EVALUATING ARCHITECTURE: {arch.upper()}")
        print(f"{'#'*60}")
        
        if arch == "linear":
            arch_data = raw_data.view(raw_data.size(0), raw_data.size(1), -1).permute(1, 0, 2)
        else:
            arch_data = raw_data.permute(1, 0, 2, 3, 4)
            
        for method in methods:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
    
            is_warming = True       
            prev_primal_residual = float('inf')
            warming_stop = None
            print(f"\n---> Testing Method: {method.upper()}")
            
            if arch == "linear":
                # 34 * 34 * 2 = 2312
                rho=linear_rho
                beta=linear_beta
                layers = nn.ModuleList([ 
                    ADMM_SpikingLinear(in_f=2312, out_f=hidden_size_spiking, h=ADMM_Heaviside(thetas=thetas), init="s-uniform", bias=False),
                    ADMM_SpikingLinear(in_f=hidden_size_spiking, out_f=10, h=None, init="s-uniform", bias=False)
                ])
            else:
                rho=conv_rho
                beta=conv_beta
                spatial = int(calc_spatial_out(34, k, p, s))
                lin_in = hidden_channels_spiking * spatial * spatial
                layers = nn.ModuleList([ 
                     ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels_spiking, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init="s-uniform", bias=False, use_fft=False,  padding_mode="zeros"),
                    ADMM_SpikingLinear(in_f=lin_in, pool_op=ADMM_Flatten(), out_f=10, h=None, init="s-uniform", bias=False)
           ])

            model = ADMM(
                layers, 
                T=n_timesteps, 
                loss_f=ADMM_Hinge(),
                rho=rho, 
                thetas=thetas, 
                deltas=deltas, 
                beta=beta, 
                init="s-uniform", 
                bias=False, 
                train_method=method
            ).to(device)
                
            m = ADMM_Metrics(model)
            model._init_states(arch_data)
            
            # ---------------------------------------------------------
            # TRAINING LOOP
            # ---------------------------------------------------------
            start_time = time.time()
            
            for epoch in range(epochs + 1):
                model.fit(arch_data, targets, warming=is_warming)            
                
                with torch.no_grad():                    
                    m.save_metrics(arch_data, targets)

                    print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

                    # --- DYNAMIC WARMING LOGIC ---
                    current_primal = m.metrics["primal_residual"][-1]
                    accuracy  = m.metrics["accuracy"][-1]
                    primal_residual_delta = abs(prev_primal_residual - current_primal)
                    prev_primal_residual = current_primal

                    if is_warming:
                        hit_accuracy = (accuracy > accuracy_threshold) and (epoch > min_warming_iters)
                        hit_time_limit = epoch >= max_warming_iters
                        
                        if hit_accuracy or hit_time_limit:
                            if primal_residual_delta < primal_delta_limit or hit_time_limit:
                                reason = "Accuracy/Delta Target Met" if hit_accuracy else "Max Epochs Reached"
                                print(f"--- STOPPING WARMING at Epoch {epoch} ({reason}) ---")
                                is_warming = False
                                warming_stop = epoch
                        
            end_time = time.time()    
            running_time = end_time - start_time
            print(f"[Finished {method} on {arch.upper()} in {running_time:.2f} seconds]")

            # ==========================================
            # 4. SAVING RESULTS TO JSON
            # ==========================================
            metrics=m.get_dic()
            metrics["running_time"] = running_time
            metrics["architecture"] = arch
            metrics["method"] = method
            metrics["batch_size"] = batch_size
            metrics["epochs"] = epochs
            metrics["seed"] = seed                    

            save_dir = f"benchmarks/results/iteration_methods/{arch}/{method}"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"batch_{batch_size}_seed_{seed}.json")

            with open(save_path, "w") as f:
                json.dump(metrics, f, indent=4)
                
            print(f"--> Saved results to: {save_path}\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()