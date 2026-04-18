import torch
import configparser
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as tr
from torch.utils.data import DataLoader
import torch.nn as nn
import time

from admm import (
    ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpikingConv2d, 
    ADMM, ADMM_Heaviside, ADMM_Metrics
)

def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read('benchmarks/benchmarks_config/config.ini')
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
    hidden_size = config.getint('config', 'hidden_size')
    hidden_channels = config.getint('config', 'hidden_channels')
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
    rho = config.getfloat('config', 'splinear_rho')
    beta = config.getfloat('config', 'splinear_beta')
    deltas = config.getfloat('config', 'deltas')
    thetas = config.getfloat('config', 'thetas')
    
    # ==========================================
    # 2. DATASET LOADING (LOADED ONCE)
    # ==========================================
    print("Loading NMNIST Dataset...")
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = tr.Compose([
        tr.Denoise(filter_time=10000),
        tr.ToFrame(sensor_size=sensor_size, time_window=1000)
    ])
    
    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    train_loader = DataLoader(
        cached_trainset, batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, 
        generator=torch.Generator().manual_seed(seed)
    )
            
    raw_data, targets_orig = next(iter(train_loader))
    raw_data, targets_orig = raw_data.to(device), targets_orig.to(device)
    targets = torch.nn.functional.one_hot(targets_orig.to(torch.long), num_classes=10)

    if raw_data.size(1) > n_timesteps:
        raw_data = raw_data[:, :n_timesteps, :]

    raw_data += 0.01 * torch.randn_like(raw_data) # Symmetry breaking noise

    # ==========================================
    # 3. BENCHMARK LOOPS
    # ==========================================
    architectures = ["linear", "conv"]
    methods = [
        "unrolled-sequential", 
        "unrolled-random", 
        "decoupled-sequential", 
        "decoupled-random",
        "vectorized"
    ]
    
    print("\n" + "="*80)
    print(f"  ADMM 5-METHOD ARCHITECTURE BENCHMARK (Device: {device.type.upper()})")
    print("="*80)

    for arch in architectures:
        print(f"\n{'#'*60}")
        print(f"### EVALUATING ARCHITECTURE: {arch.upper()}")
        print(f"{'#'*60}")
        
        # Prepare Data Shape for Architecture
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
            print(f"\n---> Testing Method: {method.upper()}")
            
            # Construct fresh layers to prevent state bleeding between runs
            h_func = ADMM_Heaviside(thetas=thetas)
            
            if arch == "linear":
                # 34 * 34 * 2 = 2312
                layers = nn.ModuleList([ 
                    ADMM_SpikingLinear(in_f=2312, out_f=hidden_size, h=h_func, init='zeros', bias=False),
                    ADMM_SpikingLinear(in_f=hidden_size, out_f=10, h=None, init='zeros', bias=False)
                ])
            else:
                spatial = int(calc_spatial_out(34, k, p, s))
                lin_in = hidden_channels * spatial * spatial
                layers = nn.ModuleList([ 
                    ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels, k=k, p=p, s=s, h=h_func, init="zeros", bias=False, padding_mode="zeros"),
                    ADMM_SpikingLinear(in_f=lin_in, pool_op=ADMM_Flatten(), out_f=10, h=None, init='zeros', bias=False)
                ])

            # Instantiate fresh model
            model = ADMM(
                layers, 
                T=n_timesteps, 
                rho=rho, 
                thetas=thetas, 
                deltas=deltas, 
                beta=beta, 
                init='zeros', 
                bias=False, 
                train_method=method
            ).to(device)
                
            m = ADMM_Metrics(model)
            model._init_states(arch_data)
            
            metrics = {"losses": [], "accuracy": [], "lagrangians": [], "lambdas": []}
            
            # ---------------------------------------------------------
            # TRAINING LOOP
            # ---------------------------------------------------------
            start_time = time.time()
            
            for epoch in range(epochs + 1):
                model.fit(arch_data, targets, warming=is_warming)            
                
                with torch.no_grad():
                    outputs, firing_rates = model.forward_model(arch_data)
                    flat_outputs = outputs.view(batch_size, -1)
                
                    _, preds = flat_outputs.max(dim=1)
                    accuracy = 100. * (preds == targets.argmax(dim=1)).sum().item() / batch_size

                    current_metrics = m.get_all_metrics(arch_data, targets)
                    
                    mse = current_metrics["mse"]
                    lagr = current_metrics["lagrangian_cost"]
                    primal = current_metrics["primal_residual"]

                    print(f"  Epoch [{epoch:3d}/{epochs}] "
                          f"| MSE: {mse:.4f} "
                          f"| Acc: {accuracy:6.2f}% "
                          f"| Lagr: {lagr:10.2f} "
                          f"| Lamb: {primal:10.2f}")

                    metrics["losses"].append(mse)
                    metrics["accuracy"].append(accuracy)
                    metrics["lagrangians"].append(lagr)  
                    metrics["lambdas"].append(primal)
                    current_primal = current_metrics.get("primal_residual", 0.0)
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