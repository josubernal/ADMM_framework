import torch
import configparser
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as tr
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import itertools
from torchvision import datasets, transforms
import time

from admm import ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpatialPool, ADMM_SpikingConv2d, ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_Heaviside, ADMM_ReLU, ADMM_Metrics

def parse_value(v):
    """Smartly parse strings from the INI file into their correct Python types."""
    v = v.strip()
    if v.lower() == 'true': return True
    if v.lower() == 'false': return False
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v  # Returns as string if it's neither bool, int, nor float

if __name__ == "__main__":
    seed = 8281003564
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = configparser.ConfigParser()
    config.read('config/config_grid.ini')
    
    # --- Parse Static Params ---
    static_params = {}
    if config.has_section('static'):
        for key, val in config.items('static'):
            static_params[key] = parse_value(val)

    # --- Parse Grid Params ---
    grid_params = {}
    if config.has_section('grid'):
        for key, val in config.items('grid'):
            grid_params[key] = [parse_value(v) for v in val.split(',')]
            
    # Generate all combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if not combinations:
        combinations = [{}] # Run once if no grid params provided

    def calc_spatial_out(size_in, k, p, s):
        return ((size_in + 2 * p - k) // s) + 1

    print(f"Total grid search configurations to evaluate: {len(combinations)}")
    print(f"Combinations: {combinations}")
    
    # ==========================================
    # GRID SEARCH LOOP
    # ==========================================
    for idx, combo in enumerate(combinations):
        print(f"\n=======================================================")
        print(f"[{idx+1}/{len(combinations)}] Running Grid Config: {combo}")
        print(f"=======================================================")

        seed = 8281003564
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        
        # Merge static params with the current grid combo (combo overwrites static)
        cfg = static_params.copy()
        cfg.update(combo)

        # Extract variables from the merged config
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


        if use_double:
            torch.set_default_dtype(torch.float64)
            current_dtype = torch.float64
        else:
            torch.set_default_dtype(torch.float32)
            current_dtype = torch.float32

        # ---------------------------------------------------------
        # 1. LOAD DATASET (Inside loop because batch_size/model can change)
        # ---------------------------------------------------------
        if model_name.split('-')[0] == "spiking":
            sensor_size = tonic.datasets.NMNIST.sensor_size
            frame_transform = tr.Compose([
                    tr.Denoise(filter_time=10000),
                    tr.ToFrame(sensor_size=sensor_size, time_window=1000)
                ])
            trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
            cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
            train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                                        collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, generator = torch.Generator().manual_seed(seed))
            
            data, targets_orig = next(iter(train_loader))
            data, targets_orig = data.to(device), targets_orig.to(device)
            targets = torch.nn.functional.one_hot(targets_orig.to(torch.long), num_classes=10).to(current_dtype)

            if data.size(1) > n_timesteps:
                data = data[:, :n_timesteps, :]

            if "linear" in model_name:
                data = data.view(data.size(0), data.size(1), -1)
                data = data.permute(1, 0, 2).to(current_dtype)
            else:
                data = data.permute(1, 0, 2, 3, 4).to(current_dtype)
        else: 
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=image_transform)
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
      
            data, targets_orig = next(iter(train_loader))
            data, targets_orig = data.to(device), targets_orig.to(device)
            if "conv" in model_name:
                data = data.to(current_dtype) 
            else:
                data = data.view(data.size(0), -1).to(current_dtype)
            targets = torch.nn.functional.one_hot(targets_orig, num_classes=10).to(current_dtype)

        # Add minor noise
        data += 0.01 * torch.randn_like(data) 

        # ---------------------------------------------------------
        # 2. INITIALIZE ARCHITECTURE
        # ---------------------------------------------------------
        hidden_dims = int(cfg.get('hidden_dims', 100))
        hidden_channels = int(cfg.get('hidden_channels', 4))
        num_layers = int(cfg.get('layers', 2))
        match model_name:
            case "spiking-linear":
                if num_layers == 2:
                    layers = nn.ModuleList([
                        ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_dims, h=ADMM_Heaviside(thetas=thetas), init=init),
                        ADMM_SpikingLinear(in_f=hidden_dims, out_f=10, h=None, init=init)
                    ])
                elif num_layers == 3:
                    mid_dims = hidden_dims // 2
                    layers = nn.ModuleList([
                        ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_dims, h=ADMM_Heaviside(thetas=thetas), init=init),
                        ADMM_SpikingLinear(in_f=hidden_dims, out_f=mid_dims, h=ADMM_Heaviside(thetas=thetas), init=init),
                        ADMM_SpikingLinear(in_f=mid_dims, out_f=10, h=None, init=init)
                    ])

            case "spiking-conv":
                k = cfg.get('kernel_size', 5)
                p = cfg.get('padding', 2)
                s = cfg.get('stride', 1)
                pool_h, pool_w = 4, 4 
                
                if num_layers == 2:
                    # Layer 2 needs FULL hidden_channels because layer 1 outputs all of them
                    lin_in = hidden_channels * pool_h * pool_w
                    layers = nn.ModuleList([
                        ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init=init),
                        ADMM_SpikingLinear(in_f=lin_in, out_f=10, init=init, h=None, pool_op=ADMM_SpatialPool((pool_h, pool_w)))
                    ])
                elif num_layers == 3:
                    mid_p = k // 2 
                    mid_c = hidden_channels // 2
                    # Layer 3 needs HALVED channels because layer 2 reduced them
                    lin_in = mid_c * pool_h * pool_w
                    layers = nn.ModuleList([
                        ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init=init),
                        ADMM_SpikingConv2d(in_c=hidden_channels, out_c=mid_c, k=k, p=mid_p, s=1, h=ADMM_Heaviside(thetas=thetas), init=init),
                        ADMM_SpikingLinear(in_f=lin_in, out_f=10, init=init, h=None, pool_op=ADMM_SpatialPool((pool_h, pool_w)))
                    ])

            case "linear":  
                if num_layers == 2:
                    layers = nn.ModuleList([
                        ADMM_Linear(in_f=28*28, out_f=hidden_dims, h=ADMM_ReLU(), init=init),
                        ADMM_Linear(in_f=hidden_dims, out_f=10, h=ADMM_ReLU(), init=init)
                    ])
                elif num_layers == 3:
                    mid_dims = hidden_dims // 2
                    layers = nn.ModuleList([
                        ADMM_Linear(in_f=28*28, out_f=hidden_dims, h=ADMM_ReLU(), init=init),
                        ADMM_Linear(in_f=hidden_dims, out_f=mid_dims, h=ADMM_ReLU(), init=init),
                        ADMM_Linear(in_f=mid_dims, out_f=10, h=ADMM_ReLU(), init=init)
                    ])
                    
            case "conv":
                k = cfg.get('kernel_size', 5)
                p = cfg.get('padding', 2)
                s = cfg.get('stride', 1)
                
                if num_layers == 2:
                    spatial_out = int(calc_spatial_out(28, k, p, s))
                    lin_in = hidden_channels * spatial_out * spatial_out
                    layers = nn.ModuleList([
                        ADMM_Conv2d(in_c=1, out_c=hidden_channels, k=k, p=p, s=s, h=ADMM_ReLU(), init=init),       
                        ADMM_Linear(in_f=lin_in, out_f=10, h=ADMM_ReLU(), init=init, pool_op=ADMM_Flatten())
                    ])
                elif num_layers == 3:
                    spatial_out_1 = int(calc_spatial_out(28, k, p, s))
                    spatial_out_2 = int(calc_spatial_out(spatial_out_1, k, p, s))
                    mid_c = hidden_channels // 2
                    lin_in = mid_c * spatial_out_2 * spatial_out_2
                    
                    layers = nn.ModuleList([
                        ADMM_Conv2d(in_c=1, out_c=hidden_channels, k=k, p=p, s=s, h=ADMM_ReLU(), init=init), 
                        ADMM_Conv2d(in_c=hidden_channels, out_c=mid_c, k=k, p=p, s=s, h=ADMM_ReLU(), init=init),
                        ADMM_Linear(in_f=lin_in, out_f=10, h=ADMM_ReLU(), init=init, pool_op=ADMM_Flatten())
                    ])
        # ---------------------------------------------------------
        # 3. INITIALIZE MODEL & METRICS
        # ---------------------------------------------------------
        if model_name.split('-')[0] == "spiking":
            model = ADMM(layers, T=n_timesteps, rho=rho, thetas=thetas, deltas=deltas, beta=beta, init=init, bias=bias, train_method=train_method).to(device)
        else: 
            model = ADMM(layers, rho=rho, beta=beta, init=init, bias=bias, train_method=train_method).to(device)
        
        m = ADMM_Metrics(model)
        model._init_states(data)

        # Setup paths
        combo_str = "_".join([f"{k}-{v}" for k, v in combo.items()]) if combo else "baseline"
        metrics_path = f'metrics_test/{model_name}/{batch_size}/{combo_str}'
        os.makedirs(metrics_path, exist_ok=True)

        print(f"Training model {model_name}...")

        metrics = {}
        lagrangians, lambdas = [], []
        soft_constraints = {"a": [], "z": []}
        losses = []
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

                    current_metrics = m.get_all_metrics(data, targets)
                    
                    mse = current_metrics["mse"]
                    lagr = current_metrics["lagrangian_cost"]
                    primal = current_metrics["primal_residual"]
                    preactivation_constraint_sum = current_metrics["preactivation_constraint_sum"]
                    activation_constraint_sum = current_metrics["activation_constraint_sum"]

                    print(f"Epoch [{epoch:3d}/{epochs}] "
                          f"| MSE: {mse:.4f} "
                          f"| Acc: {accuracy:6.2f}% "
                          f"| Firing rate: {[f'{v:.4f}' for v in firing_rates]} "
                          f"| Lagr: {lagr:10.2f} "
                          f"| Lamb: {primal:10.2f}")

                    losses.append(mse)
                    accuracy_list.append(accuracy)
                    firing_rate_list.append([f'{v:.4f}' for v in firing_rates])
                    soft_constraints["a"].append(activation_constraint_sum)
                    soft_constraints["z"].append(preactivation_constraint_sum)
                    lagrangians.append(lagr)  
                    lambdas.append(primal)
         
        end_time = time.time()    
        print(f"Model finished in {end_time - start_time:.2f} seconds.")       
        metrics["lagrangians"] = lagrangians
        metrics["lambdas"] = lambdas
        metrics["soft_constraints"] = soft_constraints
        metrics["losses"] = losses
        metrics["accuracy_list"] = accuracy_list
        metrics["firing_rate"] = firing_rate_list

        with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
