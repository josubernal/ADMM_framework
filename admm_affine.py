
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
from torchvision import datasets, transforms

from admm import  ADMM_SpikingLinear, ADMM_GAP, ADMM_Flatten, ADMM_SpatialPool, ADMM_SpikingConv2d, ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_Heaviside, ADMM_ReLU, ADMM_Metrics


if __name__ == "__main__":
    seed = 8281003564
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    use_double = config.getboolean('affine', 'use_double', fallback=False)
    
    if use_double:
        torch.set_default_dtype(torch.float64)
        current_dtype = torch.float64
        print("Running in DOUBLE (64-bit) precision.\n")
    else:
        torch.set_default_dtype(torch.float32)
        current_dtype = torch.float32
        print("Running in SINGLE (32-bit) precision.\n")  

    model_name=config.get('affine', 'model')

    epochs = config.getint('affine', 'epochs')
    warming_iters = config.getint('affine', 'warming_iters')
    batch_size = config.getint('affine', 'batch_size')
    rho = config.getfloat('affine', 'rho')
    beta = config.getfloat('affine', 'beta')
    
    print(f"Performing training for {batch_size} images, {config.get('affine', 'train_method')}.")
    print()
    
    
    init=config.get('affine', 'init')
    bias=config.getboolean('affine', 'bias')
    train_method=config.get('affine', 'train_method') 
    

        
    def calc_spatial_out(size_in, k, p, s):
        return ((size_in + 2 * p - k) // s) + 1
    
    match model_name:
        case "spiking-linear":
            hidden_dims = config.getint('affine', 'hidden_dims')
            thetas = config.getfloat('affine', 'thetas')   
            layers= nn.ModuleList([
                ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_dims, h=ADMM_Heaviside(thetas=thetas), init=init),
                ADMM_SpikingLinear(in_f=hidden_dims,  out_f=10, h=None, init=init)
            ])

        case "spiking-conv":
            hidden_channels = config.getint('affine', 'hidden_channels')
            k = config.getint('affine', 'kernel_size')
            p = config.getint('affine', 'padding')
            s = config.getint('affine', 'stride')
            thetas = config.getfloat('affine', 'thetas') 
            pool_h, pool_w = 4, 4 

            linear_in_features = hidden_channels * pool_h * pool_w
            
            layers = nn.ModuleList([
                ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init=init),
                ADMM_SpikingLinear(in_f=linear_in_features, out_f=10, init=init,  h=None, pool_op=ADMM_SpatialPool((pool_h, pool_w)))
            ])

        case "linear":  
            hidden_dims = config.getint('affine', 'hidden_dims') 
            layers= nn.ModuleList([
                    ADMM_Linear(in_f=28*28, out_f=hidden_dims, h=ADMM_ReLU(), init=init),
                    ADMM_Linear(in_f=hidden_dims,  out_f=10, h=ADMM_ReLU(), init=init)
                ])
        case "conv":
            hidden_channels = config.getint('affine', 'hidden_channels')
            k=config.getint('affine', 'kernel_size')
            p=config.getint('affine', 'padding')
            s=config.getint('affine', 'stride')
            spatial_out = calc_spatial_out(28, k, p, s)
            linear_in_features = hidden_channels * spatial_out * spatial_out
            layers = nn.ModuleList([
                ADMM_Conv2d(in_c=1, out_c=hidden_channels, k=k, p=p, s=s, h=ADMM_ReLU(), init=init),       
                ADMM_Linear(in_f=linear_in_features, out_f=10, h=ADMM_ReLU(), init=init, pool_op=ADMM_Flatten())
            ])
            
        
           
        case _:
            raise ValueError('Model not defined')


    
    if model_name.split('-')[0]=="spiking":
        n_timesteps = config.getint('affine', 'n_timesteps')
        deltas = config.getfloat('affine', 'deltas')
        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tr.Compose([
                tr.Denoise(filter_time=10000),
                tr.ToFrame(sensor_size=sensor_size, time_window=1000)
            ])

        trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
        cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
        train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                                    collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, generator = torch.Generator().manual_seed(seed))
        
        sample_data, sample_target = next(iter(train_loader))
        data, targets = next(iter(train_loader))
        data, targets = data.to(device), targets.to(device)
        
        if data.size(1) > n_timesteps:
                data = data[:, :n_timesteps, :]

        if "linear" in model_name:
            data = data.view(data.size(0), data.size(1), -1)
            data = data.permute(1, 0, 2).to(current_dtype)
        else:
            data = data.permute(1, 0, 2, 3, 4).to(current_dtype)
        
        targets = torch.nn.functional.one_hot(targets.to(torch.long), num_classes=10).to(current_dtype)

        model = ADMM(layers, T=n_timesteps,  rho= rho, thetas=thetas, deltas=deltas, beta=beta, init=init, bias=bias, train_method=train_method).to(device)
    else: 
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=image_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
  
        sample_data, sample_target = next(iter(train_loader))
        data, targets = next(iter(train_loader))
        data, targets= data.to(device), targets.to(device)
        data_flat =  data.view(data.size(0), -1).to(current_dtype)
        targets = torch.nn.functional.one_hot(targets, num_classes=10).to(current_dtype)
        
        model = ADMM(layers, rho= rho, beta=beta, init=init, bias=bias, train_method=train_method).to(device)
    
    
    m=ADMM_Metrics(model)
    
    model._init_states(data)

    print(json.dumps(m.network_size_statistics(), indent=4))
    
    data += 0.01 * torch.randn_like(data) 
    metrics = {}

    metrics_path = f'metrics_test/{batch_size}/linear'
    os.makedirs(metrics_path, exist_ok = True)

    print(f"Training model {model} - Starting Training...")

    
    lagrangians, lambdas = [], []
    soft_constraints = {"a": [], "z": []}
    losses = []
    accuracy_list = []
    firing_rate_list = []
    
    for epoch in range(epochs+1):
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
                lam_norm = model.lambda_lagrange.norm().item()
                lam_elements = model.lambda_lagrange.numel()
                lam_sum_metric = lam_norm / lam_elements

                print(f"Epoch [{epoch:3d}/{epochs}] "
                      f"| MSE: {mse:.4f} "
                      f"| Acc: {accuracy:6.2f}% "
                      f"| Firing rate: {[f'{v:.4f}' for v in firing_rates]} "
                      f"| Lambda sum: {lam_sum_metric:.4f} "
                      f"| Lagr: {lagr:10.2f} "
                      f"| Lamb: {primal:10.2f} "
                      f"| Pre: {[f'{v:.4f}' for v in preactivation_constraint_sum]} "
                      f"| Act: {[f'{v:.4f}' for v in activation_constraint_sum]}") 

                losses.append(mse)
                accuracy_list.append(accuracy)
                firing_rate_list.append([f'{v:.4f}' for v in firing_rates])
                soft_constraints["a"].append(activation_constraint_sum)
                soft_constraints["z"].append(preactivation_constraint_sum)
                lagrangians.append(lagr)  
                lambdas.append(primal)
                
    metrics["lagrangians"] = lagrangians
    metrics["lambdas"] = lambdas
    metrics["soft_constraints"] = soft_constraints
    metrics["losses"] = losses
    metrics["accuracy_list"] = accuracy_list
    metrics["firing_rate"] = firing_rate_list

    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plots to visualize training metrics
    fig, ax = plt.subplots(2, 3, figsize=(30, 5))
    ax[0, 0].semilogy(lagrangians)
    ax[0, 0].set_title("Lagrangian")
    ax[0, 1].semilogy(lambdas)
    ax[0, 1].set_title("Primal Residual Norm")
    ax[0, 2].semilogy(soft_constraints["a"])
    ax[0, 2].set_title(r"$||a - h(z, \theta)||_2$")
    ax[1, 0].semilogy(soft_constraints["z"])
    ax[1, 0].set_title(r"$||z_1 - \delta*z_{shifted} - W_1a_0 + \theta*a_{1,shifted}||_2$") 
    ax[1, 1].semilogy(losses)
    ax[1, 1].set_title("Loss")
    ax[1, 2].plot(accuracy_list)
    ax[1, 2].set_title("Train Accuracy")
        
    # Run final evaluation on training data
    pot, firing_rate = model.forward_model(data)
    _, predicted = (pot.view(batch_size, -1)).max(1)
    labels = torch.argmax(targets, dim=1)
    print("----------------------------")
    print("Train accuracy:", (predicted == labels).sum().item() / labels.size(0))
    print("----------------------------")
    plt.tight_layout()
    plt.show()
  