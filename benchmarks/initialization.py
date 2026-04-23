import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import configparser
import torch.nn.functional as F
from admm import (
    ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpikingConv2d, 
    ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_Heaviside, ADMM_ReLU, ADMM_Metrics
)
import snntorch as snn
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as tr
from torch.utils.data import DataLoader
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read('benchmarks/config/config.ini') 

seed = config.getint('config','seed')

batch_size_static  = config.getint('config', 'batch_size_static')
batch_size_spiking = config.getint('config', 'batch_size_spiking')
epochs = config.getint('config', 'epochs')
hidden_size_static = config.getint('config', 'hidden_size_static')
hidden_channels_static =  config.getint('config', 'hidden_channels_static')
hidden_size_spiking = config.getint('config', 'hidden_size_spiking')
hidden_channels_spiking =  config.getint('config', 'hidden_channels_spiking')
k =  config.getint('config', 'k')
p =  config.getint('config', 'p')
s =  config.getint('config', 's')
lr = config.getfloat('config', 'learning_rate')

linear_rho = config.getfloat('config', 'linear_rho')
linear_beta = config.getfloat('config', 'linear_beta')
conv_rho = config.getfloat('config', 'conv_rho')
conv_beta = config.getfloat('config', 'conv_beta')
splinear_rho = config.getfloat('config', 'splinear_rho')
splinear_beta = config.getfloat('config', 'splinear_beta')
spconv_rho = config.getfloat('config', 'spconv_rho')
spconv_beta = config.getfloat('config', 'spconv_beta')

deltas = config.getfloat('config', 'deltas')
thetas = config.getfloat('config', 'thetas')
input_size = 784
n_timesteps = config.getint('config', 'n_timesteps')

# WARMING CONFIG
accuracy_threshold = config.getint('config', 'acc_threshold')
min_warming_iters  = config.getint('config', 'min_warming_iters')
max_warming_iters  = config.getint('config', 'max_warming_iters')
primal_delta_limit = config.getfloat('config', 'primal_threshold') 

def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1

########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["linear", "conv", "spiking-linear", "spiking-conv"]
initializations = ["wzeros", "wxavier", "wkaiming", "wrandom","wpytorch","wdata","wthreshold","szeros","swarm","snoisy","srandom","spiking-szeros","spiking-swarm","spiking-snoisy","spiking-srandom"]

for model_name in model_types:
    print(f"\n{'='*50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'='*50}")
    for initialization in initializations:
        if model_name in ["linear", "conv"] and["wthreshold","spiking-szeros","spiking-swarm","spiking-snoisy","spiking-srandom"]:
            continue
        if model_name in ["spiking-linear", "spiking-conv"] and initialization in ["wxavier", "wkaiming","wdata","wpytorch","szeros","swarm","snoisy","srandom"]:
            continue
        print(f"\nINITIALIZATION: {initialization}")
        # Reset seeds per model to guarantee identical environments
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        


        # Reset warming logic for each model
        is_warming = True       
        prev_primal_residual = float('inf')
        warming_stop = None

        #########################################
        # DATA
        if model_name.startswith("spiking"):
            batch_size = batch_size_spiking
            sensor_size = tonic.datasets.NMNIST.sensor_size
            frame_transform = tr.Compose([
                            tr.Denoise(filter_time=10000),
                            tr.ToFrame(sensor_size=sensor_size, time_window=1000)
                        ])
            trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
            cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
            train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                                    collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
                    
            images, labels = next(iter(train_loader))
            if images.size(1) > n_timesteps:
                images = images[:, :n_timesteps, :]

        else: 
            batch_size = batch_size_static
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,))
            ])

            mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
            images, labels = next(iter(dataloader))

        images, labels = images.to(device), labels.to(device)
        images += 0.01 * torch.randn_like(images)
        labels_one_hot = F.one_hot(labels.long(), num_classes=10).float()

        #########################################
        # MODEL INSTANTIATION
        match model_name:
            case "linear":
                linear_layers = nn.ModuleList([
                    ADMM_Linear(in_f=input_size, out_f=hidden_size_static, h=ADMM_ReLU(), init=initialization, bias=True),
                    ADMM_Linear(in_f=hidden_size_static, out_f=10, h=ADMM_ReLU(), init=initialization, bias=True)
                ])
                admm_model = ADMM(linear_layers, rho=linear_rho, beta=linear_beta, init=initialization, bias=True, train_method='vectorized').to(device)
                images = images.view(images.size(0), -1)

            case "conv":
                spatial_dim = int(calc_spatial_out(28, k, p, s))
                lin_in_dim = hidden_channels_static * spatial_dim * spatial_dim
                conv_layers = nn.ModuleList([
                    ADMM_Conv2d(in_c=1, out_c=hidden_channels_static, k=k, p=p, s=s, h=ADMM_ReLU(), init=initialization, bias=True, use_fft=False, padding_mode='zeros'),
                    ADMM_Linear(in_f=lin_in_dim, out_f=10, h=ADMM_ReLU(),pool_op=ADMM_Flatten(), init=initialization, bias=True)
                ])
                admm_model = ADMM(conv_layers, rho=conv_rho, beta=conv_beta, init=initialization, bias=True, train_method='vectorized').to(device) 
            case "spiking-linear":
                splinear_layers = nn.ModuleList([ 
                    ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_size_spiking, h=ADMM_Heaviside(thetas=thetas), init=initialization, bias=False),
                    ADMM_SpikingLinear(in_f=hidden_size_spiking, out_f=10, h=None, init=initialization, bias=False)
                ])
                admm_model = ADMM(splinear_layers, T=n_timesteps, rho=splinear_rho, thetas=thetas, deltas=deltas, beta=splinear_beta, init=initialization, bias=False, train_method='decoupled-sequential').to(device)
                images = images.view(images.size(0), images.size(1), -1).permute(1, 0, 2)
            case "spiking-conv":
                spatial_dim = int(calc_spatial_out(34, k, p, s))
                lin_in_dim = hidden_channels_spiking * spatial_dim * spatial_dim
                spconv_layers = nn.ModuleList([ 
                    ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels_spiking, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init=initialization, bias=False, use_fft=False,  padding_mode="zeros"),
                    ADMM_SpikingLinear(in_f=lin_in_dim, pool_op=ADMM_Flatten(), out_f=10, h=None, init=initialization, bias=False)
                ])
                admm_model = ADMM(spconv_layers, T=n_timesteps, rho=spconv_rho, thetas=thetas, deltas=deltas, beta=spconv_beta, init=initialization, bias=False, train_method='decoupled-sequential').to(device)
                images = images.permute(1, 0, 2, 3, 4)
        #########################################
        # ADMM TRAINING LOOP
        criterion = nn.CrossEntropyLoss()
        m = ADMM_Metrics(admm_model) 
        admm_model._init_states(images)
        metrics = {}
        lagrangians, lambdas = [], []
        soft_constraints = {"a": [], "z": []}
        losses = []
        accuracy_list = []
        firing_rate_list = []

        print("Training model with ADMM...")
        for epoch in range(epochs):
            admm_model.fit(images, labels_one_hot, warming=is_warming)             
            
            with torch.no_grad():
                raw_outputs, firing_rates = admm_model.forward_model(images)
                flat_outputs = raw_outputs.view(batch_size, -1) 
                _, predictions = flat_outputs.max(dim=1)
                accuracy = 100. * (predictions == labels_one_hot.argmax(dim=1)).sum().item() / batch_size
                current_metrics = m.get_all_metrics(images, labels_one_hot)
                    
                #loss = criterion(raw_outputs, labels_one_)
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
                
                # --- DYNAMIC WARMING LOGIC ---
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
                        
        #########################################
        # SAVING RESULTS AND PLOTTING

        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["initialization"] = initialization
        metrics["warming_stop"] = warming_stop
        metrics["epochs"] = epochs
        metrics["seed"] = seed                    
        metrics["lagrangians"] = lagrangians
        metrics["lambdas"] = lambdas
        metrics["soft_constraints"] = soft_constraints
        metrics["losses"] = losses
        metrics["accuracy_list"] = accuracy_list
        metrics["firing_rate"] = firing_rate_list

        metrics_filename = f"benchmarks/results/initialization/{model_name}/{batch_size}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)


        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

