import torch
import torch.nn as nn
import configparser
import torch.nn.functional as F
from admm import (
    ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpikingConv2d, 
    ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_Heaviside, ADMM_ReLU, ADMM_Metrics,
    ADMM_SSE, ADMM_Hinge
)

import json
import os
from utils.dataset import get_data

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

for model_name in model_types:
    print(f"\n{'='*50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'='*50}")
    for loss in [ADMM_SSE(), ADMM_Hinge()]:
        print(f"LOSS: {loss}")
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
        if model_name in ["linear", "conv"]:
            batch_size= batch_size_static
            images, labels= get_data(batch_size, spiking=False, device=device, seed=seed)
        else:
            batch_size= batch_size_spiking
            images, labels = get_data(batch_size, spiking=True, device=device, n_timesteps=n_timesteps, seed=seed)
            
        labels_one_hot = F.one_hot(labels.long(), num_classes=10).float()

        #########################################
        # MODEL INSTANTIATION
        match model_name:
            case "linear":
                linear_layers = nn.ModuleList([
                    ADMM_Linear(in_f=input_size, out_f=hidden_size_static, h=ADMM_ReLU(), init="zeros", bias=True),
                    ADMM_Linear(in_f=hidden_size_static, out_f=10, h=ADMM_ReLU(), init="zeros", bias=True)
                ])
                admm_model = ADMM(linear_layers, loss_f=loss, rho=linear_rho, beta=linear_beta, init="zeros", bias=True, train_method='vectorized').to(device)
                images = images.view(images.size(0), -1)

            case "conv":
                spatial_dim = int(calc_spatial_out(28, k, p, s))
                lin_in_dim = hidden_channels_static * spatial_dim * spatial_dim
                conv_layers = nn.ModuleList([
                    ADMM_Conv2d(in_c=1, out_c=hidden_channels_static, k=k, p=p, s=s, h=ADMM_ReLU(), init="zeros", bias=True, use_fft=False, padding_mode='zeros'),
                    ADMM_Linear(in_f=lin_in_dim, out_f=10, h=ADMM_ReLU(),pool_op=ADMM_Flatten(), init="zeros", bias=True)
                ])
                admm_model = ADMM(conv_layers, loss_f=loss, rho=conv_rho, beta=conv_beta, init="zeros", bias=True, train_method='vectorized').to(device) 
            case "spiking-linear":
                splinear_layers = nn.ModuleList([ 
                    ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_size_spiking, h=ADMM_Heaviside(thetas=thetas), init="zeros", bias=False),
                    ADMM_SpikingLinear(in_f=hidden_size_spiking, out_f=10, h=None, init="zeros", bias=False)
                ])
                admm_model = ADMM(splinear_layers, loss_f=loss, T=n_timesteps, rho=splinear_rho, thetas=thetas, deltas=deltas, beta=splinear_beta, init="zeros", bias=False, train_method='decoupled-backwards').to(device)
                images = images.view(images.size(0), images.size(1), -1).permute(1, 0, 2)
            case "spiking-conv":
                spatial_dim = int(calc_spatial_out(34, k, p, s))
                lin_in_dim = hidden_channels_spiking * spatial_dim * spatial_dim
                spconv_layers = nn.ModuleList([ 
                    ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels_spiking, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init="zeros", bias=False, use_fft=False,  padding_mode="zeros"),
                    ADMM_SpikingLinear(in_f=lin_in_dim, pool_op=ADMM_Flatten(), out_f=10, h=None, init="zeros", bias=False)
                ])
                admm_model = ADMM(spconv_layers,loss_f=loss, T=n_timesteps, rho=spconv_rho, thetas=thetas, deltas=deltas, beta=spconv_beta, init="zeros", bias=False, train_method='decoupled-backwards').to(device)
                images = images.permute(1, 0, 2, 3, 4)
        #########################################
        # ADMM TRAINING LOOP
        criterion = nn.CrossEntropyLoss()
        m = ADMM_Metrics(admm_model) 
        admm_model._init_states(images)
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
                m.save_metrics(images, labels_one_hot)

                print(f"Epoch [{epoch:3d}/{epochs}] | Acc: {accuracy:6.2f}%| Firing rate: {[f'{v:.4f}' for v in firing_rates]} | {m}")
               
                accuracy_list.append(accuracy)
                firing_rate_list.append([f'{v:.4f}' for v in firing_rates])

                # --- DYNAMIC WARMING LOGIC ---
                current_primal = m.metrics["primal_residual"][-1]
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
        metrics=m.get_dic()
        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["warming_stop"] = warming_stop
        metrics["epochs"] = epochs
        metrics["seed"] = seed                    
        metrics["accuracy_list"] = accuracy_list
        metrics["firing_rate"] = firing_rate_list
        metrics["loss_function"] = str(loss)

        metrics_filename = f"benchmarks/results/loss_function/{model_name}/{batch_size}/{str(loss)}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

