import torch
import torch.nn as nn
from torchvision import datasets, transforms
import configparser
import torch.nn.functional as F
from admm import (
    ADMM_Flatten,
    ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_ReLU, ADMM_Metrics
)


import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read('benchmarks/config/config.ini') 

seed = config.getint('config','seed')

batch_size_static  = config.getint('config', 'batch_size_static')
epochs = config.getint('config', 'epochs')
hidden_size_static = 512
hidden_channels_static = 2
k =  config.getint('config', 'k')
p =  (k - 1) // 2 #To prevent shrinkage
s =  1

linear_rho = config.getfloat('config', 'linear_rho')
linear_beta = config.getfloat('config', 'linear_beta')
conv_rho = config.getfloat('config', 'conv_rho')
conv_beta = config.getfloat('config', 'conv_beta')
max_layers  = config.getint('config', 'max_layers')
input_size = 784

# WARMING CONFIG
accuracy_threshold = config.getint('config', 'acc_threshold')
min_warming_iters  = config.getint('config', 'min_warming_iters')
max_warming_iters  = config.getint('config', 'max_warming_iters')
primal_delta_limit = config.getfloat('config', 'primal_threshold') 

def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1

########################################

model_types = ["linear", "conv"]

batch_size = batch_size_static
transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,))
            ])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
images_orig, labels = next(iter(dataloader))
images_orig, labels = images_orig.to(device), labels.to(device)
images_orig += 0.01 * torch.randn_like(images_orig)
labels_one_hot = F.one_hot(labels.long(), num_classes=10).float()


for model_name in model_types:
    print(f"\n{'='*50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'='*50}")
    for layers in range(1, max_layers + 1):
        print(f"\nHidden layers:{layers}")
        images = images_orig.clone()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
 
        is_warming = True       
        prev_primal_residual = float('inf')
        warming_stop = None
        
        #########################################
        # MODEL INSTANTIATION
        layer_list=[]
        match model_name:
           
            case "linear":
                layer_list.append(ADMM_Linear(in_f=input_size, out_f=hidden_size_static, h=ADMM_ReLU(), init="s-uniform", bias=True))
                for _ in range(layers - 1):
                    layer_list.append(
                        ADMM_Linear(in_f=hidden_size_static, out_f=hidden_size_static, h=ADMM_ReLU(), init="zeros", bias=True)
                    )

                layer_list.append(ADMM_Linear(in_f=hidden_size_static, out_f=10, h=ADMM_ReLU(), init="s-uniform", bias=True))

                linear_layers = nn.ModuleList(layer_list)

                admm_model = ADMM(linear_layers, rho=linear_rho, beta=linear_beta, init="s-uniform", bias=True, train_method='vectorized').to(device)
                images = images.view(images.size(0), -1)

            case "conv":
                current_spatial = 28
                
                layer_list.append(ADMM_Conv2d(in_c=1, out_c=hidden_channels_static, k=k, p=p, s=s, h=ADMM_ReLU(), init="s-uniform", bias=True, use_fft=False, padding_mode='zeros'))
                current_spatial = int(calc_spatial_out(current_spatial, k, p, s))
                
                for _ in range(layers - 1):
                    layer_list.append(
                        ADMM_Conv2d(in_c=hidden_channels_static, out_c=hidden_channels_static, k=k, p=p, s=s, h=ADMM_ReLU(), init="s-uniform", bias=True, use_fft=False, padding_mode='zeros')
                    )
                    current_spatial = int(calc_spatial_out(current_spatial, k, p, s))

                lin_in_dim = hidden_channels_static * current_spatial * current_spatial
                layer_list.append(ADMM_Linear(in_f=lin_in_dim, out_f=10, h=ADMM_ReLU(), pool_op=ADMM_Flatten(), init="s-uniform", bias=True))
                conv_layers = nn.ModuleList(layer_list)

                admm_model = ADMM(conv_layers, rho=conv_rho, beta=conv_beta, init="s-uniform", bias=True, train_method='vectorized').to(device) 
   
        #########################################
        # ADMM TRAINING LOOP
        m = ADMM_Metrics(admm_model) 
        admm_model._init_states(images)
        accuracy_list = []            

        print("Training model with ADMM...")
        for epoch in range(epochs):
            admm_model.fit(images, labels_one_hot, warming=is_warming)             
            
            with torch.no_grad():
                raw_outputs, _ = admm_model.forward_model(images)
                flat_outputs = raw_outputs.view(batch_size, -1) 
                _, predictions = flat_outputs.max(dim=1)
                accuracy = 100. * (predictions == labels_one_hot.argmax(dim=1)).sum().item() / batch_size
                m.save_metrics(images, labels_one_hot)

                print(f"Epoch [{epoch:3d}/{epochs}] | Acc: {accuracy:6.2f}% | {m}")
               
                accuracy_list.append(accuracy)
                    
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
        # SAVING RESULTS
        metrics=m.get_dic()
        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["layers"]=layers
        metrics["warming_stop"]=warming_stop
        metrics["epochs"] = epochs
        metrics["seed"] = seed                    
        metrics["accuracy_list"] = accuracy_list

        metrics_filename = f"benchmarks/results/static_breakpoint/{model_name}/{batch_size}/{layers}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

