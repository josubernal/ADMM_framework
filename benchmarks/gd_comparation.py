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
# MODELS
class GDLinearNet(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size_static, num_classes=10):
        super(GDLinearNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GDConvNet(nn.Module):
    def __init__(self, hidden_channels=hidden_channels_static, k=k, s=s, p=p, num_classes=10):
        super(GDConvNet, self).__init__() 
        self.conv = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=k, stride=s, padding=p)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        spatial = ((28 + 2 * p - k) // s) + 1
        lin_in = hidden_channels * spatial * spatial
        self.fc2 = nn.Linear(lin_in, num_classes, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc2(x)
        return x

class GDSpLinearNet(nn.Module):
    def __init__(self, input_size=34*34*2, hidden_size=hidden_size_spiking, num_classes=10, beta=deltas, threshold=thetas, num_steps=n_timesteps):
        super(GDSpLinearNet, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size ,bias=False)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for step in range(self.num_steps):
            cur1 = self.fc1(x[step]) 
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)
        return mem2
    
class GDSpConvNet(nn.Module):
    def __init__(self, hidden_channels=hidden_channels_spiking, num_classes=10, beta=deltas, threshold=thetas, num_steps=n_timesteps):
        super(GDSpConvNet, self).__init__()
        self.num_steps = num_steps
        self.conv = nn.Conv2d(in_channels=2, out_channels=hidden_channels, kernel_size=k, stride=s, padding=p, bias=False)
        spatial = ((34 + 2 * p - k) // s) + 1
        lin_in = hidden_channels * spatial * spatial
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(lin_in, num_classes, bias=False)
        
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for step in range(self.num_steps):
            cur1 = self.conv(x[step]) 
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_flat = self.flatten(spk1)
            cur2 = self.fc2(spk1_flat)
            _, mem2 = self.lif2(cur2, mem2)
        return mem2


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["linear", "conv", "spiking-linear", "spiking-conv"]

for model_name in model_types:
    # Reset seeds per model to guarantee identical environments
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    print(f"\n{'='*50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'='*50}")

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
    labels_one_hot = F.one_hot(labels.long(), num_classes=10).float()

    #########################################
    # MODEL INSTANTIATION
    match model_name:
        case "linear":
            model  = GDLinearNet().to(device)
            linear_layers = nn.ModuleList([
                ADMM_Linear(in_f=input_size, out_f=hidden_size_static, h=ADMM_ReLU(), init='zeros', bias=True),
                ADMM_Linear(in_f=hidden_size_static, out_f=10, h=ADMM_ReLU(), init='zeros', bias=True)
            ])
            admm_model = ADMM(linear_layers, rho=linear_rho, beta=linear_beta, init='zeros', bias=True, train_method='vectorized').to(device)
#            with torch.no_grad():
#                linear_layers[0].W.copy_(model.fc1.weight)
#                linear_layers[0].b.copy_(model.fc1.bias)
#                linear_layers[1].W.copy_(model.fc2.weight)
#                linear_layers[1].b.copy_(model.fc2.bias)
            images = images.view(images.size(0), -1)

        case "conv":
            model = GDConvNet().to(device)
            spatial_dim = int(calc_spatial_out(28, k, p, s))
            lin_in_dim = hidden_channels_static * spatial_dim * spatial_dim
            conv_layers = nn.ModuleList([
                ADMM_Conv2d(in_c=1, out_c=hidden_channels_static, k=k, p=p, s=s, h=ADMM_ReLU(), init='zeros', bias=True, use_fft=False, padding_mode='zeros'),
                ADMM_Linear(in_f=lin_in_dim, out_f=10, h=ADMM_ReLU(),pool_op=ADMM_Flatten(), init='zeros', bias=True)
            ])
            admm_model = ADMM(conv_layers, rho=conv_rho, beta=conv_beta, init='zeros', bias=True, train_method='vectorized').to(device) 
#            with torch.no_grad():
#                conv_layers[0].W.copy_(model.conv.weight)
#                conv_layers[0].b.copy_(model.conv.bias)
#                conv_layers[1].W.copy_(model.fc2.weight)
#                conv_layers[1].b.copy_(model.fc2.bias)  

        case "spiking-linear":
            model = GDSpLinearNet().to(device)
            splinear_layers = nn.ModuleList([ 
                ADMM_SpikingLinear(in_f=34*34*2, out_f=hidden_size_spiking, h=ADMM_Heaviside(thetas=thetas), init='zeros', bias=False),
                ADMM_SpikingLinear(in_f=hidden_size_spiking, out_f=10, h=None, init='zeros', bias=False)
            ])
            admm_model = ADMM(splinear_layers, T=n_timesteps, rho=splinear_rho, thetas=thetas, deltas=deltas, beta=splinear_beta, init='zeros', bias=False, train_method='decoupled-sequential').to(device)
            images = images.view(images.size(0), images.size(1), -1).permute(1, 0, 2)
#            with torch.no_grad():
#                splinear_layers[0].W.copy_(model.fc1.weight)
#                splinear_layers[1].W.copy_(model.fc2.weight)

        case "spiking-conv":
            model = GDSpConvNet().to(device)
            spatial_dim = int(calc_spatial_out(34, k, p, s))
            lin_in_dim = hidden_channels_spiking * spatial_dim * spatial_dim
            spconv_layers = nn.ModuleList([ 
                ADMM_SpikingConv2d(in_c=2, out_c=hidden_channels_spiking, k=k, p=p, s=s, h=ADMM_Heaviside(thetas=thetas), init="zeros", bias=False, use_fft=False,  padding_mode="zeros"),
                ADMM_SpikingLinear(in_f=lin_in_dim, pool_op=ADMM_Flatten(), out_f=10, h=None, init='zeros', bias=False)
            ])
            admm_model = ADMM(spconv_layers, T=n_timesteps, rho=spconv_rho, thetas=thetas, deltas=deltas, beta=spconv_beta, init='zeros', bias=False, train_method='decoupled-sequential').to(device)
            images = images.permute(1, 0, 2, 3, 4)
#            with torch.no_grad():
#                spconv_layers[0].W.copy_(model.conv.weight)
#                spconv_layers[1].W.copy_(model.fc2.weight)

    #########################################
    # ADMM TRAINING LOOP
    criterion = nn.MSELoss()
    m = ADMM_Metrics(admm_model) 
    admm_model._init_states(images)
    admm_steps = []
    admm_mses = []
    admm_accs = []

    print("\nTraining model with ADMM...")
    for epoch in range(epochs):
        admm_model.fit(images, labels_one_hot, warming=is_warming)             
        
        with torch.no_grad():
            raw_outputs, firing_rates = admm_model.forward_model(images)
            flat_outputs = raw_outputs.view(batch_size, -1) 
            _, predictions = flat_outputs.max(dim=1)
            accuracy = 100. * (predictions == labels_one_hot.argmax(dim=1)).sum().item() / batch_size
            current_metrics = m.get_all_metrics(images, labels_one_hot)
                
            mse = criterion(raw_outputs, labels_one_hot)
            admm_steps.append(epoch)
            admm_mses.append(mse.item())
            admm_accs.append(accuracy)
            
            lagr = current_metrics["lagrangian_cost"]
            primal = current_metrics["primal_residual"]

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
            
            print(f"Epoch [{epoch+1:3d}/{epochs}] | MSE: {mse:.4f} | Acc: {accuracy:6.2f}% | Firing rate: {[f'{v:.4f}' for v in firing_rates]} | Lagr: {lagr:10.2f} | Lamb: {primal:10.2f}")

    #########################################
    # GRADIENT DESCENT TRAINING LOOP

    optimizer = optim.Adam(model.parameters(), lr=lr)

    gd_steps = []
    gd_mses = []
    gd_accs = []
    
    print("\nTraining model with gradient descent...")
    for epoch in range(epochs):
        outputs = model(images)
        loss = criterion(outputs, labels_one_hot)
        
        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        gd_steps.append(epoch)
        gd_mses.append(loss.item())
        gd_accs.append(accuracy)
        
        print(f"Step {epoch + 1} | MSE: {loss.item():.4f} | Acc: {accuracy:.4f}")

    #########################################
    # SAVING RESULTS AND PLOTTING
    metrics_data = {
        "model_name": model_name,
        "batch_size": batch_size,
        "warming_stop": warming_stop,
        "gd_mses": gd_mses,
        "gd_accs": gd_accs,
        "admm_mses": admm_mses,
        "admm_accs": admm_accs
    }

    metrics_filename = f"benchmarks/results/gd_comparation/{model_name}/{batch_size}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics_data, f, indent=4)


    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

