import configparser
import json
import os

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from benchmarks.utils.dataset import get_data, get_dataset
from benchmarks.utils.models import get_model
from src.admm import ADMM_SSE, ADMM_CrossEntropy_Taylor, ADMM_Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("benchmarks/config/config.ini")

seed = config.getint("config", "seed")

batch_size_static = config.getint("config", "batch_size_static")
n_batches = config.getint("config", "n_batches")
batch_size_spiking = config.getint("config", "batch_size_spiking")
epochs = config.getint("config", "epochs")
hidden_size_static = config.getint("config", "hidden_size_static")
hidden_channels_static = config.getint("config", "hidden_channels_static")
hidden_size_spiking = config.getint("config", "hidden_size_spiking")
hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")
lr = config.getfloat("config", "learning_rate")

linear_rho = config.getfloat("config", "linear_rho")
linear_beta = config.getfloat("config", "linear_beta")
conv_rho = config.getfloat("config", "conv_rho")
conv_beta = config.getfloat("config", "conv_beta")
splinear_rho = config.getfloat("config", "splinear_rho")
splinear_beta = config.getfloat("config", "splinear_beta")
spconv_rho = config.getfloat("config", "spconv_rho")
spconv_beta = config.getfloat("config", "spconv_beta")

deltas = config.getfloat("config", "deltas")
thetas = config.getfloat("config", "thetas")
input_size = 784
n_timesteps = config.getint("config", "n_timesteps")

# WARMING CONFIG
warming_iters = epochs // 2


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


########################################
# MODELS
class GDLinearNet(nn.Module):
    def __init__(
        self, input_size=input_size, hidden_size=hidden_size_static, num_classes=10
    ):
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
    def __init__(
        self, hidden_channels=hidden_channels_static, k=k, s=s, p=p, num_classes=10
    ):
        super(GDConvNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=k,
            stride=s,
            padding=p,
        )
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
    def __init__(
        self,
        input_size=34 * 34 * 2,
        hidden_size=hidden_size_spiking,
        num_classes=10,
        beta=deltas,
        threshold=thetas,
        num_steps=n_timesteps,
    ):
        super(GDSpLinearNet, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
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
    def __init__(
        self,
        hidden_channels=hidden_channels_spiking,
        num_classes=10,
        beta=deltas,
        threshold=thetas,
        num_steps=n_timesteps,
    ):
        super(GDSpConvNet, self).__init__()
        self.num_steps = num_steps
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")

    is_warming = True
    prev_primal_residual = float("inf")
    warming_stop = None

    #########################################
    # DATA
    if model_name in ["linear", "conv"]:
        train_loader = get_dataset(batch_size_static, 1, spiking=False, seed=seed)
    else:
        train_loader = get_dataset(
            batch_size_spiking,
            1,
            spiking=True,
            n_timesteps=n_timesteps,
            seed=seed,
        )

    #########################################
    # MODEL INSTANTIATION
    match model_name:
        case "linear":
            model = GDLinearNet().to(device)
        case "conv":
            model = GDConvNet().to(device)
        case "spiking-linear":
            model = GDSpLinearNet().to(device)
        case "spiking-conv":
            model = GDSpConvNet().to(device)

    match model_name:
        case "linear":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
            )
        case "conv":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
            )
        case "spiking-linear":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled-backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
            )
        case "spiking-conv":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled-backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
            )

    #########################################
    # ADMM TRAINING LOOP

    m = ADMM_Metrics(admm_model)

    for epoch in range(epochs + 1):
        admm_model.fit(train_loader, warming=epoch < warming_iters)
        if epoch % 1 == 0:
            with torch.no_grad():
                m.save_metrics()
                print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

    #########################################
    # GRADIENT DESCENT TRAINING LOOP
    criterion = nn.CrossEntropyLoss()
    # DATA
    if model_name in ["linear", "conv"]:
        batch_size = batch_size_static
        images, labels = get_data(batch_size, spiking=False, device=device, seed=seed)
    else:
        batch_size = batch_size_spiking
        images, labels = get_data(
            batch_size,
            spiking=True,
            device=device,
            n_timesteps=n_timesteps,
            seed=seed,
        )

    match model_name:
        case "linear":
            images = images.view(images.size(0), -1)
        case "spiking-linear":
            images = images.view(images.size(0), images.size(1), -1).permute(1, 0, 2)
        case "spiking-conv":
            images = images.permute(1, 0, 2, 3, 4)

    labels_one_hot = F.one_hot(labels.long(), num_classes=10).float()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    gd_accs = []

    print("\nTraining model with gradient descent...")
    for epoch in range(epochs):
        outputs = model(images)
        loss = criterion(outputs, labels.long())

        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gd_accs.append(accuracy)

        print(f"Step {epoch + 1} |  Acc: {accuracy:.4f}")

    #########################################
    # SAVING RESULTS AND PLOTTING
    metrics = m.get_dic()
    metrics["architecture"] = model_name
    metrics["batch_size"] = batch_size
    metrics["warming_stop"] = warming_stop
    metrics["epochs"] = epochs
    metrics["seed"] = seed
    metrics["gd_accuracy"] = gd_accs

    metrics_filename = (
        f"benchmarks/results/gd_comparation/{model_name}/{batch_size}/results.json"
    )
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
