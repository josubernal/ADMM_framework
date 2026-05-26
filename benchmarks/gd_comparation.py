import configparser
import json
import os

import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.utils.dataset import get_data, get_dataset
from benchmarks.utils.models import get_model
from src.admm import ADMM_SSE, ADMM_Metrics

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

ff_rho = config.getfloat("config", "ff_rho")
ff_beta = config.getfloat("config", "ff_beta")
conv_rho = config.getfloat("config", "conv_rho")
conv_beta = config.getfloat("config", "conv_beta")
spff_rho = config.getfloat("config", "spff_rho")
spff_beta = config.getfloat("config", "spff_beta")
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
class GDFFNet(nn.Module):
    def __init__(
        self, input_size=input_size, hidden_size=hidden_size_static, num_classes=10
    ):
        super(GDFFNet, self).__init__()
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


class GDSpFFNet(nn.Module):
    def __init__(
        self,
        input_size=34 * 34 * 2,
        hidden_size=hidden_size_spiking,
        num_classes=10,
        beta=deltas,
        threshold=thetas,
        num_steps=n_timesteps,
    ):
        super(GDSpFFNet, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x, return_spikes=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        if return_spikes:
            spikes_layer1 = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            if return_spikes:
                spikes_layer1.append(spk1)

            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)

        if return_spikes:
            # Calculate mean firing rate across time, batch, and neurons
            fr1 = torch.stack(spikes_layer1).mean().item()
            return mem2, [fr1]

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

    def forward(self, x, return_spikes=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        if return_spikes:
            spikes_layer1 = []

        for step in range(self.num_steps):
            cur1 = self.conv(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            if return_spikes:
                spikes_layer1.append(spk1)

            spk1_flat = self.flatten(spk1)
            cur2 = self.fc2(spk1_flat)
            _, mem2 = self.lif2(cur2, mem2)

        if return_spikes:
            # Calculate mean firing rate across time, batch, and spatial dimensions
            fr1 = torch.stack(spikes_layer1).mean().item()
            return mem2, [fr1]

        return mem2


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["feedforward", "conv", "spiking-feedforward", "spiking-conv"]

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
    if model_name in ["feedforward", "conv"]:
        train_loader = get_dataset(model_name, batch_size_static, 1, seed=seed)
    else:
        train_loader = get_dataset(
            model_name,
            batch_size_spiking,
            1,
            n_timesteps=n_timesteps,
            seed=seed,
        )

    #########################################
    # MODEL INSTANTIATION
    match model_name:
        case "feedforward":
            model = GDFFNet().to(device)
        case "conv":
            model = GDConvNet().to(device)
        case "spiking-feedforward":
            model = GDSpFFNet().to(device)
        case "spiking-conv":
            model = GDSpConvNet().to(device)

    match model_name:
        case "feedforward":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
                block_method="two-block",
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
                block_method="two-block",
            )
        case "spiking-feedforward":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="unrolled-backwards",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
                block_method="two-block",
            )
        case "spiking-conv":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="unrolled-backwards",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
                block_method="two-block",
            )

    #########################################
    # ADMM TRAINING LOOP

    m = ADMM_Metrics(admm_model)

    for epoch in range(epochs + 1):
        admm_model.fit(train_loader, warming=epoch < warming_iters)
        if epoch % 1 == 0:
            with torch.no_grad():
                m.save_metrics()
                print(
                    f"Epoch {epoch} | W_norm: {admm_model.layers[0].W.norm().item():.2f}"
                )
                print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

    #########################################
    # GRADIENT DESCENT TRAINING LOOP
    criterion = nn.CrossEntropyLoss()
    # DATA
    images, labels = get_data(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        n_timesteps=n_timesteps,
        seed=seed,
    )

    if labels.dim() > 1:
        labels = labels.argmax(dim=1)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    gd_accs = []
    gd_losses = []
    gd_frs = []

    print("\nTraining model with gradient descent...")
    for epoch in range(epochs):
        # Extract outputs and conditionally extract firing rates
        if model_name in ["spiking-feedforward", "spiking-conv"]:
            outputs, frs = model(images, return_spikes=True)
        else:
            outputs = model(images)
            frs = [float("nan")]

        loss = criterion(outputs, labels.long())

        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append metrics
        gd_accs.append(accuracy)
        gd_losses.append(loss.item())
        gd_frs.append(frs)

        print(f"Step {epoch + 1} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")

    #########################################
    # SAVING RESULTS AND PLOTTING
    metrics = m.get_dic()
    metrics["architecture"] = model_name
    metrics["batch_size"] = batch_size
    metrics["warming_stop"] = warming_stop
    metrics["epochs"] = epochs
    metrics["seed"] = seed
    metrics["gd_accuracy"] = gd_accs
    metrics["gd_loss"] = gd_losses
    metrics["gd_firing_rate"] = gd_frs

    metrics_filename = (
        f"benchmarks/results/gd_comparation/{model_name}/{batch_size}/results.json"
    )
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
