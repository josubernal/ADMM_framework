import configparser
import json
import os

import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim

from admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_Conv2d,
    ADMM_CrossEntropy_Taylor,
    ADMM_Flatten,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Linear,
    ADMM_Metrics,
    ADMM_ReLU,
    ADMM_SpikingConv2d,
    ADMM_SpikingLinear,
)
from utils.dataset import get_dataset

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
accuracy_threshold = config.getint("config", "acc_threshold")
min_warming_iters = config.getint("config", "min_warming_iters")
max_warming_iters = config.getint("config", "max_warming_iters")
primal_delta_limit = config.getfloat("config", "primal_threshold")


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
        batch_size = batch_size_static

        dataset = get_dataset(batch_size, n_batches, spiking=False, seed=seed)
    else:
        batch_size = batch_size_spiking
        dataset = get_dataset(
            batch_size,
            spiking=True,
            n_batches=n_batches,
            n_timesteps=n_timesteps,
            seed=seed,
        )

    #########################################
    # MODEL INSTANTIATION
    match model_name:
        case "linear":
            model = GDLinearNet().to(device)
            layer_config = ADMM_LayerConfig(
                use_bias=True, rho=linear_rho, beta=linear_beta
            )
            linear_layers = nn.ModuleList(
                [
                    ADMM_Linear(
                        in_f=input_size,
                        out_f=hidden_size_static,
                        h=ADMM_ReLU(),
                        config=layer_config,
                    ),
                    ADMM_Linear(
                        in_f=hidden_size_static,
                        out_f=10,
                        h=ADMM_ReLU(),
                        config=layer_config,
                    ),
                ]
            )
            config = ADMM_Config(init="pytorch", train_method="vectorized")
            admm_model = ADMM(linear_layers, loss_f=ADMM_SSE(), config=config).to(
                device
            )

        case "conv":
            model = GDConvNet().to(device)
            spatial_dim = int(calc_spatial_out(28, k, p, s))
            lin_in_dim = hidden_channels_static * spatial_dim * spatial_dim
            layer_config = ADMM_LayerConfig(
                use_bias=True,
                rho=conv_rho,
                beta=conv_beta,
                use_fft=False,
            )
            conv_layers = nn.ModuleList(
                [
                    ADMM_Conv2d(
                        in_c=1,
                        out_c=hidden_channels_static,
                        k=k,
                        p=p,
                        s=s,
                        h=ADMM_ReLU(),
                        config=layer_config,
                        padding_mode="zeros",
                    ),
                    ADMM_Linear(
                        in_f=lin_in_dim,
                        out_f=10,
                        h=ADMM_ReLU(),
                        pool_op=ADMM_Flatten(),
                        config=layer_config,
                    ),
                ]
            )
            config = ADMM_Config(init="pytorch", train_method="vectorized")
            admm_model = ADMM(
                conv_layers,
                loss_f=ADMM_SSE(),
                config=config,
            ).to(device)

        case "spiking-linear":
            model = GDSpLinearNet().to(device)
            layer_config = ADMM_LayerConfig(
                use_bias=False,
                rho=splinear_rho,
                beta=splinear_beta,
                thetas=thetas,
                deltas=deltas,
            )
            splinear_layers = nn.ModuleList(
                [
                    ADMM_SpikingLinear(
                        in_f=34 * 34 * 2,
                        out_f=hidden_size_spiking,
                        h=ADMM_Heaviside(),
                        config=layer_config,
                    ),
                    ADMM_SpikingLinear(
                        in_f=hidden_size_spiking,
                        out_f=10,
                        h=None,
                        config=layer_config,
                        use_reset=False,
                    ),
                ]
            )
            config = ADMM_Config(
                init="s-uniform",
                train_method="decoupled-backwards",
            )
            admm_model = ADMM(
                splinear_layers,
                loss_f=ADMM_CrossEntropy_Taylor(),
                T=n_timesteps,
                config=config,
            ).to(device)

        case "spiking-conv":
            model = GDSpConvNet().to(device)
            spatial_dim = int(calc_spatial_out(34, k, p, s))
            lin_in_dim = hidden_channels_spiking * spatial_dim * spatial_dim
            layer_config = ADMM_LayerConfig(
                use_bias=False,
                rho=spconv_rho,
                beta=spconv_beta,
                thetas=thetas,
                deltas=deltas,
                use_fft=False,
            )
            spconv_layers = nn.ModuleList(
                [
                    ADMM_SpikingConv2d(
                        in_c=2,
                        out_c=hidden_channels_spiking,
                        k=k,
                        p=p,
                        s=s,
                        h=ADMM_Heaviside(),
                        config=layer_config,
                        padding_mode="zeros",
                    ),
                    ADMM_SpikingLinear(
                        in_f=lin_in_dim,
                        pool_op=ADMM_Flatten(),
                        out_f=10,
                        h=None,
                        config=layer_config,
                        use_reset=False,
                    ),
                ]
            )
            config = ADMM_Config(
                init="s-uniform",
                train_method="decoupled-backwards",
            )
            admm_model = ADMM(
                spconv_layers,
                loss_f=ADMM_CrossEntropy_Taylor(),
                T=n_timesteps,
                config=config,
            ).to(device)

    #########################################
    # ADMM TRAINING LOOP
    criterion = nn.CrossEntropyLoss()
    m = ADMM_Metrics(admm_model)

    print("\nTraining model with ADMM...")
    for epoch in range(epochs):
        admm_model.fit(dataset, warming=is_warming)

        with torch.no_grad():
            inputs, labels, batch_state = model.state_handler.load_batch(0)
            if labels.dim() == 1 or labels.shape[-1] != 10:
                labels = torch.nn.functional.one_hot(
                    labels.long(), num_classes=10
                ).float()
            m.save_metrics(inputs, labels, batch_state)
            print(f"Epoch [{epoch:3d}/{epochs}] | {m}")
            # --- DYNAMIC WARMING LOGIC ---
            current_primal = m.metrics["primal_residual"][-1]
            accuracy = m.metrics["accuracy"][-1]
            primal_residual_delta = abs(prev_primal_residual - current_primal)
            prev_primal_residual = current_primal

            if is_warming:
                hit_accuracy = (accuracy > accuracy_threshold) and (
                    epoch > min_warming_iters
                )
                hit_time_limit = epoch >= max_warming_iters

                if hit_accuracy or hit_time_limit:
                    if primal_residual_delta < primal_delta_limit or hit_time_limit:
                        reason = (
                            "Accuracy/Delta Target Met"
                            if hit_accuracy
                            else "Max Epochs Reached"
                        )
                        print(f"--- STOPPING WARMING at Epoch {epoch} ({reason}) ---")
                        is_warming = False
                        warming_stop = epoch

    #########################################
    # GRADIENT DESCENT TRAINING LOOP

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
