import configparser
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ADMM_Scheduler,
    ADMM_SpikingConv2d,
    ADMM_SpikingLinear,
)
from utils.dataset import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("benchmarks/config/config.ini")

seed = config.getint("config", "seed")

batch_size_static = config.getint("config", "batch_size_static")
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


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
model_types = ["linear", "conv", "spiking-linear", "spiking-conv"]

# --- NEW: The 4 Lagrange Configurations ---
choices = ["none", "last_only", "first_only", "both"]

for model_name in model_types:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")

    for lagrange_config in choices:
        print(f"\nLAGRANGE CONFIG: {lagrange_config.upper()}")

        # Reset seeds per model to guarantee identical environments
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #########################################
        # DATA
        if model_name in ["linear", "conv"]:
            batch_size = batch_size_static
            images, labels = get_data(
                batch_size, spiking=False, device=device, seed=seed
            )
        else:
            batch_size = batch_size_spiking
            images, labels = get_data(
                batch_size,
                spiking=True,
                device=device,
                n_timesteps=n_timesteps,
                seed=seed,
            )

        labels_one_hot = F.one_hot(labels.long(), num_classes=10).float()

        #########################################
        # MODEL INSTANTIATION
        match model_name:
            case "linear":
                layer_config = ADMM_LayerConfig(
                    rho=linear_rho, beta=linear_beta, use_bias=True
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
                images = images.view(images.size(0), -1)

            case "conv":
                spatial_dim = int(calc_spatial_out(28, k, p, s))
                lin_in_dim = hidden_channels_static * spatial_dim * spatial_dim
                layer_config = ADMM_LayerConfig(
                    rho=conv_rho, beta=conv_beta, use_bias=True, use_fft=False
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
                layer_config = ADMM_LayerConfig(
                    rho=splinear_rho,
                    beta=splinear_beta,
                    thetas=thetas,
                    deltas=deltas,
                    use_bias=False,
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
                    init="s-uniform", train_method="decoupled-backwards"
                )
                admm_model = ADMM(
                    splinear_layers,
                    T=n_timesteps,
                    loss_f=ADMM_CrossEntropy_Taylor(),
                    config=config,
                ).to(device)
                images = images.view(images.size(0), images.size(1), -1).permute(
                    1, 0, 2
                )

            case "spiking-conv":
                spatial_dim = int(calc_spatial_out(34, k, p, s))
                lin_in_dim = hidden_channels_spiking * spatial_dim * spatial_dim
                layer_config = ADMM_LayerConfig(
                    rho=spconv_rho,
                    beta=spconv_beta,
                    thetas=thetas,
                    deltas=deltas,
                    use_bias=False,
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
                admm_model = ADMM(
                    spconv_layers,
                    T=n_timesteps,
                    loss_f=ADMM_CrossEntropy_Taylor(),
                    config=config,
                ).to(device)
                images = images.permute(1, 0, 2, 3, 4)

        # ---------------------------------------------------------
        # --- NEW: APPLY LAYER-WISE LAGRANGE OVERRIDES HERE ---
        # ---------------------------------------------------------
        if lagrange_config == "none":
            admm_model.layers[0].config.use_lagrange = False
            admm_model.layers[1].config.use_lagrange = False
        elif lagrange_config == "last_only":
            admm_model.layers[0].config.use_lagrange = False
            admm_model.layers[1].config.use_lagrange = True
        elif lagrange_config == "first_only":
            admm_model.layers[0].config.use_lagrange = True
            admm_model.layers[1].config.use_lagrange = False
        elif lagrange_config == "both":
            admm_model.layers[0].config.use_lagrange = True
            admm_model.layers[1].config.use_lagrange = True

        #########################################
        # ADMM TRAINING LOOP
        m = ADMM_Metrics(admm_model)
        balancer = ADMM_Scheduler(
            admm_model, mu=10.0, tau=2.0, balance_freq=5, stop_epoch=int(epochs * 0.75)
        )

        print("Training model with ADMM...")
        for epoch in range(epochs):
            balancer.capture_state()
            admm_model.fit(images, labels_one_hot, warming=False)

            with torch.no_grad():
                m.save_metrics(images, labels_one_hot)

                print(f"Epoch [{epoch:3d}/{epochs}] |  {m}")

                # 3. Read the Primal Residuals (These are LISTS of per-layer residuals)
                primal_rho_list = m.metrics["preactivation_constraint_sum"][-1]
                primal_beta_list = m.metrics["activation_constraint_sum"][-1]

                # 4. Balance the network!
                balancer.step(primal_rho_list, primal_beta_list)

        #########################################
        # SAVING RESULTS AND PLOTTING
        metrics = m.get_dic()
        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["lagrange_config"] = lagrange_config  # Log the config!
        metrics["epochs"] = epochs
        metrics["seed"] = seed

        # Output to a lagrange folder for easy plotting
        metrics_filename = f"benchmarks/results/lagrange/{model_name}/{batch_size}/{lagrange_config}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next configuration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
