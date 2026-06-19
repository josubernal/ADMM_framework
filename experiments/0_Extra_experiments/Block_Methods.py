import configparser
import json
import os
import time

import torch

from src.admm import ADMM_SSE, ADMM_CrossEntropy_Taylor, ADMM_Metrics

from .utils.dataset import get_dataset
from .utils.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("experiments/0_Extra_experiments/config/config.ini")

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

ff_rho = config.getfloat("config", "ff_rho")
ff_beta = config.getfloat("config", "ff_beta")
conv_rho = config.getfloat("config", "conv_rho")
conv_beta = config.getfloat("config", "conv_beta")
spff_rho = config.getfloat("config", "spff_rho")
spff_beta = config.getfloat("config", "spff_beta")
spconv_rho = config.getfloat("config", "spconv_rho")
spconv_beta = config.getfloat("config", "spconv_beta")

ffdeltas = config.getfloat("config", "ffdeltas")
ffthetas = config.getfloat("config", "ffthetas")
convdeltas = config.getfloat("config", "convdeltas")
convthetas = config.getfloat("config", "convthetas")
input_size = 784
n_timesteps = config.getint("config", "n_timesteps")

# WARMING CONFIG
warming_iters = config.getint("config", "warming_iters")


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["feedforward", "conv", "spiking-feedforward", "spiking-conv"]

for model_name in model_types:
    for block in ["two-block", "multi-block"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"\n{'=' * 50}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'=' * 50}")

        #########################################
        # DATA
        if model_name in ["feedforward", "conv"]:
            train_loader = get_dataset(
                model_name=model_name,
                batch_size=batch_size_static,
                device=device,
                seed=seed,
            )

        else:
            train_loader = get_dataset(
                model_name=model_name,
                batch_size=batch_size_spiking,
                device=device,
                n_timesteps=n_timesteps,
                seed=seed,
            )

        #########################################
        # MODEL INSTANTIATION

        match model_name:
            case "feedforward":
                batch_size = batch_size_static
                admm_model = get_model(
                    model_name=model_name,
                    init="pytorch",
                    train_method="vectorized",
                    layer_order="sequential",
                    loss=ADMM_SSE(),
                    z_first=False,
                    block_method=block,
                )
            case "conv":
                batch_size = batch_size_static
                admm_model = get_model(
                    model_name=model_name,
                    init="pytorch",
                    train_method="vectorized",
                    layer_order="sequential",
                    loss=ADMM_SSE(),
                    z_first=False,
                    block_method=block,
                )
            case "spiking-feedforward":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method="decoupled",
                    time_order="backwards",
                    layer_order="sequential",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=False,
                    block_method=block,
                )
            case "spiking-conv":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method="decoupled",
                    time_order="backwards",
                    layer_order="sequential",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=False,
                    block_method=block,
                )

        #########################################
        # ADMM TRAINING LOOP

        m = ADMM_Metrics(admm_model)

        admm_times = []
        start_time = time.time()

        for epoch in range(epochs + 1):
            admm_model.fit(train_loader, warming=epoch < warming_iters)
            if epoch % 1 == 0:
                with torch.no_grad():
                    m.save_metrics()
                    elapsed_time = time.time() - start_time
                    admm_times.append(elapsed_time)
                    print(
                        f"Epoch {epoch} | W_norm: {admm_model.layers[0].W.norm().item():.2f}"
                    )
                    print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

        #########################################
        # SAVING RESULTS AND PLOTTING
        metrics = m.get_dic()
        metrics["architecture"] = model_name

        metrics_filename = f"experiments/0_Extra_experiments/results/Block_Methods/{model_name}/{block}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
