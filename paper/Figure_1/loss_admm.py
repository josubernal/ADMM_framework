import configparser
import json
import os
import time

import torch

from src.admm import (
    ADMM_SSE,
    ADMM_CrossEntropy,
    ADMM_CrossEntropy_Taylor,
    ADMM_Hinge,
    ADMM_Metrics,
)

from ..utils.dataset import get_dataset_spiking_admm
from ..utils.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("paper/config/config.ini")

seed = config.getint("config", "seed")

batch_size_spiking = config.getint("config", "batch_size_spiking")
epochs = config.getint("config", "epochs")
hidden_size_spiking = config.getint("config", "hidden_size_spiking")
hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")

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


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models

for loss_f in [
    ADMM_SSE(),
    ADMM_Hinge(),
    ADMM_CrossEntropy(),
    ADMM_CrossEntropy_Taylor(),
]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {loss_f}")
    print(f"{'=' * 50}")

    #########################################
    # DATA

    train_loader = get_dataset_spiking_admm(
        model_name="spiking-feedforward",
        batch_size=batch_size_spiking,
        n_timesteps=n_timesteps,
        seed=seed,
    )

    batch_size = batch_size_spiking
    admm_model = get_model(
        model_name="spiking-feedforward",
        init="s-uniform",
        train_method="decoupled",
        time_order="backwards",
        layer_order="backwards",
        loss=loss_f,
        z_first=True,
        block_method="two-block",
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
                print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

    ############################
    # SAVING RESULTS AND PLOTTING
    metrics = m.get_dic()
    metrics["loss_function"] = loss_f
    metrics["epochs"] = epochs
    metrics["seed"] = seed

    # ADMM Metrics
    metrics["admm_time"] = admm_times

    metrics_filename = f"/paper/results/admm_spiking_feedforward_{loss_f}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
