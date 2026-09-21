import configparser
import json
import os
import time

import torch

from src.admm import (
    ADMM_CrossEntropy_Taylor,
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

spff_rho = config.getfloat("config", "spff_rho")
spff_beta = config.getfloat("config", "spff_beta")
input_size = 784
n_timesteps = config.getint("config", "n_timesteps")

# WARMING CONFIG
warming_iters = config.getint("config", "warming_iters")


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#########################################
# DATA

train_loader = get_dataset_spiking_admm(
    model_name="spiking-feedforward",
    batch_size=batch_size_spiking,
    n_timesteps=n_timesteps,
    seed=seed,
)


admm_model = get_model(
    model_name="spiking-feedforward",
    init="s-uniform",
    train_method="vectorized",
    time_order="backwards",
    layer_order="backwards",
    loss=ADMM_CrossEntropy_Taylor(),
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
metrics["method"] = "Jacobi"
metrics["epochs"] = epochs
metrics["seed"] = seed

# ADMM Metrics
metrics["admm_time"] = admm_times

metrics_filename = "paper/results/admm_spiking_feedforward_jacobi/results.json"
os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

with open(metrics_filename, "w") as f:
    json.dump(metrics, f, indent=4)

# Free up memory before the next model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
