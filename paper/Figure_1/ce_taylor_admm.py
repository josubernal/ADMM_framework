import configparser
import json
import os
import time

import torch

from src.admm import (
    ADMM_CrossEntropy_Taylor,
    ADMM_Metrics,
)

from ..utils.dataset import (
    get_dataset_spiking_admm,
    get_dataset_spiking_only_pad,
    get_dataset_spiking_wo_noise,
)
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

print("Starting pure DataLoader test...")

iterator = iter(train_loader)

for i in range(3):
    t0 = time.perf_counter()

    inputs, labels = next(iterator)

    t1 = time.perf_counter()

    print(f"DataLoader batch {i}: {t1 - t0:.3f}s | inputs={inputs.shape}")

train_loader = get_dataset_spiking_wo_noise(
    model_name="spiking-feedforward",
    batch_size=batch_size_spiking,
    n_timesteps=n_timesteps,
    seed=seed,
)


iterator = iter(train_loader)

for i in range(3):
    t0 = time.perf_counter()

    inputs, labels = next(iterator)

    t1 = time.perf_counter()

    print(f"DataLoader batch {i}: {t1 - t0:.3f}s | inputs={inputs.shape}")

train_loader = get_dataset_spiking_only_pad(
    model_name="spiking-feedforward",
    batch_size=batch_size_spiking,
    n_timesteps=n_timesteps,
    seed=seed,
)


iterator = iter(train_loader)

for i in range(3):
    t0 = time.perf_counter()

    inputs, labels = next(iterator)

    t1 = time.perf_counter()

    print(f"DataLoader only pad batch {i}: {t1 - t0:.3f}s | inputs={inputs.shape}")

admm_model = get_model(
    model_name="spiking-feedforward",
    init="s-uniform",
    train_method="decoupled",
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
            m.save_metrics(train_loader)
            elapsed_time = time.time() - start_time
            admm_times.append(elapsed_time)
            print(f"Epoch [{epoch:3d}/{epochs}] | {m}")
admm_model.close()  # Close the model to ensure all resources are released

############################
# SAVING RESULTS AND PLOTTING
metrics = m.get_dic()
metrics["loss_function"] = "ADMM_CrossEntropy_Taylor"
metrics["epochs"] = epochs
metrics["seed"] = seed

# ADMM Metrics
metrics["admm_time"] = admm_times

metrics_filename = (
    "paper/results/admm_spiking_feedforward_cross_entropy_taylor/results.json"
)
os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

with open(metrics_filename, "w") as f:
    json.dump(metrics, f, indent=4)

# Free up memory before the next model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
