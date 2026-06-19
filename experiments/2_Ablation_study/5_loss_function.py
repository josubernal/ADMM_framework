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

from .utils.dataset import get_dataset
from .utils.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()
config.read("experiments/2_Ablation_study/config/config.ini")

seed = config.getint("config", "seed")

batch_size = config.getint("config", "batch_size")
epochs = config.getint("config", "epochs")

n_timesteps = config.getint("config", "n_timesteps")
warming_iters = config.getint("config", "warming_iters")


for loss in [
    ADMM_SSE(),
    ADMM_Hinge(),
    ADMM_CrossEntropy(),
    ADMM_CrossEntropy_Taylor(),
]:
    print(f"\nLOSS FUNCTION: {loss}")
    # Reset seeds per model to guarantee identical environments
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #########################################
    # DATA
    train_loader = get_dataset(
        batch_size=batch_size,
        device=device,
        n_timesteps=n_timesteps,
        seed=seed,
    )

    #########################################
    # MODEL INSTANTIATION
    admm_model = get_model(
        init="s-uniform",
        train_method="decoupled",
        time_order="backwards",
        layer_order="backwards",
        loss=loss,
        z_first=True,
        block_method="two-block",
        lagrange_config="last_only",
    )

    #########################################
    # ADMM TRAINING LOOP
    m = ADMM_Metrics(admm_model)
    start_time = time.time()
    for epoch in range(epochs + 1):
        admm_model.fit(train_loader, warming=epoch < warming_iters)
        if epoch % 1 == 0:
            with torch.no_grad():
                m.save_metrics()
                print(f"Epoch [{epoch:3d}/{epochs}] | {m}")
    end_time = time.time()
    #########################################
    # SAVING RESULTS
    metrics = m.get_dic()
    metrics["method"] = str(loss)
    metrics["seed"] = seed
    metrics["running_time"] = end_time - start_time

    metrics_filename = (
        f"experiments/2_Ablation_study/results/loss_functions/{loss}/results.json"
    )
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
