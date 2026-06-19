import configparser
import json
import os

import torch

from src.admm import ADMM_CrossEntropy_Taylor, ADMM_Metrics

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


methods = [
    "zeros",
    "xavier",
    "wrandom",
    "pytorch",
    "z-uniform",
    "s-uniform",
    "relaxed",
]

for method in methods:
    print(f"\nINITIALIZATION STATES: {method}")
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
        init=method,
        train_method="decoupled",
        time_order="backwards",
        layer_order="backwards",
        loss=ADMM_CrossEntropy_Taylor(),
        z_first=True,
        block_method="two-block",
        lagrange_config="last-only",
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
    # SAVING RESULTS
    metrics = m.get_dic()
    metrics["method"] = str(method)
    metrics["seed"] = seed

    metrics_filename = f"experiments/2_Ablation_study/results/initialization_states/{method}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
