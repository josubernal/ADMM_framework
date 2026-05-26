import configparser
import json
import os

import torch

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.models import get_model
from src.admm import ADMM_SSE, ADMM_CrossEntropy_Taylor, ADMM_Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("benchmarks/config/config.ini")

seed = config.getint("config", "seed")

batch_size_static = config.getint("config", "batch_size_static")
batch_size_spiking = config.getint("config", "batch_size_spiking")
epochs = config.getint("config", "epochs")

n_timesteps = config.getint("config", "n_timesteps")
warming_iters = epochs // 2

model_types = ["feedforward", "conv", "spiking-feedforward", "spiking-conv"]
choices = [True, False]
for model_name in model_types:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")
    for z_first in choices:
        print(f"\nZ first: {z_first}")
        # Reset seeds per model to guarantee identical environments
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
                batch_size = batch_size_static
                admm_model = get_model(
                    model_name=model_name,
                    init="pytorch",
                    train_method="vectorized",
                    layer_order="backwards",
                    loss=ADMM_SSE(),
                    z_first=z_first,
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
                    z_first=z_first,
                    block_method="two-block",
                )
            case "spiking-feedforward":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method="decoupled-backwards",
                    layer_order="backwards",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=z_first,
                    block_method="two-block",
                )
            case "spiking-conv":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method="decoupled-backwards",
                    layer_order="backwards",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=z_first,
                    block_method="two-block",
                )

        #########################################
        # ADMM TRAINING LOOP
        m = ADMM_Metrics(admm_model)

        for epoch in range(epochs + 1):
            admm_model.fit(train_loader, warming=epoch < warming_iters)
            with torch.no_grad():
                m.save_metrics()
                print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

        #########################################
        # SAVING RESULTS
        metrics = m.get_dic()
        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["update_z_first"] = z_first
        metrics["warming_stop"] = warming_iters
        metrics["epochs"] = epochs
        metrics["seed"] = seed

        metrics_filename = f"benchmarks/results/update_z_first/{model_name}/{batch_size}/{z_first}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
