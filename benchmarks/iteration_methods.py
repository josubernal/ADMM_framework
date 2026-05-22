import configparser
import json
import os

import torch

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.models import get_model
from src.admm import ADMM_CrossEntropy_Taylor, ADMM_Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()
config.read("benchmarks/config/config.ini")

seed = config.getint("config", "seed")
batch_size = config.getint("config", "batch_size_spiking")
epochs = config.getint("config", "epochs")
hidden_size_spiking = config.getint("config", "hidden_size_spiking")
hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")
n_timesteps = config.getint("config", "n_timesteps")
warming_iters = epochs // 2

architectures = ["spiking-linear", "spiking-conv"]
methods = [
    "unrolled-sequential",
    "unrolled-random",
    "unrolled-backwards",
    "decoupled-sequential",
    "decoupled-random",
    "decoupled-backwards",
    "vectorized",
]

for model_name in architectures:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")

    for method in methods:
        print(f"\nMETHOD: {method}")
        # Reset seeds per model to guarantee identical environments
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #########################################
        # DATA
        train_loader = get_dataset(
            batch_size,
            1,
            spiking=True,
            n_timesteps=n_timesteps,
            seed=seed,
        )

        #########################################
        # MODEL INSTANTIATION
        match model_name:
            case "spiking-linear":
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method=method,
                    layer_order="backwards",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=False,
                )
            case "spiking-conv":
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method=method,
                    layer_order="backwards",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=False,
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
        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["iteration_method"] = method
        metrics["warming_stop"] = warming_iters
        metrics["epochs"] = epochs
        metrics["seed"] = seed

        metrics_filename = f"benchmarks/results/initialization_states/{model_name}/{batch_size}/{method}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
