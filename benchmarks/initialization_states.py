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

architectures = ["linear", "conv", "spiking-linear", "spiking-conv"]
initializations = [
    "zeros",
    "xavier",
    "wrandom",
    "pytorch",
    "z-uniform",
    "s-uniform",
    "relaxed",
]
for model_name in architectures:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")
    for initialization in initializations:
        print(f"\nINITIALIZATION: {initialization}")
        # Reset seeds per model to guarantee identical environments
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #########################################
        # DATA
        if model_name in ["linear", "conv"]:
            train_loader = get_dataset(batch_size_static, 1, spiking=False, seed=seed)
        else:
            train_loader = get_dataset(
                batch_size_spiking,
                1,
                spiking=True,
                n_timesteps=n_timesteps,
                seed=seed,
            )

        #########################################
        # MODEL INSTANTIATION
        match model_name:
            case "linear":
                batch_size = batch_size_static
                admm_model = get_model(
                    model_name=model_name,
                    init=initialization,
                    train_method="vectorized",
                    layer_order="backwards",
                    loss=ADMM_SSE(),
                    z_first=False,
                )
            case "conv":
                batch_size = batch_size_static
                admm_model = get_model(
                    model_name=model_name,
                    init=initialization,
                    train_method="vectorized",
                    layer_order="backwards",
                    loss=ADMM_SSE(),
                    z_first=False,
                )
            case "spiking-linear":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init=initialization,
                    train_method="decoupled-backwards",
                    layer_order="backwards",
                    loss=ADMM_CrossEntropy_Taylor(),
                    z_first=False,
                )
            case "spiking-conv":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init=initialization,
                    train_method="decoupled-backwards",
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
        metrics["initialization"] = initialization
        metrics["warming_stop"] = warming_iters
        metrics["epochs"] = epochs
        metrics["seed"] = seed

        metrics_filename = f"benchmarks/results/initialization_states/{model_name}/{batch_size}/{initialization}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
