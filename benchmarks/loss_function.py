import configparser
import json
import os

import torch

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.models import get_model
from src.admm import (
    ADMM_SSE,
    ADMM_CrossEntropy,
    ADMM_CrossEntropy_Taylor,
    ADMM_Hinge,
    ADMM_Metrics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("benchmarks/config/config.ini")

seed = config.getint("config", "seed")

batch_size_static = config.getint("config", "batch_size_static")
batch_size_spiking = config.getint("config", "batch_size_spiking")
epochs = config.getint("config", "epochs")

n_timesteps = config.getint("config", "n_timesteps")
warming_iters = epochs // 2

########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["feedforward", "conv", "spiking-feedforward", "spiking-conv"]

for model_name in model_types:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")
    for loss in [
        ADMM_SSE(),
        ADMM_Hinge(),
        ADMM_CrossEntropy(),
        ADMM_CrossEntropy_Taylor(),
    ]:
        print(f"LOSS: {loss}")
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
                    loss=loss,
                    z_first=False,
                )
            case "conv":
                batch_size = batch_size_static
                admm_model = get_model(
                    model_name=model_name,
                    init="pytorch",
                    train_method="vectorized",
                    layer_order="backwards",
                    loss=loss,
                    z_first=False,
                )
            case "spiking-feedforward":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method="decoupled-backwards",
                    layer_order="backwards",
                    loss=loss,
                    z_first=False,
                )
            case "spiking-conv":
                batch_size = batch_size_spiking
                admm_model = get_model(
                    model_name=model_name,
                    init="s-uniform",
                    train_method="decoupled-backwards",
                    layer_order="backwards",
                    loss=loss,
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
        metrics["warming_stop"] = warming_iters
        metrics["epochs"] = epochs
        metrics["seed"] = seed
        metrics["loss_function"] = str(loss)

        metrics_filename = f"benchmarks/results/loss_function/{model_name}/{batch_size}/{str(loss)}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        # Free up memory before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
