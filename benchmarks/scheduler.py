import configparser
import json
import os

import torch

from benchmarks.utils.dataset import get_dataset
from benchmarks.utils.models import get_model
from src.admm import ADMM_SSE, ADMM_CrossEntropy_Taylor, ADMM_Metrics, ADMM_Scheduler

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

for model_name in model_types:
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")
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
                z_first=False,
            )
        case "conv":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
            )
        case "spiking-feedforward":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled-backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
            )
        case "spiking-conv":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
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
    # 2ND MODEL INSTANTIATION (SCHEDULER RUN)
    #########################################

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    match model_name:
        case "feedforward":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
            )
        case "conv":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_SSE(),
                z_first=False,
            )
        case "spiking-feedforward":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled-backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
            )
        case "spiking-conv":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled-backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
            )

    #########################################
    m2 = ADMM_Metrics(admm_model)

    balancer = ADMM_Scheduler(
        admm_model, mu=10.0, tau=2.0, balance_freq=5, stop_epoch=int(epochs * 0.75)
    )

    print("\nTraining model with ADMM (WITH SCHEDULER)...")
    for epoch in range(epochs + 1):
        balancer.capture_state()
        admm_model.fit(train_loader, warming=epoch < warming_iters)
        if epoch % 1 == 0:
            with torch.no_grad():
                m2.save_metrics()
                print(f"Epoch [{epoch:3d}/{epochs}] | {m2}")
                # 3. Read the Primal Residuals (These are LISTS of per-layer residuals)
                primal_rho_list = m2.metrics["preactivation_constraint_sum"][-1]
                primal_beta_list = m2.metrics["activation_constraint_sum"][-1]

                # 4. Balance the network!
                balancer.step(primal_rho_list, primal_beta_list)

    #########################################
    # SAVING RESULTS

    metrics_no_sched = m.get_dic()
    metrics_no_sched["architecture"] = model_name
    metrics_no_sched["batch_size"] = batch_size
    metrics_no_sched["warming_stop"] = warming_iters
    metrics_no_sched["epochs"] = epochs
    metrics_no_sched["seed"] = seed

    metrics_filename1 = f"benchmarks/results/scheduler/{model_name}/{batch_size}/no-scheduler/results.json"
    os.makedirs(os.path.dirname(metrics_filename1), exist_ok=True)
    with open(metrics_filename1, "w") as f:
        json.dump(metrics_no_sched, f, indent=4)

    # Save the SECOND loop's metrics (With Scheduler)
    metrics_sched = m2.get_dic()
    metrics_sched["architecture"] = model_name
    metrics_sched["batch_size"] = batch_size
    metrics_sched["epochs"] = epochs
    metrics_sched["seed"] = seed

    metrics_filename2 = (
        f"benchmarks/results/scheduler/{model_name}/{batch_size}/scheduler/results.json"
    )
    os.makedirs(os.path.dirname(metrics_filename2), exist_ok=True)
    with open(metrics_filename2, "w") as f:
        json.dump(metrics_sched, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
