import json
import os
import random

import torch
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_CrossEntropy_Taylor,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_SpikingFeedForward,
)

from .utils.dataset import get_dataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 1000
    WARMING_ITERS = 300
    N_TIMESTEPS = 150
    BATCH_SIZE = 200
    HIDDEN_DIMS = 512
    RHO = 1
    BETA = 0.1
    DELTAS = 0.95
    THETAS = 1

    seed = 8281003564

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = get_dataset(
        batch_size=BATCH_SIZE,
        device=device,
        n_timesteps=N_TIMESTEPS,
        seed=seed,
    )
    ###################################################################
    # SOTA
    ###################################################################
    SOTA_hidden_layer_config = ADMM_LayerConfig(
        rho=RHO, beta=BETA, deltas=DELTAS, thetas=THETAS, use_bias=False
    )
    SOTA_out_layer_config = ADMM_LayerConfig(
        rho=RHO,
        beta=BETA,
        deltas=DELTAS,
        thetas=THETAS,
        use_bias=False,
        use_lagrange=True,
    )
    SOTA_layers = nn.ModuleList(
        [
            ADMM_SpikingFeedForward(
                in_f=34 * 34 * 2,
                out_f=HIDDEN_DIMS,
                h=ADMM_Heaviside(THETAS),
                config=SOTA_hidden_layer_config,
            ),
            ADMM_SpikingFeedForward(
                in_f=HIDDEN_DIMS,
                out_f=10,
                h=None,
                config=SOTA_out_layer_config,
                use_reset=False,
            ),
        ]
    )
    SOTA_config = ADMM_Config(
        init="s-uniform",
        train_method="unrolled",
        time_order="random-last",
        layer_order="random-last",
        update_z_first=False,
        block_method="multi-block",
    )

    SOTA_model = ADMM(
        SOTA_layers, loss_f=ADMM_SSE(), T=N_TIMESTEPS, config=SOTA_config
    ).to(device)

    m1 = ADMM_Metrics(SOTA_model)
    print("Starting SOTA model training...")
    for epoch in range(EPOCHS + 1):
        SOTA_model.fit(train_loader, warming=epoch < WARMING_ITERS)
        if epoch % 1 == 0:
            with torch.no_grad():
                m1.save_metrics()
                print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m1}")

    #########################################
    # SAVING RESULTS
    metrics = m1.get_dic()

    metrics_filename = "experiments/1_SOTA_evaluation/results/SOTA/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ###################################################################
    # NEW
    ###################################################################
    seed = 8281003564
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = get_dataset(
        batch_size=BATCH_SIZE,
        device=device,
        n_timesteps=N_TIMESTEPS,
        seed=seed,
    )

    hidden_layer_config = ADMM_LayerConfig(
        rho=RHO, beta=BETA, deltas=DELTAS, thetas=THETAS, use_bias=False
    )
    out_layer_config = ADMM_LayerConfig(
        rho=RHO,
        beta=BETA,
        deltas=DELTAS,
        thetas=THETAS,
        use_bias=False,
        use_lagrange=True,
    )
    layers = nn.ModuleList(
        [
            ADMM_SpikingFeedForward(
                in_f=34 * 34 * 2,
                out_f=HIDDEN_DIMS,
                h=ADMM_Heaviside(THETAS),
                config=hidden_layer_config,
            ),
            ADMM_SpikingFeedForward(
                in_f=HIDDEN_DIMS,
                out_f=10,
                h=None,
                config=out_layer_config,
                use_reset=False,
            ),
        ]
    )
    config = ADMM_Config(
        init="s-uniform",
        train_method="decoupled",
        time_order="backwards",
        layer_order="backwards",
        update_z_first=True,
        block_method="two-block",
    )

    model = ADMM(
        layers, loss_f=ADMM_CrossEntropy_Taylor(), T=N_TIMESTEPS, config=config
    ).to(device)

    m2 = ADMM_Metrics(model)
    print("Starting NEW model training...")
    for epoch in range(EPOCHS + 1):
        model.fit(train_loader, warming=epoch < WARMING_ITERS)
        if epoch % 1 == 0:
            with torch.no_grad():
                m2.save_metrics()
                print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m2}")

    #########################################
    # SAVING RESULTS
    metrics = m2.get_dic()

    metrics_filename = "experiments/1_SOTA_evaluation/results/NEW/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
