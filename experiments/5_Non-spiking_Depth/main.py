import configparser
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.admm import (
    ADMM,
    ADMM_FeedForward,
    ADMM_Metrics,
    ADMM_ReLU,
)
from src.admm.dataclasses import ADMM_Config, ADMM_LayerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("benchmarks/config/config.ini")

seed = 8281003564

BATCHSIZE = 60000
EPOCHS = 1000
WARMING_ITERS = 300
HIDDEN_SIZE = 1000

RHO = 0.1
BETA = 0.1
MAX_LAYERS = 5
input_size = 784


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    trainset,
    batch_size=BATCHSIZE,
    shuffle=True,
    drop_last=True,
)
data, targets = next(iter(train_loader))
data = data.view(data.size(0), -1)
data = data.to(device).float()
targets = targets.to(device)
data += 0.01 * torch.randn_like(data)

train_loader = [(data, targets)]


for layers in range(1, MAX_LAYERS + 1):
    print(f"\nHidden layers:{layers}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #########################################
    # MODEL INSTANTIATION
    layer_list = []

    layer_config = ADMM_LayerConfig(
        rho=RHO,
        beta=BETA,
        use_bias=True,
    )
    layer_list.append(
        ADMM_FeedForward(
            in_f=input_size,
            out_f=HIDDEN_SIZE,
            h=ADMM_ReLU(),
            config=layer_config,
        )
    )
    for _ in range(layers - 1):
        layer_list.append(
            ADMM_FeedForward(
                in_f=HIDDEN_SIZE,
                out_f=HIDDEN_SIZE,
                h=ADMM_ReLU(),
                config=layer_config,
            )
        )

    layer_list.append(
        ADMM_FeedForward(
            in_f=HIDDEN_SIZE,
            out_f=10,
            h=ADMM_ReLU(),
            config=layer_config,
        )
    )

    model_layers = nn.ModuleList(layer_list)
    config = ADMM_Config(
        init="pytorch",
        train_method="vectorized",
    )
    admm_model = ADMM(model_layers, config=config).to(device)

    #########################################
    # ADMM TRAINING LOOP
    m = ADMM_Metrics(admm_model)

    for epoch in range(EPOCHS + 1):
        admm_model.fit(train_loader, warming=epoch < WARMING_ITERS)
        if epoch % 1 == 0:
            with torch.no_grad():
                m.save_metrics()
                print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m}")

    # SAVING RESULTS
    metrics = m.get_dic()
    metrics["layers"] = layers

    metrics_filename = f"experiments/5_Non-spiking_Depth/results/{layers}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
