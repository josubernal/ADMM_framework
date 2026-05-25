import configparser
import json
import os

import torch
import torch.nn as nn

from benchmarks.utils.dataset import get_dataset
from src.admm import (
    ADMM,
    ADMM_Conv2d,
    ADMM_FeedForward,
    ADMM_Flatten,
    ADMM_Metrics,
    ADMM_ReLU,
)
from src.admm.dataclasses import ADMM_Config, ADMM_LayerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("benchmarks/config/config.ini")

seed = config.getint("config", "seed")

batch_size_static = config.getint("config", "batch_size_static")
epochs = config.getint("config", "epochs")
hidden_size_static = 512
hidden_channels_static = 2
k = config.getint("config", "k")
p = (k - 1) // 2  # To prevent shrinkage
s = 1

ff_rho = config.getfloat("config", "ff_rho")
ff_beta = config.getfloat("config", "ff_beta")
conv_rho = config.getfloat("config", "conv_rho")
conv_beta = config.getfloat("config", "conv_beta")
max_layers = config.getint("config", "max_layers")
input_size = 784

warming_iters = epochs // 2


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


########################################

model_types = ["feedforward", "conv"]

batch_size = batch_size_static

for model_name in model_types:
    train_loader = get_dataset(model_name, batch_size_static, 1, seed=seed)
    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")
    for layers in range(1, max_layers + 1):
        print(f"\nHidden layers:{layers}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #########################################
        # MODEL INSTANTIATION
        layer_list = []
        match model_name:
            case "feedforward":
                layer_config = ADMM_LayerConfig(
                    rho=ff_rho,
                    beta=ff_beta,
                    use_bias=True,
                )
                layer_list.append(
                    ADMM_FeedForward(
                        in_f=input_size,
                        out_f=hidden_size_static,
                        h=ADMM_ReLU(),
                        config=layer_config,
                    )
                )
                for _ in range(layers - 1):
                    layer_list.append(
                        ADMM_FeedForward(
                            in_f=hidden_size_static,
                            out_f=hidden_size_static,
                            h=ADMM_ReLU(),
                            config=layer_config,
                        )
                    )

                layer_list.append(
                    ADMM_FeedForward(
                        in_f=hidden_size_static,
                        out_f=10,
                        h=ADMM_ReLU(),
                        config=layer_config,
                    )
                )

                ff_layers = nn.ModuleList(layer_list)
                config = ADMM_Config(
                    init="pytorch",
                    train_method="vectorized",
                )
                admm_model = ADMM(ff_layers, config=config).to(device)

            case "conv":
                current_spatial = 28
                layer_config = ADMM_LayerConfig(
                    rho=conv_rho, beta=conv_beta, use_bias=True, use_fft=True
                )
                layer_list.append(
                    ADMM_Conv2d(
                        in_c=1,
                        out_c=hidden_channels_static,
                        k=k,
                        p=p,
                        s=s,
                        h=ADMM_ReLU(),
                        config=layer_config,
                        padding_mode="circular",
                    )
                )
                current_spatial = int(calc_spatial_out(current_spatial, k, p, s))

                for _ in range(layers - 1):
                    layer_list.append(
                        ADMM_Conv2d(
                            in_c=hidden_channels_static,
                            out_c=hidden_channels_static,
                            k=k,
                            p=p,
                            s=s,
                            h=ADMM_ReLU(),
                            config=layer_config,
                            padding_mode="circular",
                        )
                    )
                    current_spatial = int(calc_spatial_out(current_spatial, k, p, s))

                lin_in_dim = hidden_channels_static * current_spatial * current_spatial
                layer_list.append(
                    ADMM_FeedForward(
                        in_f=lin_in_dim,
                        out_f=10,
                        h=ADMM_ReLU(),
                        pool_op=ADMM_Flatten(),
                        config=layer_config,
                    )
                )
                conv_layers = nn.ModuleList(layer_list)
                config = ADMM_Config(
                    init="pytorch",
                    train_method="vectorized",
                )
                admm_model = ADMM(conv_layers, config=config).to(device)

        #########################################
        # ADMM TRAINING LOOP
        m = ADMM_Metrics(admm_model)

        for epoch in range(epochs + 1):
            admm_model.fit(train_loader, warming=epoch < warming_iters)
            if epoch % 1 == 0:
                with torch.no_grad():
                    m.save_metrics()
                    print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

        # SAVING RESULTS
        metrics = m.get_dic()
        metrics["architecture"] = model_name
        metrics["batch_size"] = batch_size
        metrics["layers"] = layers
        metrics["warming_stop"] = warming_iters
        metrics["epochs"] = epochs
        metrics["seed"] = seed

        metrics_filename = f"benchmarks/results/static_breakpoint/{model_name}/{batch_size}/{layers}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
