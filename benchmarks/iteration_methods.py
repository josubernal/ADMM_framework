import configparser
import json
import os
import time

import torch
import torch.nn as nn

from admm import (
    ADMM,
    ADMM_Config,
    ADMM_CrossEntropy_Taylor,
    ADMM_Flatten,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_SpikingConv2d,
    ADMM_SpikingLinear,
)
from utils.dataset import get_data


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read("benchmarks/config/config.ini")
    # ==========================================
    # 1. GLOBAL SETUP & SEEDING
    # ==========================================
    seed = config.getint("config", "seed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = config.getint("config", "batch_size_spiking")
    epochs = config.getint("config", "epochs")
    hidden_size_spiking = config.getint("config", "hidden_size_spiking")
    hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")
    k = config.getint("config", "k")
    p = config.getint("config", "p")
    s = config.getint("config", "s")
    n_timesteps = config.getint("config", "n_timesteps")
    accuracy_threshold = config.getint("config", "acc_threshold")
    min_warming_iters = config.getint("config", "min_warming_iters")
    max_warming_iters = config.getint("config", "max_warming_iters")
    primal_delta_limit = config.getfloat("config", "primal_threshold")

    prev_primal_residual = float("inf")

    # Standard Scalar Hyperparameters
    linear_rho = config.getfloat("config", "splinear_rho")
    linear_beta = config.getfloat("config", "splinear_beta")
    conv_rho = config.getfloat("config", "spconv_rho")
    conv_beta = config.getfloat("config", "spconv_beta")
    deltas = config.getfloat("config", "deltas")
    thetas = config.getfloat("config", "thetas")

    # ==========================================
    # 3. BENCHMARK LOOPS
    # ==========================================
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

    print("\n" + "=" * 80)
    print(f"  ADMM 5-METHOD ARCHITECTURE BENCHMARK (Device: {device.type.upper()})")
    print("=" * 80)

    for arch in architectures:
        print(f"\n{'#' * 60}")
        print(f"### EVALUATING ARCHITECTURE: {arch.upper()}")
        print(f"{'#' * 60}")

        for method in methods:
            # 1. Reset seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            raw_data, targets = get_data(
                batch_size,
                spiking=True,
                device=device,
                n_timesteps=n_timesteps,
                seed=seed,
            )
            targets = torch.nn.functional.one_hot(
                targets.long(), num_classes=10
            ).float()

            if arch == "spiking-linear":
                arch_data = raw_data.view(
                    raw_data.size(0), raw_data.size(1), -1
                ).permute(1, 0, 2)
            else:
                arch_data = raw_data.permute(1, 0, 2, 3, 4)

            is_warming = True
            prev_primal_residual = float("inf")
            warming_stop = None
            print(f"\n---> Testing Method: {method.upper()}")

            if arch == "spiking-linear":
                # 34 * 34 * 2 = 2312
                rho = linear_rho
                beta = linear_beta
                layer_config = ADMM_LayerConfig(
                    rho=rho, beta=beta, thetas=thetas, deltas=deltas
                )
                layers = nn.ModuleList(
                    [
                        ADMM_SpikingLinear(
                            in_f=2312,
                            out_f=hidden_size_spiking,
                            h=ADMM_Heaviside(),
                            config=layer_config,
                        ),
                        ADMM_SpikingLinear(
                            in_f=hidden_size_spiking,
                            out_f=10,
                            h=None,
                            config=layer_config,
                            use_reset=False,
                        ),
                    ]
                )
            else:
                rho = conv_rho
                beta = conv_beta
                layer_config = ADMM_LayerConfig(
                    rho=rho, beta=beta, thetas=thetas, deltas=deltas, use_fft=False
                )
                spatial = int(calc_spatial_out(34, k, p, s))
                lin_in = hidden_channels_spiking * spatial * spatial
                layers = nn.ModuleList(
                    [
                        ADMM_SpikingConv2d(
                            in_c=2,
                            out_c=hidden_channels_spiking,
                            k=k,
                            p=p,
                            s=s,
                            h=ADMM_Heaviside(),
                            config=layer_config,
                            padding_mode="zeros",
                        ),
                        ADMM_SpikingLinear(
                            in_f=lin_in,
                            pool_op=ADMM_Flatten(),
                            out_f=10,
                            h=None,
                            config=layer_config,
                            use_reset=False,
                        ),
                    ]
                )

            config = ADMM_Config(train_method=method, init="s-uniform")
            model = ADMM(
                layers,
                T=n_timesteps,
                loss_f=ADMM_CrossEntropy_Taylor(),
                config=config,
            ).to(device)

            m = ADMM_Metrics(model)
            model._init_states(arch_data)

            # ---------------------------------------------------------
            # TRAINING LOOP
            # ---------------------------------------------------------
            start_time = time.time()

            for epoch in range(epochs + 1):
                model.fit(arch_data, targets, warming=is_warming)

                with torch.no_grad():
                    m.save_metrics(arch_data, targets)

                    print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

                    # --- DYNAMIC WARMING LOGIC ---
                    current_primal = m.metrics["primal_residual"][-1]
                    accuracy = m.metrics["accuracy"][-1]
                    primal_residual_delta = abs(prev_primal_residual - current_primal)
                    prev_primal_residual = current_primal

                    if is_warming:
                        hit_accuracy = (accuracy > accuracy_threshold) and (
                            epoch > min_warming_iters
                        )
                        hit_time_limit = epoch >= max_warming_iters

                        if hit_accuracy or hit_time_limit:
                            if (
                                primal_residual_delta < primal_delta_limit
                                or hit_time_limit
                            ):
                                reason = (
                                    "Accuracy/Delta Target Met"
                                    if hit_accuracy
                                    else "Max Epochs Reached"
                                )
                                print(
                                    f"--- STOPPING WARMING at Epoch {epoch} ({reason}) ---"
                                )
                                is_warming = False
                                warming_stop = epoch

            end_time = time.time()
            running_time = end_time - start_time
            print(
                f"[Finished {method} on {arch.upper()} in {running_time:.2f} seconds]"
            )

            # ==========================================
            # 4. SAVING RESULTS TO JSON
            # ==========================================
            metrics = m.get_dic()
            metrics["running_time"] = running_time
            metrics["architecture"] = arch
            metrics["method"] = method
            metrics["batch_size"] = batch_size
            metrics["epochs"] = epochs
            metrics["seed"] = seed

            metrics_filename = f"benchmarks/results/iteration_methods/{arch}/{batch_size}/{method}/results.json"
            os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

            with open(metrics_filename, "w") as f:
                json.dump(metrics, f, indent=4)

            # Free up memory before the next model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
