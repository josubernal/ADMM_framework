import configparser
import itertools
import json
import os
import random
import time

import torch
import torch.nn as nn

from experiments.utils.dataset import get_dataset
from src.admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_Conv2d,
    ADMM_CrossEntropy_Taylor,
    ADMM_FeedForward,
    ADMM_Flatten,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_ReLU,
    ADMM_SpatialPool,
    ADMM_SpikingConv2d,
    ADMM_SpikingFeedForward,
)


def is_valid_combination(params: dict) -> bool:
    is_spiking = params.get("model").split("-")[0] == "spiking"
    print(is_spiking)
    train_method = params.get("train_method", "vectorized")
    if not is_spiking and train_method != "vectorized":
        return False
    return True


def parse_value(v):
    """Smartly parse strings from the INI file into their correct Python types."""
    v = v.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


if __name__ == "__main__":
    # --- Base Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = configparser.ConfigParser()
    config.read("scripts/config/config.ini")

    # --- Smart Config Parsing ---
    grid_params = {}
    if config.has_section("config"):
        for key, val in config.items("config"):
            # Split by comma to support both static (length 1) and grid (length > 1)
            grid_params[key] = [parse_value(v) for v in val.split(",")]
    else:
        raise ValueError("Config file must contain a [config] section.")

    # Generate all combinations
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combinations = [combo for combo in combinations if is_valid_combination(combo)]

    is_static_run = len(combinations) == 1

    print(f"Total configurations to evaluate: {len(combinations)}")
    if not is_static_run:
        print("Grid Parameters detected.")

    # To store metrics for plotting if it's a static run
    final_metrics = {}
    dynamic_keys = [k for k, v in grid_params.items() if len(v) > 1]
    # ==========================================
    # EXECUTION LOOP
    # ==========================================
    for idx, cfg in enumerate(combinations):
        print("\n=======================================================")
        if is_static_run:
            print(
                f"Running Static Config: {cfg['model']} | Train Method: {cfg['train_method']}"
            )
        else:
            print(
                f"[{idx + 1}/{len(combinations)}] Running Grid Config: {[f'{k}:{cfg[k]}' for k in dynamic_keys]}"
            )
        print("=======================================================")

        # Ensure perfect reproducibility for each run
        seed = cfg.get("seed", 8281003564)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Extract variables
        use_double = cfg.get("use_double", False)
        model_name = cfg.get("model", "feedforward")
        epochs = cfg.get("epochs", 20)
        warming_iters = cfg.get("warming_iters", 6)
        batch_size = cfg.get("batch_size", 50)
        n_batches = cfg.get("n_batches", 1)
        init = cfg.get("init", "s-uniform")
        bias = cfg.get("bias", False)
        train_method = cfg.get("train_method", "unrolled")
        rho = cfg.get("rho", 1.0)
        beta = cfg.get("beta", 0.1)
        thetas = cfg.get("thetas", 0.3)
        deltas = cfg.get("deltas", 0.95)
        n_timesteps = cfg.get("n_timesteps", 150)
        num_layers = cfg.get("layers", 2)
        use_fft = cfg.get("use_fft", True)
        padding_mode = cfg.get("padding_mode", "circular")

        if use_double:
            torch.set_default_dtype(torch.float64)
            current_dtype = torch.float64
        else:
            torch.set_default_dtype(torch.float32)
            current_dtype = torch.float32

        # ---------------------------------------------------------
        # 1. LOAD DATASET
        # ---------------------------------------------------------

        if model_name in ["feedforward", "conv"]:
            train_loader = get_dataset(model_name, batch_size, n_batches, seed=seed)
        else:
            train_loader = get_dataset(
                model_name,
                batch_size,
                n_batches,
                n_timesteps=n_timesteps,
                seed=seed,
            )

        # ---------------------------------------------------------
        # 2. INITIALIZE ARCHITECTURE
        # ---------------------------------------------------------
        hidden_dims = int(cfg.get("hidden_dims", 100))
        hidden_channels = int(cfg.get("hidden_channels", 4))

        layer_config = ADMM_LayerConfig(
            beta=beta,
            rho=rho,
            thetas=thetas,
            deltas=deltas,
            use_bias=bias,
            use_fft=use_fft,
        )
        match model_name:
            case "spiking-feedforward":
                if num_layers == 2:
                    layers = nn.ModuleList(
                        [
                            ADMM_SpikingFeedForward(
                                in_f=34 * 34 * 2,
                                out_f=hidden_dims,
                                h=ADMM_Heaviside(),
                                config=layer_config,
                            ),
                            ADMM_SpikingFeedForward(
                                in_f=hidden_dims,
                                out_f=10,
                                h=None,
                                config=layer_config,
                                use_reset=False,
                            ),
                        ]
                    )
                elif num_layers == 3:
                    mid_dims = hidden_dims // 2
                    layers = nn.ModuleList(
                        [
                            ADMM_SpikingFeedForward(
                                in_f=34 * 34 * 2,
                                out_f=hidden_dims,
                                h=ADMM_Heaviside(),
                                config=layer_config,
                            ),
                            ADMM_SpikingFeedForward(
                                in_f=hidden_dims,
                                out_f=mid_dims,
                                h=ADMM_Heaviside(),
                                config=layer_config,
                            ),
                            ADMM_SpikingFeedForward(
                                in_f=mid_dims,
                                out_f=10,
                                h=None,
                                config=layer_config,
                                use_reset=False,
                            ),
                        ]
                    )

            case "spiking-conv":
                k = cfg.get("kernel_size", 5)
                p = cfg.get("padding", 2)
                s = cfg.get("stride", 1)  # FFTs require stride=1
                pool_h, pool_w = 8, 8
                # N-MNIST is 34x34. Calculate the exact output size of the Conv layer
                if num_layers == 2:
                    # Dynamically calculate the 18x18 output
                    spatial_out = int(calc_spatial_out(34, k, p, s))
                    # 2 * 18 * 18 = 648
                    lin_in = hidden_channels * spatial_out * spatial_out

                    layers = nn.ModuleList(
                        [
                            ADMM_SpikingConv2d(
                                in_c=2,
                                out_c=hidden_channels,
                                k=k,
                                p=p,
                                s=s,
                                h=ADMM_Heaviside(),
                                config=layer_config,
                                padding_mode="zeros",
                            ),
                            # Passing 648 to in_f and using Flatten
                            ADMM_SpikingFeedForward(
                                in_f=lin_in,
                                out_f=10,
                                h=None,
                                pool_op=ADMM_Flatten(),
                                config=layer_config,
                                use_reset=False,
                            ),
                        ]
                    )
                elif num_layers == 3:
                    mid_p = k // 2
                    mid_c = hidden_channels // 2
                    lin_in = mid_c * pool_h * pool_w
                    layers = nn.ModuleList(
                        [
                            ADMM_SpikingConv2d(
                                in_c=2,
                                out_c=hidden_channels,
                                k=k,
                                p=p,
                                s=1,
                                h=ADMM_Heaviside(),
                                config=layer_config,
                                use_reset=False,
                                padding_mode="circular",
                            ),
                            ADMM_SpikingConv2d(
                                in_c=hidden_channels,
                                out_c=mid_c,
                                k=k,
                                p=mid_p,
                                s=s,
                                h=ADMM_Heaviside(),
                                config=layer_config,
                                use_reset=False,
                                padding_mode="zeros",
                            ),
                            ADMM_SpikingFeedForward(
                                in_f=lin_in,
                                out_f=10,
                                h=None,
                                pool_op=ADMM_SpatialPool((pool_h, pool_w)),
                                config=layer_config,
                                use_reset=False,
                            ),
                        ]
                    )

            case "feedforward":
                layer_list = []
                current_in = 28 * 28

                # 1. Build all hidden layers (constant width)
                for _ in range(num_layers - 1):
                    layer_list.append(
                        ADMM_FeedForward(
                            in_f=current_in,
                            out_f=hidden_dims,
                            h=ADMM_ReLU(),
                            config=layer_config,
                        )
                    )
                    # For all subsequent layers, the input is now hidden_dims
                    current_in = hidden_dims

                # 2. Build the final classification layer
                layer_list.append(
                    ADMM_FeedForward(
                        in_f=current_in,
                        out_f=10,
                        h=ADMM_ReLU(),
                        config=layer_config,
                        use_reset=False,
                    )
                )

                # 3. Convert to PyTorch ModuleList
                layers = nn.ModuleList(layer_list)

            case "conv":
                k = cfg.get("kernel_size", 5)
                p = cfg.get("padding", 2)
                s = cfg.get("stride", 1)

                layer_list = []

                # Initial Image dimensions (1 channel, 28x28)
                current_in_c = 1
                current_spatial = 28

                # 1. Build N-2 Convolutional Layers (High-Res Feature Extractors)
                for _ in range(num_layers - 2):
                    layer_list.append(
                        ADMM_Conv2d(
                            in_c=current_in_c,
                            out_c=hidden_channels,
                            k=k,
                            p=p,
                            s=1,
                            h=ADMM_ReLU(),
                            config=layer_config,
                            padding_mode="circular",
                        )  # <-- Changed to True for speed!
                    )

                    # Update spatial size mathematically (using stride 1)
                    current_spatial = int(calc_spatial_out(current_spatial, k, p, 1))
                    current_in_c = hidden_channels

                # 2. Build the final Convolutional Layer (The Bottleneck)
                layer_list.append(
                    ADMM_Conv2d(
                        in_c=current_in_c,
                        out_c=hidden_channels,
                        k=k,
                        p=p,
                        s=s,
                        h=ADMM_ReLU(),
                        config=layer_config,
                        padding_mode=padding_mode,
                    )
                )

                # CRITICAL FIX: Update the spatial size one last time based on stride `s`!
                current_spatial = int(calc_spatial_out(current_spatial, k, p, s))

                # 3. Build the final FeedForward Classification Layer
                lin_in = hidden_channels * current_spatial * current_spatial

                layer_list.append(
                    ADMM_FeedForward(
                        in_f=lin_in,
                        out_f=10,
                        h=ADMM_ReLU(),
                        pool_op=ADMM_Flatten(),
                        config=layer_config,
                        use_reset=False,
                    )
                )

                # 4. Convert to PyTorch ModuleList
                layers = nn.ModuleList(layer_list)
        # ---------------------------------------------------------
        # 3. INITIALIZE MODEL & METRICS
        # ---------------------------------------------------------
        config = ADMM_Config(init=init, train_method=train_method)
        if model_name.split("-")[0] == "spiking":
            model = ADMM(
                layers, loss_f=ADMM_CrossEntropy_Taylor(), T=n_timesteps, config=config
            ).to(device)
        else:
            model = ADMM(layers, loss_f=ADMM_SSE(), config=config).to(device)

        m = ADMM_Metrics(model)

        if is_static_run:
            print(json.dumps(m.network_size_statistics(), indent=4))

        keys_to_exclude = {"model", "batch_size", "seed"}
        filtered_keys = [k for k in dynamic_keys if k not in keys_to_exclude]

        # 3. Build the combo string using the filtered list
        if is_static_run or not filtered_keys:
            combo_str = "baseline"
        else:
            combo_str = "_".join([f"{k}-{cfg[k]}" for k in filtered_keys])

        metrics_path = f"scripts/results/{model_name}/{batch_size}/{combo_str}/{seed}"
        os.makedirs(metrics_path, exist_ok=True)

        print("Training...")

        # ---------------------------------------------------------
        # 4. TRAINING LOOP
        # ---------------------------------------------------------
        start_time = time.time()
        for epoch in range(epochs + 1):
            model.fit(train_loader, warming=epoch < warming_iters)
            if epoch % 1 == 0:
                with torch.no_grad():
                    m.save_metrics()
                    print(f"Epoch [{epoch:3d}/{epochs}] | {m}")
        #########################################
        # SAVING RESULTS AND PLOTTING

        end_time = time.time()
        running_time = end_time - start_time
        print(f"Model finished in {end_time - start_time:.2f} seconds.")
        metrics = m.get_dic()
        metrics["running_time"] = running_time

        # Store for plotting if static
        if is_static_run:
            final_metrics = metrics

        with open(os.path.join(metrics_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    # ==========================================
    # VISUALIZATION (Static Run Only)
    # ==========================================
    if is_static_run:
        m.plot()
