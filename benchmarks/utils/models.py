import configparser

import torch
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_Config,
    ADMM_Conv2d,
    ADMM_Flatten,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Linear,
    ADMM_ReLU,
    ADMM_SpikingConv2d,
    ADMM_SpikingLinear,
)


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


def get_model(model_name, init, train_method, layer_order, loss, z_first):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()

    config.read("benchmarks/config/config.ini")

    hidden_size_static = config.getint("config", "hidden_size_static")
    hidden_channels_static = config.getint("config", "hidden_channels_static")
    hidden_size_spiking = config.getint("config", "hidden_size_spiking")
    hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")
    k = config.getint("config", "k")
    p = config.getint("config", "p")
    s = config.getint("config", "s")

    linear_rho = config.getfloat("config", "linear_rho")
    linear_beta = config.getfloat("config", "linear_beta")
    conv_rho = config.getfloat("config", "conv_rho")
    conv_beta = config.getfloat("config", "conv_beta")
    splinear_rho = config.getfloat("config", "splinear_rho")
    splinear_beta = config.getfloat("config", "splinear_beta")
    spconv_rho = config.getfloat("config", "spconv_rho")
    spconv_beta = config.getfloat("config", "spconv_beta")

    deltas = config.getfloat("config", "deltas")
    thetas = config.getfloat("config", "thetas")
    input_size = 784
    n_timesteps = config.getint("config", "n_timesteps")

    match model_name:
        case "linear":
            hidden_layer_config = ADMM_LayerConfig(
                rho=linear_rho, beta=linear_beta, use_bias=True
            )
            out_layer_config = ADMM_LayerConfig(
                rho=linear_rho, beta=linear_beta, use_bias=True
            )
            linear_layers = nn.ModuleList(
                [
                    ADMM_Linear(
                        in_f=input_size,
                        out_f=hidden_size_static,
                        h=ADMM_ReLU(),
                        config=hidden_layer_config,
                    ),
                    ADMM_Linear(
                        in_f=hidden_size_static,
                        out_f=10,
                        h=ADMM_ReLU(),
                        config=out_layer_config,
                    ),
                ]
            )
            config = ADMM_Config(
                init=init,
                train_method="vectorized",
                layer_order=layer_order,
                update_z_first=z_first,
            )
            admm_model = ADMM(
                linear_layers,
                loss_f=loss,
                config=config,
            ).to(device)

        case "conv":
            spatial_dim = int(calc_spatial_out(28, k, p, s))
            lin_in_dim = hidden_channels_static * spatial_dim * spatial_dim
            hidden_layer_config = ADMM_LayerConfig(
                rho=conv_rho, beta=conv_beta, use_bias=True, use_fft=False
            )
            out_layer_config = ADMM_LayerConfig(
                rho=conv_rho, beta=conv_beta, use_bias=True, use_fft=False
            )
            conv_layers = nn.ModuleList(
                [
                    ADMM_Conv2d(
                        in_c=1,
                        out_c=hidden_channels_static,
                        k=k,
                        p=p,
                        s=s,
                        h=ADMM_ReLU(),
                        config=hidden_layer_config,
                        padding_mode="zeros",
                    ),
                    ADMM_Linear(
                        in_f=lin_in_dim,
                        out_f=10,
                        h=ADMM_ReLU(),
                        pool_op=ADMM_Flatten(),
                        config=out_layer_config,
                    ),
                ]
            )
            config = ADMM_Config(
                init=init,
                train_method="vectorized",
                layer_order=layer_order,
                update_z_first=z_first,
            )
            admm_model = ADMM(
                conv_layers,
                loss_f=loss,
                config=config,
            ).to(device)
        case "spiking-linear":
            hidden_layer_config = ADMM_LayerConfig(
                rho=splinear_rho,
                beta=splinear_beta,
                use_bias=False,
                deltas=deltas,
                thetas=thetas,
                use_reset=True,
            )
            out_layer_config = ADMM_LayerConfig(
                rho=splinear_rho,
                beta=splinear_beta,
                use_bias=False,
                deltas=deltas,
                thetas=thetas,
                use_reset=False,
            )
            splinear_layers = nn.ModuleList(
                [
                    ADMM_SpikingLinear(
                        in_f=34 * 34 * 2,
                        out_f=hidden_size_spiking,
                        h=ADMM_Heaviside(thetas=thetas),
                        config=hidden_layer_config,
                    ),
                    ADMM_SpikingLinear(
                        in_f=hidden_size_spiking,
                        out_f=10,
                        h=None,
                        config=out_layer_config,
                    ),
                ]
            )
            config = ADMM_Config(
                init=init,
                train_method=train_method,
                layer_order=layer_order,
                update_z_first=z_first,
            )
            admm_model = ADMM(
                splinear_layers,
                loss_f=loss,
                T=n_timesteps,
                config=config,
            ).to(device)
        case "spiking-conv":
            spatial_dim = int(calc_spatial_out(34, k, p, s))
            lin_in_dim = hidden_channels_spiking * spatial_dim * spatial_dim
            hidden_layer_config = ADMM_LayerConfig(
                rho=spconv_rho,
                beta=spconv_beta,
                thetas=thetas,
                deltas=deltas,
                use_bias=False,
                use_fft=False,
                use_reset=True,
            )
            out_layer_config = ADMM_LayerConfig(
                rho=spconv_rho,
                beta=spconv_beta,
                thetas=thetas,
                deltas=deltas,
                use_bias=False,
                use_fft=False,
                use_reset=False,
            )
            spconv_layers = nn.ModuleList(
                [
                    ADMM_SpikingConv2d(
                        in_c=2,
                        out_c=hidden_channels_spiking,
                        k=k,
                        p=p,
                        s=s,
                        h=ADMM_Heaviside(thetas=thetas),
                        config=hidden_layer_config,
                        padding_mode="zeros",
                    ),
                    ADMM_SpikingLinear(
                        in_f=lin_in_dim,
                        pool_op=ADMM_Flatten(),
                        out_f=10,
                        h=None,
                        config=out_layer_config,
                    ),
                ]
            )
            config = ADMM_Config(
                init=init,
                train_method=train_method,
                layer_order=layer_order,
                update_z_first=z_first,
            )

            admm_model = ADMM(
                spconv_layers,
                loss_f=loss,
                T=n_timesteps,
                config=config,
            ).to(device)

    return admm_model
