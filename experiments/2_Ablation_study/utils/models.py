import configparser

import torch
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_Config,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_SpikingFeedForward,
)


def get_model(
    init,
    train_method,
    time_order,
    layer_order,
    loss,
    z_first,
    lagrange_config,
    block_method,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()

    config.read("experiments/2_Ablation_study/config/config.ini")
    hidden_size = config.getint("config", "hidden_size")
    rho = config.getfloat("config", "rho")
    beta = config.getfloat("config", "beta")
    deltas = config.getfloat("config", "deltas")
    thetas = config.getfloat("config", "thetas")
    n_timesteps = config.getint("config", "n_timesteps")

    hidden_layer_config = ADMM_LayerConfig(
        rho=rho,
        beta=beta,
        use_bias=False,
        deltas=deltas,
        thetas=thetas,
        use_reset=True,
        use_lagrange=True if lagrange_config in ["first_only", "both"] else False,
    )
    out_layer_config = ADMM_LayerConfig(
        rho=rho,
        beta=beta,
        use_bias=False,
        deltas=deltas,
        thetas=thetas,
        use_reset=False,
        use_lagrange=True if lagrange_config in ["last_only", "both"] else False,
    )
    spff_layers = nn.ModuleList(
        [
            ADMM_SpikingFeedForward(
                in_f=34 * 34 * 2,
                out_f=hidden_size,
                h=ADMM_Heaviside(thetas=thetas),
                config=hidden_layer_config,
            ),
            ADMM_SpikingFeedForward(
                in_f=hidden_size,
                out_f=10,
                h=None,
                config=out_layer_config,
            ),
        ]
    )

    config = ADMM_Config(
        init=init,
        train_method=train_method,
        time_order=time_order,
        layer_order=layer_order,
        update_z_first=z_first,
        block_method=block_method,
    )
    admm_model = ADMM(
        spff_layers,
        loss_f=loss,
        T=n_timesteps,
        config=config,
    ).to(device)

    return admm_model
