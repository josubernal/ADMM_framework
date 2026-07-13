import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from src.admm import (
    ADMM,
    ADMM_Config,
    ADMM_CrossEntropy_Taylor,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_SpikingFeedForward,
)

from .utils.dataset import get_dataset, get_distributed_dataset


def setup(rank, world_size):
    """Configures the local environment for the distributed processes."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["USE_LIBUV"] = "0"  # ADD THIS LINE FOR WINDOWS COMPATIBILITY
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_baseline(
    EPOCHS,
    WARMING_ITERS,
    N_TIMESTEPS,
    BATCH_SIZE,
    HIDDEN_DIMS,
    RHO,
    BETA,
    DELTAS,
    THETAS,
    seed,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print("Starting model training...")
    for epoch in range(EPOCHS + 1):
        model.fit(train_loader, warming=epoch < WARMING_ITERS)
        if epoch % 1 == 0:
            with torch.no_grad():
                m2.save_metrics()
                print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m2}")

    #########################################
    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {k: v.cpu() for k, v in model.state_dict().items()}


def run_admm_worker(
    rank,
    EPOCHS,
    WARMING_ITERS,
    N_TIMESTEPS,
    BATCH_SIZE,
    HIDDEN_DIMS,
    RHO,
    BETA,
    DELTAS,
    THETAS,
    seed,
    world_size,
    shared_dict,
):
    """This function executes independently on each simulated 'node'."""
    print(f"Starting ADMM worker on rank {rank}/{world_size}")
    setup(rank, world_size)

    # 1. Initialize one model in each worker
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = get_distributed_dataset(
        batch_size=BATCH_SIZE,
        device=device,
        n_timesteps=N_TIMESTEPS,
        seed=seed,
        rank=rank,
        world_size=world_size,
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
        block_method="distributed",
    )

    model = ADMM(
        layers, loss_f=ADMM_CrossEntropy_Taylor(), T=N_TIMESTEPS, config=config
    ).to(device)

    m2 = ADMM_Metrics(model)
    if rank == 0:
        print("Starting model training...")
    for epoch in range(EPOCHS + 1):
        model.fit(train_loader, warming=epoch < WARMING_ITERS)
        if epoch % 1 == 0:
            with torch.no_grad():
                m2.save_distributed_metrics()
                if rank == 0:
                    print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m2}")

    #########################################
    # SAVING RESULTS
    if rank == 0:
        shared_dict["dist_state"] = {k: v.cpu() for k, v in model.state_dict().items()}

    cleanup()


if __name__ == "__main__":
    WORLD_SIZE = 2
    EPOCHS = 10
    WARMING_ITERS = 3
    N_TIMESTEPS = 15
    BATCH_SIZE = 20
    HIDDEN_DIMS = 50
    RHO = 1
    BETA = 0.1
    DELTAS = 0.95
    THETAS = 1

    seed = 8281003564
    manager = mp.Manager()
    shared_dict = manager.dict()
    mp.spawn(
        run_admm_worker,
        args=(
            EPOCHS,
            WARMING_ITERS,
            N_TIMESTEPS,
            BATCH_SIZE // WORLD_SIZE,
            HIDDEN_DIMS,
            RHO,
            BETA,
            DELTAS,
            THETAS,
            seed,
            WORLD_SIZE,
            shared_dict,
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
    base_state = run_baseline(
        EPOCHS,
        WARMING_ITERS,
        N_TIMESTEPS,
        BATCH_SIZE,
        HIDDEN_DIMS,
        RHO,
        BETA,
        DELTAS,
        THETAS,
        seed,
    )
    dist_state = shared_dict["dist_state"]

    print("\n" + "=" * 50)
    print("VERIFYING WEIGHTS AND BIASES IN MEMORY")
    print("=" * 50)

    all_match = True

    for key in base_state.keys():
        tensor_base = base_state[key]
        tensor_dist = dist_state[key]

        if tensor_base.shape != tensor_dist.shape:
            print(
                f"[FAIL] {key}: Shape mismatch! Base {tensor_base.shape} vs Dist {tensor_dist.shape}"
            )
            all_match = False
            continue

        is_close = torch.allclose(tensor_base, tensor_dist, atol=1e-5)

        if is_close:
            print(f"[PASS] {key} perfectly matches.")
        else:
            print(
                f"[FAIL] {key} differs! Max diff: {torch.max(torch.abs(tensor_base - tensor_dist)).item():.8f}"
            )
            all_match = False

    if all_match:
        print("SUCCESS: Distributed and Baseline models are mathematically identical!")
    else:
        print("WARNING: Some parameters diverged during training.")
