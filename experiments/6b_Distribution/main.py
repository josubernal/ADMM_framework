import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim

from src.admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_FeedForward,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_ReLU,
)

from .utils.dataset import get_data, get_dataset, get_distributed_dataset


def setup(mode):
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    if mode == "internode":
        # Use Gloo for multi-node to bypass cluster firewalls
        chosen_backend = "nccl"
    # Inside your setup(mode) function:
    elif mode == "cpu_dist":
        chosen_backend = "gloo"
    else:
        # Keep NCCL for intranode since it doesn't have to leave the physical server
        chosen_backend = "nccl"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=chosen_backend, init_method="env://", rank=rank, world_size=world_size
    )
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def run_adam(
    EPOCHS,
    BATCH_SIZE,
    HIDDEN_DIMS,
    LR,
    seed,
    save_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class GDFFNet(nn.Module):
        def __init__(self, input_size=784, hidden_size=HIDDEN_DIMS, num_classes=10):
            super(GDFFNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = GDFFNet().to(device)
    criterion = nn.CrossEntropyLoss()
    images, labels = get_data(
        batch_size=BATCH_SIZE,
        device=device,
        seed=seed,
    )

    if labels.dim() > 1:
        labels = labels.argmax(dim=1)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    adam_accs = []
    adam_losses = []
    adam_f1s = []
    adam_times = []

    print("\nTraining model with gradient descent...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        outputs = model(images)

        loss = criterion(outputs, labels.long())

        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100

        # F1 Score Calculation
        classes = torch.unique(torch.cat((labels, predictions)))
        f1_sum = 0.0

        for c in classes:
            # Calculate True Positives, False Positives, and False Negatives for class 'c'
            tp = ((predictions == c) & (labels == c)).sum().float()
            fp = ((predictions == c) & (labels != c)).sum().float()
            fn = ((predictions != c) & (labels == c)).sum().float()

            # Calculate F1 for this class (1e-8 is an epsilon to prevent division by zero)
            f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # Macro F1 average across all found classes
        f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append metrics
        adam_accs.append(accuracy)
        adam_losses.append(loss.item())
        adam_f1s.append(f1)
        elapsed_time = time.time() - start_time
        adam_times.append(elapsed_time)

        print(f"Step {epoch + 1} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")
    metrics = {}
    metrics["accuracy"] = adam_accs
    metrics["loss"] = adam_losses
    metrics["times"] = adam_times
    metrics["f1"] = adam_f1s
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_baseline(
    EPOCHS,
    WARMING_ITERS,
    BATCH_SIZE,
    HIDDEN_DIMS,
    RHO,
    BETA,
    seed,
    save_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = get_dataset(batch_size=BATCH_SIZE, device=device, seed=seed)

    hidden_layer_config = ADMM_LayerConfig(rho=RHO, beta=BETA, use_bias=True)
    out_layer_config = ADMM_LayerConfig(
        rho=RHO,
        beta=BETA,
        use_bias=True,
        use_lagrange=True,
    )

    layers = nn.ModuleList(
        [
            ADMM_FeedForward(
                in_f=784,
                out_f=HIDDEN_DIMS,
                h=ADMM_ReLU(),
                config=hidden_layer_config,
            ),
            ADMM_FeedForward(
                in_f=HIDDEN_DIMS,
                out_f=10,
                h=ADMM_ReLU(),
                config=out_layer_config,
            ),
        ]
    )

    config = ADMM_Config(
        init="pytorch",
        train_method="vectorized",
        layer_order="backwards",
        update_z_first=False,
        block_method="two-block",
    )

    model = ADMM(layers, loss_f=ADMM_SSE(), config=config).to(device)
    m2 = ADMM_Metrics(model)

    print("Starting Baseline Training...")
    baseline_times = []
    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        model.fit(train_loader, warming=epoch < WARMING_ITERS)
        baseline_times.append(time.time() - start_time)

        if epoch % 1 == 0:
            with torch.no_grad():
                m2.save_metrics()
                print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m2}")

    metrics = m2.get_dic()
    metrics["times"] = baseline_times
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_admm_worker(
    rank,
    local_rank,
    EPOCHS,
    WARMING_ITERS,
    BATCH_SIZE,
    HIDDEN_DIMS,
    RHO,
    BETA,
    seed,
    world_size,
    save_path,
):
    print(
        f"Starting ADMM worker on global rank {rank}/{world_size}, local rank {local_rank}"
    )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = get_distributed_dataset(
        batch_size=BATCH_SIZE,
        device=device,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    hidden_layer_config = ADMM_LayerConfig(rho=RHO, beta=BETA, use_bias=True)
    out_layer_config = ADMM_LayerConfig(
        rho=RHO,
        beta=BETA,
        use_bias=True,
        use_lagrange=True,
    )

    layers = nn.ModuleList(
        [
            ADMM_FeedForward(
                in_f=784,
                out_f=HIDDEN_DIMS,
                h=ADMM_ReLU(),
                config=hidden_layer_config,
            ),
            ADMM_FeedForward(
                in_f=HIDDEN_DIMS,
                out_f=10,
                h=ADMM_ReLU(),
                config=out_layer_config,
            ),
        ]
    )

    config = ADMM_Config(
        init="pytorch",
        train_method="vectorized",
        layer_order="backwards",
        update_z_first=False,
        block_method="distributed",
    )

    model = ADMM(layers, loss_f=ADMM_SSE(), config=config).to(device)
    m2 = ADMM_Metrics(model)

    if rank == 0:
        print("Starting Distributed Training...")

    distributed_times = []
    start_time = time.time()

    for epoch in range(EPOCHS + 1):
        model.fit(train_loader, warming=epoch < WARMING_ITERS)
        distributed_times.append(time.time() - start_time)

        if epoch % 1 == 0:
            with torch.no_grad():
                m2.save_distributed_metrics()
                if rank == 0:
                    print(f"Epoch [{epoch:3d}/{EPOCHS}] | {m2}")

    if rank == 0:
        metrics = m2.get_dic()
        metrics["times"] = distributed_times
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "baseline",
            "internode",
            "intranode",
            "adam",
            "baseline-cpu",
            "cpu_dist",
        ],
        required=True,
    )
    args = parser.parse_args()

    EPOCHS = 1000
    WARMING_ITERS = 300
    BATCH_SIZE = 60000
    HIDDEN_DIMS = 1000
    RHO = 0.1
    BETA = 0.1
    seed = 8281003564
    LR = 0.001

    if args.mode == "baseline":
        run_baseline(
            EPOCHS,
            WARMING_ITERS,
            BATCH_SIZE,
            HIDDEN_DIMS,
            RHO,
            BETA,
            seed,
            save_path="experiments/6b_Distribution/results/baseline/results.json",
        )
    if args.mode == "baseline-cpu":
        run_baseline(
            EPOCHS,
            WARMING_ITERS,
            BATCH_SIZE,
            HIDDEN_DIMS,
            RHO,
            BETA,
            seed,
            save_path="experiments/6b_Distribution/results/baseline-cpu/results.json",
        )
    if args.mode == "adam":
        run_adam(
            EPOCHS,
            BATCH_SIZE,
            HIDDEN_DIMS,
            LR,
            seed,
            save_path="experiments/6b_Distribution/results/adam/results.json",
        )
    else:
        rank, world_size, local_rank = setup(args.mode)

        if (
            args.mode == "intranode"
            or args.mode == "internode"
            or args.mode == "cpu_dist"
        ):
            folder_name = f"{args.mode}_{world_size}w"
        else:
            folder_name = args.mode

        save_path = f"experiments/6b_Distribution/results/{folder_name}/results.json"

        run_admm_worker(
            rank=rank,
            local_rank=local_rank,
            EPOCHS=EPOCHS,
            WARMING_ITERS=WARMING_ITERS,
            BATCH_SIZE=BATCH_SIZE // world_size,
            HIDDEN_DIMS=HIDDEN_DIMS,
            RHO=RHO,
            BETA=BETA,
            seed=seed,
            world_size=world_size,
            save_path=save_path,
        )
