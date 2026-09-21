import os
from functools import partial

import tonic
import tonic.transforms as tr
import torch
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def collate_static(batch, model_name):
    data, targets = torch.utils.data.default_collate(batch)

    data = format_images(data, model_name)

    return data, targets


def collate_spiking(batch, model_name, n_timesteps):
    data, targets = tonic.collation.PadTensors()(batch)

    data = format_images(data, model_name)

    if data.size(1) > n_timesteps:
        data = data[:, :n_timesteps, ...]

    data = data.transpose(0, 1).contiguous()

    data += 0.01 * torch.randn_like(data)

    return data, targets


def format_images(images, model_name):
    """Formats the image tensor geometry based on the model architecture."""
    if model_name == "spiking-feedforward":
        # Flatten spatial dims: [T, B, C, H, W] -> [T, B, Features]
        images = images.view(images.size(0), images.size(1), -1)
    elif model_name == "feedforward":
        # Flatten spatial dims: [B, C, H, W] -> [B, Features]
        images = images.view(images.size(0), -1)

    return images.contiguous()


def get_dataset_static_gd(
    model_name,
    batch_size,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_static,
            model_name=model_name,
        ),
    )
    return train_loader


def get_dataset_static_admm(
    model_name,
    batch_size,
    device=None,
    seed=64,
):

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
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    data, targets = next(iter(train_loader))
    data = format_images(data, model_name)
    data = data.to(device).float()
    targets = targets.to(device)
    data += 0.01 * torch.randn_like(data)

    return [(data, targets)]


def get_dataset_spiking_gd(
    batch_size,
    seed=64,
    num_workers=None,
):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = tr.Compose(
        [
            tr.Denoise(filter_time=10000),
            tr.ToFrame(
                sensor_size=sensor_size,
                time_window=1000,
            ),
        ]
    )

    trainset = tonic.datasets.NMNIST(
        save_to="./data",
        transform=frame_transform,
        train=True,
    )

    cached_trainset = DiskCachedDataset(
        trainset,
        cache_path="./cache/nmnist/train",
    )

    # Auto-pick worker count if not specified
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    persistent = num_workers > 0

    train_loader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        generator=torch.Generator().manual_seed(seed),
    )

    return train_loader


def get_dataset_spiking_admm(
    model_name,
    batch_size,
    seed=64,
    n_timesteps=150,
):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = tr.Compose(
        [
            tr.Denoise(filter_time=10000),
            tr.ToFrame(
                sensor_size=sensor_size,
                time_window=1000,
            ),
        ]
    )

    trainset = tonic.datasets.NMNIST(
        save_to="./data",
        transform=frame_transform,
        train=True,
    )

    cached_trainset = DiskCachedDataset(
        trainset,
        cache_path="./cache/nmnist/train",
    )

    train_loader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_spiking,
            model_name=model_name,
            n_timesteps=n_timesteps,
        ),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    return train_loader
