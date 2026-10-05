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


def format_images(images, model_name):
    """Formats the image tensor geometry based on the model architecture."""
    if model_name == "spiking-feedforward":
        # Flatten spatial dims: [B, T, C, H, W] -> [B, T, Features]
        images = images.view(images.size(0), images.size(1), -1)

    return images.contiguous()


def format_images_static(images, model_name):
    """Formats the image tensor geometry based on the model architecture."""
    if model_name == "feedforward":
        # Flatten spatial dims: [B, C, H, W] -> [B, Features]
        images = images.view(images.size(0), -1)

    return images


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
        shuffle=False,
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
        shuffle=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    data, targets = next(iter(train_loader))
    data = format_images_static(data, model_name)
    data = data.to(device).float()
    targets = targets.to(device)
    data += 0.01 * torch.randn_like(data)

    return [(data, targets)]


def get_dataset(
    model_name,
    batch_size,
    device=None,
    seed=64,
    n_timesteps=150,
    num_workers=0,
    noise_std=0.01,
):

    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = tr.Compose(
        [
            tr.Denoise(filter_time=10000),
            tr.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )
    trainset = tonic.datasets.NMNIST(
        save_to="./data",
        transform=frame_transform,
        train=True,
    )
    cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")
    train_loader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        # each prefetched batch is several GB, keep the lookahead minimal
        prefetch_factor=1 if num_workers > 0 else None,
    )

    # Seeded, so the noise is drawn ONCE and stays identical every epoch
    # (same as the old single-batch version).
    noise_gen = torch.Generator().manual_seed(seed)

    batches = []
    for i, (data, targets) in enumerate(train_loader):
        data = format_images(data, model_name)  # [B, T_padded, F]

        if data.size(1) > n_timesteps:
            # .contiguous() is essential: a plain slice is a view that keeps
            # the whole padded tensor alive in RAM.
            data = data[:, :n_timesteps, :].contiguous()

        data = data.float()
        data.add_(torch.randn(data.shape, generator=noise_gen), alpha=noise_std)

        # [B, T, F] -> [T, B, F] view, same layout the old code returned
        batches.append((data.transpose(0, 1), targets))

        gb = data.numel() * data.element_size() / 1024**3
        print(
            f"[get_dataset] batch {i + 1}/{len(train_loader)} "
            f"shape={tuple(data.transpose(0, 1).shape)} {gb:.2f} GB",
            flush=True,
        )

    t_sizes = {b[0].size(0) for b in batches}
    if len(t_sizes) != 1:
        print(f"[get_dataset] WARNING: batches have different T: {t_sizes}")

    # Tensors stay on the CPU (83 GB does not fit on a GPU);
    # ADMM._iterate_batches moves each batch to the device when it needs it.
    return batches
