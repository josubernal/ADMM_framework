import os
import shutil
from functools import partial
from pathlib import Path

import tonic
import tonic.transforms as tr
import torch
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader, Dataset
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

    data = 0.01 * torch.randn_like(data)

    return data, targets


def format_images(images, model_name):
    """Formats the image tensor geometry based on the model architecture."""
    if model_name == "spiking-feedforward":
        # Flatten spatial dims: [T, B, C, H, W] -> [T, B, Features]
        images = images.view(images.size(0), images.size(1), -1)
    elif model_name == "feedforward":
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
        shuffle=False,
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

    # Existing Tonic cache.
    # cached_trainset = DiskCachedDataset(
    #     trainset,
    #     cache_path="./cache/nmnist/train",
    # )

    # padding + formatting + truncation + transpose + fixed noise
    processed_trainset = CachedSpikingDataset(
        dataset=trainset,
        cache_path=f"./cache/nmnist/temp_admm_{model_name}_{batch_size}_{n_timesteps}_{seed}",
        batch_size=batch_size,
        n_timesteps=n_timesteps,
        model_name=model_name,
        noise_std=0.01,
        seed=seed,
    )

    train_loader = DataLoader(
        processed_trainset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )
    shutil.rmtree("./data")
    return train_loader


class CachedSpikingDataset(Dataset):
    def __init__(
        self,
        dataset,
        cache_path,
        batch_size,
        n_timesteps,
        model_name,
        noise_std=0.01,
        seed=64,
    ):
        self.dataset = dataset
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.model_name = model_name
        self.noise_std = noise_std
        self.seed = seed

        self.cache_path.mkdir(parents=True, exist_ok=True)

        self._prepare_cache()

    def _prepare_cache(self):
        marker = self.cache_path / "complete"

        if marker.exists():
            print(f"Using cached dataset: {self.cache_path}")
            return

        print("Preprocessing data...")

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        batch_id = 0

        for start in range(0, len(self.dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dataset))

            print(
                f"Batch: {batch_id + 1} "
                f"of {(len(self.dataset) + self.batch_size - 1) // self.batch_size}"
            )

            # Get the individual samples belonging to this batch
            batch = [self.dataset[idx] for idx in range(start, end)]

            data, targets = tonic.collation.PadTensors()(batch)

            data = format_images(
                data,
                self.model_name,
            )
            if data.size(1) > self.n_timesteps:
                data = data[:, : self.n_timesteps, ...]

            data = data.transpose(0, 1).contiguous()

            noise = torch.randn(
                data.shape,
                generator=generator,
                dtype=data.dtype,
            )

            data = data + self.noise_std * noise

            torch.save(
                (data, targets),
                self.cache_path / f"batch_{batch_id}.pt",
            )

            batch_id += 1

        torch.save(
            batch_id,
            self.cache_path / "num_batches.pt",
        )

        marker.touch()

        print(f"Finished creating cache with {batch_id} batches.")

    def __len__(self):
        return torch.load(
            self.cache_path / "num_batches.pt",
            weights_only=True,
        )

    def __getitem__(self, idx):
        return torch.load(
            self.cache_path / f"batch_{idx}.pt",
            weights_only=True,
        )
