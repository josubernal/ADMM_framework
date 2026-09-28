import json
import os
from functools import partial
from pathlib import Path

import numpy as np
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
    cache_path = (
        f"./cache/nmnist/temp_admm_{model_name}_{batch_size}_{n_timesteps}_{seed}"
    )

    processed_trainset = CachedSpikingDataset(
        dataset=None,
        cache_path=cache_path,
        batch_size=batch_size,
        n_timesteps=n_timesteps,
        model_name=model_name,
        noise_std=0.01,
        seed=seed,
    )

    train_loader = torch.utils.data.DataLoader(
        processed_trainset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

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

        self.cache_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data_path = self.cache_path / "data.dat"
        self.targets_path = self.cache_path / "targets.dat"
        self.metadata_path = self.cache_path / "metadata.json"
        self.marker = self.cache_path / "complete"

        if not self.marker.exists():
            if self.dataset is None:
                raise RuntimeError(
                    "Processed cache does not exist. "
                    "Run preprocess_admm_spiking.py first."
                )

            self._prepare_cache()

        self._open_cache()

    def _open_cache(self):
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.num_samples = metadata["num_samples"]
        self.batch_size = metadata["batch_size"]
        self.data_shape = tuple(metadata["data_shape"])

        # Packed data shape:
        # [N, T, 867]
        self.data_memmap = np.memmap(
            self.data_path,
            dtype=np.uint8,
            mode="r",
            shape=self.data_shape,
        )

        self.targets_memmap = np.memmap(
            self.targets_path,
            dtype=np.int64,
            mode="r",
            shape=(self.num_samples,),
        )

        # Original feature dimension before 3-bit packing
        self.original_feature_dim = metadata.get(
            "original_feature_dim",
            self.data_shape[-1] * 8 // 3,
        )

        # Deterministic noise
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Reused for every batch
        self.noise_buffer = None

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(
            start + self.batch_size,
            self.num_samples,
        )

        # -------------------------------------------------------------
        # Load packed data
        # Shape: [B, T, 867]
        # -------------------------------------------------------------

        packed = self.data_memmap[start:end]

        # -------------------------------------------------------------
        # Unpack 3-bit values
        # Shape: [B, T, 2312]
        # -------------------------------------------------------------

        data = unpack_3bit(packed)

        targets = self.targets_memmap[start:end]

        # -------------------------------------------------------------
        # Convert to PyTorch
        # -------------------------------------------------------------

        data = torch.from_numpy(data)
        targets = torch.from_numpy(targets)

        # -------------------------------------------------------------
        # Convert clean uint8 data to float32
        # -------------------------------------------------------------

        data = data.float()

        # -------------------------------------------------------------
        # Add Gaussian noise
        # -------------------------------------------------------------

        if self.noise_std > 0:
            if self.noise_buffer is None or self.noise_buffer.shape != data.shape:
                self.noise_buffer = torch.empty_like(data)

            torch.randn(
                data.shape,
                generator=self.generator,
                out=self.noise_buffer,
            )

            self.noise_buffer.mul_(self.noise_std)
            data.add_(self.noise_buffer)

        # -------------------------------------------------------------
        # [B, T, F] -> [T, B, F]
        # -------------------------------------------------------------

        data = data.transpose(0, 1)

        return data, targets


def unpack_3bit(packed: np.ndarray) -> np.ndarray:
    """
    Unpack 3-bit values packed by pack_3bit().

    Input:
        (..., 3 * N) uint8

    Output:
        (..., 8 * N) uint8
    """
    if packed.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {packed.dtype}")

    packed_size = packed.shape[-1]

    if packed_size % 3 != 0:
        raise ValueError(
            f"Packed last dimension must be divisible by 3, got {packed_size}"
        )

    num_groups = packed_size // 3

    x = packed.reshape(-1, num_groups, 3)

    b0 = x[..., 0]
    b1 = x[..., 1]
    b2 = x[..., 2]

    unpacked = np.empty(
        (x.shape[0], num_groups, 8),
        dtype=np.uint8,
    )

    unpacked[..., 0] = b0 & 0b00000111
    unpacked[..., 1] = (b0 >> 3) & 0b00000111

    unpacked[..., 2] = ((b0 >> 6) & 0b00000011) | ((b1 & 0b00000001) << 2)

    unpacked[..., 3] = (b1 >> 1) & 0b00000111
    unpacked[..., 4] = (b1 >> 4) & 0b00000111

    unpacked[..., 5] = ((b1 >> 7) & 0b00000001) | ((b2 & 0b00000111) << 1)

    unpacked[..., 6] = (b2 >> 2) & 0b00000111
    unpacked[..., 7] = (b2 >> 5) & 0b00000111

    return unpacked.reshape(
        *packed.shape[:-1],
        num_groups * 8,
    )
