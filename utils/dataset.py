import tonic
import tonic.transforms as tr
import torch
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def one_hot_target(y, num_classes=10):
    """Converts a standard integer label into a one-hot float tensor."""
    return torch.nn.functional.one_hot(
        torch.tensor(y, dtype=torch.long), num_classes=num_classes
    ).float()


def spiking_collate_fn(batch, n_timesteps):
    """Custom collate function to handle ADMM's [Time, Batch, ...] requirements."""
    images, labels = tonic.collation.PadTensors(batch_first=False)(batch)

    if images.size(0) > n_timesteps:
        images = images[:n_timesteps]
    elif images.size(0) < n_timesteps:
        pad_size = n_timesteps - images.size(0)
        padding = torch.zeros(pad_size, *images.shape[1:], dtype=images.dtype)
        images = torch.cat([images, padding], dim=0)

    return images.contiguous(), labels


def get_dataset(batch_size, n_batches, spiking=False, seed=64, n_timesteps=150):
    if spiking:
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
            target_transform=one_hot_target,
            train=True,
        )

        # Test subset
        subset_indices = list(range(batch_size * n_batches))
        trainset = Subset(trainset, subset_indices)

        cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")
        train_loader = DataLoader(
            cached_trainset,
            batch_size=batch_size,
            collate_fn=lambda b: spiking_collate_fn(b, n_timesteps),
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return train_loader

    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
            target_transform=one_hot_target,
        )

        subset_indices = list(range(batch_size * n_batches))
        trainset = Subset(trainset, subset_indices)

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return train_loader
