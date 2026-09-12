import tonic
import tonic.transforms as tr
import torch
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def format_images(images, model_name):
    """Formats the image tensor geometry based on the model architecture."""
    if model_name == "spiking-feedforward":
        # Flatten spatial dims: [T, B, C, H, W] -> [T, B, Features]
        images = images.view(images.size(0), images.size(1), -1)
    elif model_name == "feedforward":
        # Flatten spatial dims: [B, C, H, W] -> [B, Features]
        images = images.view(images.size(0), -1)

    return images.contiguous()


def get_dataset(
    model_name, batch_size, device=None, seed=64, n_timesteps=150, n_batches=1
):
    spiking = model_name.startswith("spiking")

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
        )

        batches = []
        for i, (data, targets) in enumerate(train_loader):
            if i >= n_batches:
                break

            data = format_images(data, model_name)

            if data.size(1) > n_timesteps:
                data = data[:, :n_timesteps, :]

            data = data.transpose(0, 1).contiguous()
            data += 0.01 * torch.randn_like(data)

            batches.append((data.to(device), targets.to(device)))

        return batches
    else:
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
        )
        data, targets = next(iter(train_loader))
        data = format_images(data, model_name)
        data = data.to(device).float()
        targets = targets.to(device)
        data += 0.01 * torch.randn_like(data)

        return [(data, targets)]


#############################################
# FOR GD
def get_data(model_name, batch_size, device=None, n_timesteps=150, seed=64):
    spiking = model_name.startswith("spiking")

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
        )

        data, targets = next(iter(train_loader))
        data = format_images(data, model_name)

        if data.size(1) > n_timesteps:
            data = data[:, :n_timesteps, :]

        data = data.transpose(0, 1).contiguous()
        data += 0.01 * torch.randn_like(data)
        return data.to(device), targets.to(device)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        mnist_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        dataloader = torch.utils.data.DataLoader(
            mnist_train, batch_size=batch_size, shuffle=True
        )
        images, labels = next(iter(dataloader))
        images = format_images(images, model_name)

        images = images.to(device).float()
        labels = labels.to(device)
        images += 0.01 * torch.randn_like(images)

        return images, labels
