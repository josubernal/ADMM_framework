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


def format_images(images, model_name):
    """Formats the image tensor geometry based on the model architecture."""
    if model_name == "spiking-feedforward":
        # Flatten spatial dims: [T, B, C, H, W] -> [T, B, Features]
        images = images.view(images.size(0), images.size(1), -1)
    elif model_name == "feedforward":
        # Flatten spatial dims: [B, C, H, W] -> [B, Features]
        images = images.view(images.size(0), -1)

    return images.contiguous()


def spiking_collate_fn(batch, n_timesteps, model_name):
    """Custom collate function to handle ADMM's [Time, Batch, ...] requirements."""
    dummy_batch = [(data, 0) for data, label in batch]
    # Explicitly enforce batch_first=False to guarantee [Time, Batch, C, H, W]
    images, _ = tonic.collation.PadTensors(batch_first=False)(dummy_batch)

    images += 0.01 * torch.randn_like(images)

    clean_labels = []
    for _, label in batch:
        t = label if isinstance(label, torch.Tensor) else torch.tensor(label)
        if t.dim() > 0 and t.size(-1) > 1:
            t = t.argmax()
        clean_labels.append(t.to(torch.long))

    labels = torch.stack(clean_labels)

    # 3. Handle timestep clipping/padding as usual
    if images.size(0) > n_timesteps:
        images = images[:n_timesteps]
    elif images.size(0) < n_timesteps:
        pad_size = n_timesteps - images.size(0)
        padding = torch.zeros(pad_size, *images.shape[1:], dtype=images.dtype)
        images = torch.cat([images, padding], dim=0)

    # Apply architecture-specific formatting
    images = format_images(images, model_name)

    return images, labels


def get_dataset(model_name, batch_size, n_batches, seed=64, n_timesteps=150):
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

        subset_indices = list(range(batch_size * n_batches))
        trainset = Subset(trainset, subset_indices)

        cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")
        train_loader = DataLoader(
            cached_trainset,
            batch_size=batch_size,
            collate_fn=lambda b: spiking_collate_fn(b, n_timesteps, model_name),
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
            target_transform=one_hot_target,
            transform=transform,
        )

        subset_indices = list(range(batch_size * n_batches))
        trainset = Subset(trainset, subset_indices)

        # For static models, we format within a custom collate function
        def static_collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            images = format_images(images, model_name)
            return images, labels

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=static_collate_fn,
        )
        return train_loader


#############################################
# FOR GD


def gd_collate_fn(batch, n_timesteps, model_name):
    dummy_batch = [(data, 0) for data, label in batch]
    # Guarantee [T, B, C, H, W] for consistency
    images, _ = tonic.collation.PadTensors(batch_first=False)(dummy_batch)

    raw_labels = [label for data, label in batch]
    clean_labels = []
    for label in raw_labels:
        t = label if isinstance(label, torch.Tensor) else torch.tensor(label)
        if t.dim() == 0:
            t = torch.nn.functional.one_hot(t.to(torch.long), num_classes=10).float()
        clean_labels.append(t)

    labels = torch.stack(clean_labels)

    # Truncate or Pad time dimension (dim 0)
    if images.size(0) > n_timesteps:
        images = images[:n_timesteps]
    elif images.size(0) < n_timesteps:
        pad_size = n_timesteps - images.size(0)
        padding = torch.zeros(pad_size, *images.shape[1:], dtype=images.dtype)
        images = torch.cat([images, padding], dim=0)

    images = format_images(images, model_name)
    return images, labels


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
            save_to="./data", transform=frame_transform, train=True
        )
        cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")
        train_loader = DataLoader(
            cached_trainset,
            batch_size=batch_size,
            collate_fn=lambda b: gd_collate_fn(b, n_timesteps, model_name),
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed),
        )

        images, labels = next(iter(train_loader))

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

    # Restored: Apply noise as done in your previous script
    images += 0.01 * torch.randn_like(images)

    return images, labels
