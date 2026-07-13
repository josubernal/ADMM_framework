import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def get_dataset(batch_size, device=None, seed=64):
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
    data = data.view(data.size(0), -1)
    data = data.to(device).float()
    targets = targets.to(device)
    data += 0.01 * torch.randn_like(data)

    return [(data, targets)]


def get_distributed_dataset(batch_size, device=None, seed=64, rank=0, world_size=1):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    sampler = DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )

    data, targets = next(iter(train_loader))
    data = data.view(data.size(0), -1)
    data = data.to(device).float()
    targets = targets.to(device)
    data += 0.01 * torch.randn_like(data)

    return [(data, targets)]


def get_data(batch_size, device=None, seed=64):
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
    images = images.view(images.size(0), -1)

    images = images.to(device).float()
    labels = labels.to(device)
    images += 0.01 * torch.randn_like(images)

    return images, labels
