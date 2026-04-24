from torchvision import datasets, transforms
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as tr
from torch.utils.data import DataLoader
import torch

def get_data(batch_size, spiking=False, device=None, n_timesteps=150, seed=64):
    if spiking:
        sensor_size = tonic.datasets.NMNIST.sensor_size
        frame_transform = tr.Compose([
                        tr.Denoise(filter_time=10000),
                        tr.ToFrame(sensor_size=sensor_size, time_window=1000)
                    ])
        trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
        cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
        train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                                  collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed))
                
        images, labels = next(iter(train_loader))
        if images.size(1) > n_timesteps:
            images = images[:, :n_timesteps, :]

    else: 
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])

        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        images, labels = next(iter(dataloader))

    images, labels = images.to(device), labels.to(device)
    images += 0.01 * torch.randn_like(images)
    return images, labels