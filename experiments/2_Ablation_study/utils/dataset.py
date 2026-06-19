import tonic
import tonic.transforms as tr
import torch
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


def get_dataset(batch_size, device=None, seed=64, n_timesteps=150):

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
    data = data.view(data.size(0), data.size(1), -1)

    if data.size(1) > n_timesteps:
        data = data[:, :n_timesteps, :]

    data = data.transpose(0, 1)
    data += 0.01 * torch.randn_like(data)
    return [(data.to(device), targets.to(device))]
