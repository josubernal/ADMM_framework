import configparser
import json
from pathlib import Path

import numpy as np
import tonic
import tonic.transforms as tr

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
config = configparser.ConfigParser()

config.read("paper/config/config.ini")

RAW_ROOT = "./data"
CACHE_ROOT = "./cache/nmnist"

MODEL_NAME = "spiking-feedforward"
BATCH_SIZE = config.getint("config", "batch_size_spiking")
N_TIMESTEPS = config.getint("config", "n_timesteps")
SEED = config.getint("config", "seed")
NOISE_STD = 0.01


def pack_3bit(data: np.ndarray) -> np.ndarray:
    """
    Pack uint8 values in [0, 7] using exactly 3 bits/value.

    Input:
        (..., N), where N must be divisible by 8.

    Output:
        (..., N * 3 // 8)
    """
    if data.dtype != np.uint8:
        raise TypeError(f"Expected uint8, got {data.dtype}")

    n = data.shape[-1]

    if n % 8 != 0:
        raise ValueError(f"Last dimension must be divisible by 8, got {n}")

    x = data.reshape(-1, n)

    x = x.reshape(-1, n // 8, 8)

    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]
    x3 = x[..., 3]
    x4 = x[..., 4]
    x5 = x[..., 5]
    x6 = x[..., 6]
    x7 = x[..., 7]

    packed = np.empty(
        (*x.shape[:-1], 3),
        dtype=np.uint8,
    )

    packed[..., 0] = x0 | (x1 << 3) | (x2 << 6)

    packed[..., 1] = (x2 >> 2) | (x3 << 1) | (x4 << 4) | (x5 << 7)

    packed[..., 2] = (x5 >> 1) | (x6 << 2) | (x7 << 5)

    return packed.reshape(
        *data.shape[:-1],
        n * 3 // 8,
    )


# ---------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------


def preprocess_admm_spiking(
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    n_timesteps=N_TIMESTEPS,
    seed=SEED,
    noise_std=NOISE_STD,
):

    # -----------------------------------------------------------------
    # Final cache
    # -----------------------------------------------------------------

    cache_path = (
        Path(CACHE_ROOT) / f"temp_admm_{model_name}_{batch_size}_{n_timesteps}_{seed}"
    )

    cache_path.mkdir(parents=True, exist_ok=True)

    data_path = cache_path / "data.dat"
    targets_path = cache_path / "targets.dat"
    metadata_path = cache_path / "metadata.json"
    marker = cache_path / "complete"

    if marker.exists():
        print(f"Processed cache already exists: {cache_path}")
        return
    # -----------------------------------------------------------------

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
        save_to=RAW_ROOT,
        transform=frame_transform,
        train=True,
    )

    num_samples = len(trainset)

    print(f"Raw dataset contains {num_samples} samples.")
    print(f"Batch size: {batch_size}")
    print(f"Model: {model_name}")
    print(f"Timesteps: {n_timesteps}")

    # -----------------------------------------------------------------
    # Deterministic noise generator
    # -----------------------------------------------------------------

    data_memmap = None
    targets_memmap = None

    num_batches = (num_samples + batch_size - 1) // batch_size

    # -----------------------------------------------------------------
    # Process one batch at a time
    # -----------------------------------------------------------------

    for batch_id, start in enumerate(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)

        print(f"\nBatch {batch_id + 1}/{num_batches} [{start}:{end}]")

        # -------------------------------------------------------------
        # Remember the raw files BEFORE loading the batch.
        # -------------------------------------------------------------

        raw_files = [Path(trainset.data[idx]) for idx in range(start, end)]

        # -------------------------------------------------------------
        # Load and process only this batch
        # -------------------------------------------------------------

        batch = [trainset[idx] for idx in range(start, end)]

        deleted = 0

        for raw_file in raw_files:
            if raw_file.exists():
                raw_file.unlink()
                deleted += 1

        print(f"Deleted {deleted} raw files.")

        data, targets = tonic.collation.PadTensors()(batch)

        data = data.view(data.size(0), data.size(1), -1)

        # Limit number of timesteps
        if data.size(1) > n_timesteps:
            data = data[:, :n_timesteps, ...]

        storage_dtype = np.uint8

        # -------------------------------------------------------------
        # Convert to NumPy in [B, T, ...] form for the memmap
        # -------------------------------------------------------------

        data = (
            data.contiguous()
            .numpy()
            .astype(
                storage_dtype,
                copy=False,
            )
        )

        min_value = data.min()
        max_value = data.max()

        if min_value < 0 or max_value > 7:
            raise ValueError(
                f"3-bit packing requires values in [0, 7], "
                f"got [{min_value}, {max_value}]"
            )

        data_to_save = pack_3bit(data)
        targets_to_save = targets.numpy()

        # -------------------------------------------------------------
        # Create memmaps after seeing the first batch
        # -------------------------------------------------------------

        if data_memmap is None:
            data_shape = (
                num_samples,
                data_to_save.shape[1],  # T
                data_to_save.shape[2],  # features
            )

            data_memmap = np.memmap(
                data_path,
                dtype=np.uint8,
                mode="w+",
                shape=data_shape,
            )

            targets_memmap = np.memmap(
                targets_path,
                dtype=np.int64,
                mode="w+",
                shape=(num_samples,),
            )

            metadata = {
                "num_samples": num_samples,
                "batch_size": batch_size,
                # Shape stored on disk
                "data_shape": list(data_shape),
                # Original unpacked shape
                "original_feature_dim": int(data.shape[2]),
                "targets_shape": [num_samples],
                "dtype": "uint8",
                "targets_dtype": "int64",
                "packing": "3bit",
                "model_name": model_name,
                "n_timesteps": n_timesteps,
                "seed": seed,
                "noise_std": noise_std,
            }

            with open(metadata_path, "w") as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                )

        # -------------------------------------------------------------
        # Write processed batch
        # -------------------------------------------------------------

        data_memmap[start:end] = data_to_save
        targets_memmap[start:end] = targets_to_save

        data_memmap.flush()
        targets_memmap.flush()

        print(f"Saved batch and deleted {deleted} raw files.")

        # -------------------------------------------------------------
        # Free memory
        # -------------------------------------------------------------

        del batch
        del data
        del data_to_save
        del targets
        del targets_to_save

    # -----------------------------------------------------------------
    # Finish
    # -----------------------------------------------------------------

    data_memmap.flush()
    targets_memmap.flush()

    del data_memmap
    del targets_memmap

    # Only mark complete after EVERYTHING succeeded
    marker.touch()

    print("\nPreprocessing completed successfully.")
    print(f"Processed cache: {cache_path}")

    # Remove now-empty train directory if possible
    train_folder = Path(trainset.location_on_system) / trainset.folder_name

    if train_folder.exists():
        try:
            train_folder.rmdir()
            print(f"Removed empty directory: {train_folder}")
        except OSError:
            # Directory wasn't empty; leave it alone.
            pass


if __name__ == "__main__":
    preprocess_admm_spiking()
