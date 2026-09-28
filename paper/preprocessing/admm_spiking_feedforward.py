import configparser
import json
from pathlib import Path

import numpy as np
import tonic
import tonic.transforms as tr
import torch
import zarr
from numcodecs import Blosc

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

    archive_path = Path(trainset.location_on_system) / trainset.filename

    if archive_path.exists():
        archive_path.unlink()
        print(f"Deleted downloaded archive: {archive_path}")

    num_samples = len(trainset)

    print(f"Raw dataset contains {num_samples} samples.")
    print(f"Batch size: {batch_size}")
    print(f"Model: {model_name}")
    print(f"Timesteps: {n_timesteps}")

    # -----------------------------------------------------------------
    # Final processed cache
    # -----------------------------------------------------------------

    cache_path = (
        Path(CACHE_ROOT) / f"temp_admm_{model_name}_{batch_size}_{n_timesteps}_{seed}"
    )

    cache_path.mkdir(parents=True, exist_ok=True)

    zarr_path = cache_path / "data.zarr"
    targets_path = cache_path / "targets.npy"
    metadata_path = cache_path / "metadata.json"
    marker = cache_path / "complete"

    if marker.exists():
        print(f"Processed cache already exists: {cache_path}")
        return

    # -----------------------------------------------------------------
    # Deterministic RNG
    # -----------------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(seed)

    data_zarr = None
    targets_memmap = None

    num_batches = (num_samples + batch_size - 1) // batch_size

    # -----------------------------------------------------------------
    # Process one batch at a time
    # -----------------------------------------------------------------

    for batch_id, start in enumerate(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)

        print(f"\nBatch {batch_id + 1}/{num_batches} [{start}:{end}]")

        # -------------------------------------------------------------
        # Remember raw files before loading
        # -------------------------------------------------------------

        raw_files = [Path(trainset.data[idx]) for idx in range(start, end)]

        # -------------------------------------------------------------
        # Load and process batch
        # -------------------------------------------------------------

        batch = [trainset[idx] for idx in range(start, end)]

        # Delete raw files immediately after loading
        deleted = 0

        for raw_file in raw_files:
            if raw_file.exists():
                raw_file.unlink()
                deleted += 1

        print(f"Deleted {deleted} raw files.")

        data, targets = tonic.collation.PadTensors()(batch)

        data = data.view(
            data.size(0),
            data.size(1),
            -1,
        )

        # Limit timesteps
        if data.size(1) > n_timesteps:
            data = data[:, :n_timesteps, ...]

        # [B, T, F]
        data = data.contiguous()

        # -------------------------------------------------------------
        # Determine storage dtype on first batch
        # -------------------------------------------------------------

        if data_zarr is None:
            max_value = data.max().item()

            print(f"Maximum event count: {max_value}")

            if max_value <= np.iinfo(np.uint8).max:
                storage_dtype = np.uint8
            elif max_value <= np.iinfo(np.uint16).max:
                storage_dtype = np.uint16
            else:
                raise ValueError(f"Event count {max_value} is too large for uint16.")

            print(f"Using storage dtype: {storage_dtype}")

            data_shape = (
                num_samples,
                data.shape[1],  # T
                data.shape[2],  # F
            )

            # ---------------------------------------------------------
            # Zarr array
            # ---------------------------------------------------------

            data_zarr = zarr.open(
                zarr_path,
                mode="w",
                shape=data_shape,
                chunks=(
                    batch_size,
                    data.shape[1],
                    data.shape[2],
                ),
                dtype=storage_dtype,
                compressor=Blosc(
                    cname="lz4",
                    clevel=1,
                    shuffle=Blosc.BITSHUFFLE,
                ),
            )

            # ---------------------------------------------------------
            # Targets
            # ---------------------------------------------------------

            targets_memmap = np.lib.format.open_memmap(
                targets_path,
                mode="w+",
                dtype=np.int64,
                shape=(num_samples,),
            )

            metadata = {
                "num_samples": num_samples,
                "batch_size": batch_size,
                "data_shape": list(data_shape),
                "targets_shape": [num_samples],
                "dtype": str(np.dtype(storage_dtype)),
                "targets_dtype": "int64",
                "model_name": model_name,
                "n_timesteps": n_timesteps,
                "seed": seed,
                "noise_std": noise_std,
                "storage": "zarr",
                "compression": "lz4",
            }

            with open(metadata_path, "w") as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                )

        # -------------------------------------------------------------
        # Save CLEAN data
        # -------------------------------------------------------------

        data_numpy = data.numpy()

        data_zarr[start:end] = data_numpy
        targets_memmap[start:end] = targets.numpy()

        print(f"Saved clean batch [{start}:{end}]")

        # -------------------------------------------------------------
        # Flush targets
        # -------------------------------------------------------------

        targets_memmap.flush()

        # -------------------------------------------------------------
        # Free memory
        # -------------------------------------------------------------

        del batch
        del data
        del data_numpy
        del targets

    # -----------------------------------------------------------------
    # Finish
    # -----------------------------------------------------------------

    targets_memmap.flush()

    del data_zarr
    del targets_memmap

    marker.touch()

    print("\nPreprocessing completed successfully.")
    print(f"Processed cache: {cache_path}")

    # -----------------------------------------------------------------
    # Remove empty raw dataset directory
    # -----------------------------------------------------------------

    train_folder = Path(trainset.location_on_system) / trainset.folder_name

    if train_folder.exists():
        try:
            train_folder.rmdir()
            print(f"Removed empty directory: {train_folder}")
        except OSError:
            pass


if __name__ == "__main__":
    preprocess_admm_spiking()
