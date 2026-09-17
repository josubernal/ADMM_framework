import configparser
import gc
import time

import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.dataset import get_dataset_giovanni

# ============================================================
# CONFIG
# ============================================================

config = configparser.ConfigParser()
config.read("experiments/3_SGD_evaluation/config/config.ini")

seed = config.getint("config", "seed")
n_batches = config.getint("config", "n_batches")
batch_size = config.getint("config", "batch_size_spiking")
hidden_size = config.getint("config", "hidden_size_spiking")
hidden_channels = config.getint("config", "hidden_channels_spiking")

k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")

n_timesteps = config.getint("config", "n_timesteps")

ffdeltas = config.getfloat("config", "ffdeltas")
ffthetas = config.getfloat("config", "ffthetas")
convdeltas = config.getfloat("config", "convdeltas")
convthetas = config.getfloat("config", "convthetas")

learning_rate = config.getfloat("config", "learning_rate_gd")


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DEVICE TEST")
print("=" * 70)

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")


# ============================================================
# MEMORY DEBUGGING
# ============================================================


def print_memory(label):
    if not torch.cuda.is_available():
        return

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[{label}] "
        f"allocated={allocated:.3f} GB | "
        f"reserved={reserved:.3f} GB | "
        f"max={max_allocated:.3f} GB"
    )


# ============================================================
# MODEL 1: FEEDFORWARD
# ============================================================


class GDSpFFNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_steps = n_timesteps

        self.fc1 = nn.Linear(
            784,
            hidden_size,
            bias=False,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            10,
            bias=False,
        )

        self.lif1 = snn.Leaky(
            beta=ffdeltas,
            threshold=ffthetas,
        )

        self.lif2 = snn.Leaky(
            beta=ffdeltas,
            reset_mechanism="none",
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)

        return mem2


# ============================================================
# MODEL 2: CONVOLUTIONAL
# ============================================================


class GDSpConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_steps = n_timesteps

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )

        spatial = ((34 + 2 * p - k) // s) + 1

        self.fc2 = nn.Linear(
            hidden_channels * spatial * spatial,
            10,
            bias=False,
        )

        self.flatten = nn.Flatten()

        self.lif1 = snn.Leaky(
            beta=convdeltas,
            threshold=convthetas,
        )

        self.lif2 = snn.Leaky(
            beta=convdeltas,
            reset_mechanism="none",
        )

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(self.num_steps):
            cur1 = self.conv(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            spk1_flat = self.flatten(spk1)

            cur2 = self.fc2(spk1_flat)
            _, mem2 = self.lif2(cur2, mem2)

        return mem2


# ============================================================
# START CLEAN
# ============================================================

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

gc.collect()

print_memory("initial")


# ============================================================
# TEST ONE MODEL AT A TIME
# ============================================================

model_name = "spiking-feedforward"
# model_name = "spiking-conv"


print()
print("=" * 70)
print(f"TESTING: {model_name}")
print("=" * 70)


# ============================================================
# DATASET
# ============================================================

print()
print("Loading dataset...")

train_loader = get_dataset_giovanni(
    model_name=model_name,
    batch_size=batch_size,
    device=device,
    n_timesteps=n_timesteps,
    seed=seed,
    n_batches=n_batches,
)

print("Dataset loaded.")
print_memory("after dataset")


# ============================================================
# INSPECT DATA
# ============================================================

print()
print("=" * 70)
print("DATA INSPECTION")
print("=" * 70)

print("Number of batches:", len(train_loader))

for i, (images, labels) in enumerate(train_loader):
    print(f"\nBatch {i}")
    print("  images shape:", images.shape)
    print("  images dtype:", images.dtype)
    print("  images device:", images.device)

    print("  labels shape:", labels.shape)
    print("  labels dtype:", labels.dtype)
    print("  labels device:", labels.device)

    print(
        "  images memory:",
        images.numel() * images.element_size() / 1024**2,
        "MB",
    )

    print_memory(f"after inspecting batch {i}")

    if i == 0:
        break


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("MODEL CREATION")
print("=" * 70)

if model_name == "spiking-feedforward":
    model = GDSpFFNet().to(device)
else:
    model = GDSpConvNet().to(device)

print(model)

print_memory("after model")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

criterion = nn.CrossEntropyLoss()

print_memory("after optimizer")


# ============================================================
# TEST ONE BATCH
# ============================================================

print()
print("=" * 70)
print("TESTING ONE BATCH")
print("=" * 70)

batch_images, batch_labels = train_loader[0]

print("Input shape:", batch_images.shape)
print("Input device:", batch_images.device)

print_memory("before forward")


# ------------------------------------------------------------
# FORWARD
# ------------------------------------------------------------

print()
print("Running forward...")

start = time.time()

outputs = model(batch_images)

torch.cuda.synchronize() if torch.cuda.is_available() else None

print(f"Forward completed in {time.time() - start:.3f}s")

print("Output shape:", outputs.shape)
print("Output device:", outputs.device)

print_memory("after forward")


# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

print()
print("Calculating loss...")

loss = criterion(
    outputs,
    batch_labels.long(),
)

print("Loss:", loss.item())

print_memory("after loss")


# ------------------------------------------------------------
# BACKWARD
# ------------------------------------------------------------

print()
print("Running backward...")

start = time.time()

loss.backward()

torch.cuda.synchronize() if torch.cuda.is_available() else None

print(f"Backward completed in {time.time() - start:.3f}s")

print_memory("after backward")


# ============================================================
# OPTIMIZER STEP
# ============================================================

print()
print("Running optimizer step...")

optimizer.step()

print_memory("after optimizer step")


# ============================================================
# CLEANUP
# ============================================================

del outputs
del loss
del batch_images
del batch_labels
del optimizer
del model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print_memory("after cleanup")


print()
print("=" * 70)
print("TEST FINISHED")
print("=" * 70)
