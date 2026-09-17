import configparser
import json
import os
import time

import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.dataset import get_dataset_minibatch

# ============================================================
# CONFIG
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("experiments/3_SGD_evaluation/config/config.ini")

seed = config.getint("config", "seed")

n_batches = config.getint("config", "n_batches")
batch_size_spiking = config.getint("config", "batch_size_spiking")
minibatch_size = config.getint("config", "minibatch_size")

epochs = config.getint("config", "epochs")

hidden_size_spiking = config.getint("config", "hidden_size_spiking")
hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")

k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")

lr_gd_mini = config.getfloat("config", "learning_rate_gd_mini")
lr_adam_mini = config.getfloat("config", "learning_rate_adam_mini")

ffdeltas = config.getfloat("config", "ffdeltas")
ffthetas = config.getfloat("config", "ffthetas")

convdeltas = config.getfloat("config", "convdeltas")
convthetas = config.getfloat("config", "convthetas")

n_timesteps = config.getint("config", "n_timesteps")


# ============================================================
# EXPERIMENT SIZE
# ============================================================

# Your intended dataset:
# 20 batches * 3000 samples = 60000 samples
total_samples = n_batches * batch_size_spiking


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODELS
# ============================================================


class GDSpFFNet(nn.Module):
    def __init__(
        self,
        input_size=34 * 34 * 2,
        hidden_size=hidden_size_spiking,
        num_classes=10,
        beta=ffdeltas,
        threshold=ffthetas,
        num_steps=n_timesteps,
    ):
        super().__init__()

        self.num_steps = num_steps

        self.fc1 = nn.Linear(
            input_size,
            hidden_size,
            bias=False,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            num_classes,
            bias=False,
        )

        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
        )

        self.lif2 = snn.Leaky(
            beta=beta,
            reset_mechanism="none",
        )

    def forward(self, x, return_spikes=False):
        """
        x shape:
            [batch, time, features]
        """

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        if return_spikes:
            fr_sum = torch.zeros(
                (),
                device=x.device,
                dtype=torch.float32,
            )

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step])

            spk1, mem1 = self.lif1(
                cur1,
                mem1,
            )

            if return_spikes:
                # Detach because firing-rate statistics do not
                # participate in the optimization graph.
                fr_sum += spk1.detach().float().mean()

            cur2 = self.fc2(spk1)

            _, mem2 = self.lif2(
                cur2,
                mem2,
            )

        if return_spikes:
            fr1 = fr_sum / self.num_steps
            return mem2, [fr1]

        return mem2


class GDSpConvNet(nn.Module):
    def __init__(
        self,
        hidden_channels=hidden_channels_spiking,
        num_classes=10,
        beta=convdeltas,
        threshold=convthetas,
        num_steps=n_timesteps,
    ):
        super().__init__()

        self.num_steps = num_steps

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )

        spatial = ((34 + 2 * p - k) // s) + 1

        lin_in = hidden_channels * spatial * spatial

        self.flatten = nn.Flatten()

        self.fc2 = nn.Linear(
            lin_in,
            num_classes,
            bias=False,
        )

        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
        )

        self.lif2 = snn.Leaky(
            beta=beta,
            reset_mechanism="none",
        )

    def forward(self, x, return_spikes=False):
        """
        x shape:
            [batch, time, channels, height, width]
        """

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        if return_spikes:
            fr_sum = torch.zeros(
                (),
                device=x.device,
                dtype=torch.float32,
            )

        for step in range(self.num_steps):
            cur1 = self.conv(x[:, step])

            spk1, mem1 = self.lif1(
                cur1,
                mem1,
            )

            if return_spikes:
                fr_sum += spk1.detach().float().mean()

            spk1_flat = self.flatten(spk1)

            cur2 = self.fc2(spk1_flat)

            _, mem2 = self.lif2(
                cur2,
                mem2,
            )

        if return_spikes:
            fr1 = fr_sum / self.num_steps
            return mem2, [fr1]

        return mem2


# ============================================================
# MODEL FACTORY
# ============================================================


def create_model(model_name):

    if model_name == "spiking-feedforward":
        return GDSpFFNet().to(device)

    if model_name == "spiking-conv":
        return GDSpConvNet().to(device)

    raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_batch(
    batch_images,
    batch_labels,
    model_name,
):
    """
    DataLoader returns:

        [batch, time, ...]

    The models expect exactly the same batch-first format,
    so no full-batch transpose/copy is needed.
    """

    # Truncate time BEFORE sending data to the GPU.
    batch_images = batch_images[:, :n_timesteps]

    if model_name == "spiking-feedforward":
        batch_images = batch_images.reshape(
            batch_images.size(0),
            batch_images.size(1),
            -1,
        )

    elif model_name == "spiking-conv":
        # Keep [B, T, C, H, W]
        pass

    else:
        raise ValueError(f"Unknown model: {model_name}")

    batch_images = batch_images.to(
        device,
        non_blocking=True,
    )

    batch_labels = batch_labels.to(
        device,
        non_blocking=True,
    )

    return batch_images, batch_labels


# ============================================================
# F1 COMPUTATION
# ============================================================


def compute_macro_f1(confusion_matrix):

    tp = torch.diag(confusion_matrix)

    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp

    denominator = 2 * tp + fp + fn

    f1 = torch.zeros_like(
        tp,
        dtype=torch.float32,
    )

    valid = denominator > 0

    f1[valid] = 2 * tp[valid].float() / denominator[valid].float()

    # Only classes occurring in either predictions or labels,
    # matching the logic in your original implementation.
    present = (confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1)) > 0

    if present.any():
        return f1[present].mean().item()

    return 0.0


# ============================================================
# TRAINING
# ============================================================


def train_model(
    model,
    optimizer,
    model_name,
    train_loader,
    experiment_name,
):
    criterion = nn.CrossEntropyLoss()

    accuracies = []
    losses = []
    firing_rates = []
    f1_scores = []
    times = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        epoch_loss = torch.zeros(
            (),
            device=device,
        )

        epoch_correct = torch.zeros(
            (),
            device=device,
        )

        epoch_firing_rate = torch.zeros(
            (),
            device=device,
        )

        confusion_matrix = torch.zeros(
            (10, 10),
            dtype=torch.long,
            device=device,
        )

        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = prepare_batch(
                batch_images,
                batch_labels,
                model_name,
            )

            current_batch_size = batch_labels.size(0)

            optimizer.zero_grad(set_to_none=True)

            outputs, frs = model(
                batch_images,
                return_spikes=True,
            )

            loss = criterion(
                outputs,
                batch_labels.long(),
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            predictions = outputs.argmax(dim=1)

            epoch_loss += loss.detach() * current_batch_size

            epoch_correct += (predictions == batch_labels).sum()

            epoch_firing_rate += frs[0].detach() * current_batch_size

            # Confusion matrix
            encoded = batch_labels * 10 + predictions

            confusion_matrix += torch.bincount(
                encoded,
                minlength=100,
            ).reshape(10, 10)

        # ----------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------

        avg_epoch_loss = (epoch_loss / total_samples).item()

        epoch_accuracy = (epoch_correct / total_samples * 100).item()

        avg_fr = (epoch_firing_rate / total_samples).item()

        epoch_f1 = compute_macro_f1(confusion_matrix)

        elapsed = time.time() - start_time

        losses.append(avg_epoch_loss)
        accuracies.append(epoch_accuracy)
        firing_rates.append([avg_fr])
        f1_scores.append(epoch_f1)
        times.append(elapsed)

        print(
            f"{experiment_name} | "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_epoch_loss:.4f} | "
            f"Acc: {epoch_accuracy:.4f} | "
            f"F1: {epoch_f1:.4f}"
        )

    return {
        "accuracy": accuracies,
        "loss": losses,
        "firing_rate": firing_rates,
        "f1": f1_scores,
        "time": times,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

model_types = [
    "spiking-feedforward",
    "spiking-conv",
]

for model_name in model_types:
    print("\n" + "=" * 60)
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # ADAM
    # --------------------------------------------------------

    print("\nTraining with Adam...")

    set_seed(seed)

    # Important:
    # 3000 x 20 = 60000 total samples,
    # but the DataLoader itself uses 200-sample batches.
    train_loader_adam = get_dataset_minibatch(
        model_name=model_name,
        batch_size=minibatch_size,
        seed=seed,
    )

    model_adam = create_model(model_name)

    optimizer_adam = optim.Adam(
        model_adam.parameters(),
        lr=lr_adam_mini,
    )

    adam_results = train_model(
        model=model_adam,
        optimizer=optimizer_adam,
        model_name=model_name,
        train_loader=train_loader_adam,
        experiment_name="Adam",
    )

    del model_adam
    del optimizer_adam
    del train_loader_adam

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # SGD
    # --------------------------------------------------------

    print("\nTraining with SGD...")

    # Same initialization as Adam for a fair comparison.
    set_seed(seed)

    train_loader_sgd = get_dataset_minibatch(
        model_name=model_name,
        batch_size=minibatch_size,
        seed=seed,
    )

    model_sgd = create_model(model_name)

    optimizer_sgd = optim.SGD(
        model_sgd.parameters(),
        lr=lr_gd_mini,
    )

    sgd_results = train_model(
        model=model_sgd,
        optimizer=optimizer_sgd,
        model_name=model_name,
        train_loader=train_loader_sgd,
        experiment_name="SGD",
    )

    del model_sgd
    del optimizer_sgd
    del train_loader_sgd

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    metrics = {
        "architecture": model_name,
        "epochs": epochs,
        "seed": seed,
        "mini_adam_accuracy": adam_results["accuracy"],
        "mini_adam_loss": adam_results["loss"],
        "mini_adam_firing_rate": adam_results["firing_rate"],
        "mini_adam_time": adam_results["time"],
        "mini_adam_f1": adam_results["f1"],
        "mini_sgd_accuracy": sgd_results["accuracy"],
        "mini_sgd_loss": sgd_results["loss"],
        "mini_sgd_firing_rate": sgd_results["firing_rate"],
        "mini_sgd_time": sgd_results["time"],
        "mini_sgd_f1": sgd_results["f1"],
    }

    metrics_filename = f"experiments/3_SGD_evaluation/results/{model_name}/results.json"

    os.makedirs(
        os.path.dirname(metrics_filename),
        exist_ok=True,
    )

    with open(
        metrics_filename,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(f"\nSaved results to {metrics_filename}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
