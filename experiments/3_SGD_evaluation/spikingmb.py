import configparser
import json
import os
import time

import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim

from src.admm import ADMM_CrossEntropy_Taylor, ADMM_Metrics

from .utils.dataset import get_dataset
from .utils.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("experiments/3_SGD_evaluation/config/config.ini")

seed = config.getint("config", "seed")

n_batches = config.getint("config", "n_batches")
batch_size_spiking = config.getint("config", "batch_size_spiking")
epochs = config.getint("config", "epochs")
hidden_size_spiking = config.getint("config", "hidden_size_spiking")
hidden_channels_spiking = config.getint("config", "hidden_channels_spiking")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")
lr_gd = config.getfloat("config", "learning_rate_gd")
lr_adam = config.getfloat("config", "learning_rate_adam")
lr_gd_mini = config.getfloat("config", "learning_rate_gd_mini")
lr_adam_mini = config.getfloat("config", "learning_rate_adam_mini")

run_mini_batch = config.getboolean("config", "run_mini_batch")

minibatch_size = config.getint("config", "minibatch_size")
spff_rho = config.getfloat("config", "spff_rho")
spff_beta = config.getfloat("config", "spff_beta")
spconv_rho = config.getfloat("config", "spconv_rho")
spconv_beta = config.getfloat("config", "spconv_beta")

ffdeltas = config.getfloat("config", "ffdeltas")
ffthetas = config.getfloat("config", "ffthetas")
convdeltas = config.getfloat("config", "convdeltas")
convthetas = config.getfloat("config", "convthetas")
input_size = 784
n_timesteps = config.getint("config", "n_timesteps")

# WARMING CONFIG
warming_iters = config.getint("config", "warming_iters")


def calc_spatial_out(size_in, k, p, s):
    return ((size_in + 2 * p - k) // s) + 1


########################################
# MODELS
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
        super(GDSpFFNet, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x, return_spikes=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        if return_spikes:
            spikes_layer1 = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            if return_spikes:
                spikes_layer1.append(spk1)

            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)

        if return_spikes:
            # Calculate mean firing rate across time, batch, and neurons
            fr1 = torch.stack(spikes_layer1).mean().item()
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
        super(GDSpConvNet, self).__init__()
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
        self.fc2 = nn.Linear(lin_in, num_classes, bias=False)

        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")

    def forward(self, x, return_spikes=False):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        if return_spikes:
            spikes_layer1 = []

        for step in range(self.num_steps):
            cur1 = self.conv(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            if return_spikes:
                spikes_layer1.append(spk1)

            spk1_flat = self.flatten(spk1)
            cur2 = self.fc2(spk1_flat)
            _, mem2 = self.lif2(cur2, mem2)

        if return_spikes:
            # Calculate mean firing rate across time, batch, and spatial dimensions
            fr1 = torch.stack(spikes_layer1).mean().item()
            return mem2, [fr1]

        return mem2


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["spiking-feedforward", "spiking-conv"]

for model_name in model_types:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n{'=' * 50}")
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print(f"{'=' * 50}")

    #########################################
    # DATA

    train_loader = get_dataset(
        model_name=model_name,
        batch_size=batch_size_spiking,
        device=device,
        n_timesteps=n_timesteps,
        seed=seed,
        n_batches=n_batches,
    )

    #########################################
    # MODEL INSTANTIATION
    match model_name:
        case "spiking-feedforward":
            model = GDSpFFNet().to(device)
        case "spiking-conv":
            model = GDSpConvNet().to(device)

    match model_name:
        case "spiking-feedforward":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled",
                time_order="backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=True,
                block_method="two-block",
            )
        case "spiking-conv":
            batch_size = batch_size_spiking
            admm_model = get_model(
                model_name=model_name,
                init="s-uniform",
                train_method="decoupled",
                time_order="backwards",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=True,
                block_method="two-block",
            )

    #########################################
    # ADMM TRAINING LOOP

    m = ADMM_Metrics(admm_model)

    admm_times = []
    start_time = time.time()

    for epoch in range(epochs + 1):
        admm_model.fit(train_loader, warming=epoch < warming_iters)
        if epoch % 1 == 0:
            with torch.no_grad():
                m.save_metrics()
                elapsed_time = time.time() - start_time
                admm_times.append(elapsed_time)
                print(
                    f"Epoch {epoch} | W_norm: {admm_model.layers[0].W.norm().item():.2f}"
                )
                print(f"Epoch [{epoch:3d}/{epochs}] | {m}")

    #########################################
    # ADAM GRADIENT DESCENT TRAINING LOOP
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr_adam)

    adam_accs = []
    adam_losses = []
    adam_frs = []
    adam_f1s = []
    adam_times = []

    print("\nTraining model with gradient descent (Accumulated Full-Batch)...")
    start_time = time.time()

    # Determine which dimension holds the batch size
    batch_dim = 1 if model_name in ["spiking-feedforward", "spiking-conv"] else 0
    total_samples = sum([b[0].size(batch_dim) for b in train_loader])

    for epoch in range(epochs):
        optimizer.zero_grad()

        epoch_loss = 0.0
        epoch_correct = 0
        all_preds = []
        all_labels = []
        epoch_frs = []

        # Iterate through batches to accumulate gradients
        for batch_images, batch_labels in train_loader:
            if model_name in ["spiking-feedforward", "spiking-conv"]:
                outputs, frs = model(batch_images, return_spikes=True)
                epoch_frs.append(frs[0])
            else:
                outputs = model(batch_images)
                epoch_frs.append(float("nan"))

            loss = criterion(outputs, batch_labels.long())

            # Scale the loss so the accumulated gradients average over the whole dataset
            batch_size_current = batch_images.size(batch_dim)
            scaled_loss = loss * (batch_size_current / total_samples)
            scaled_loss.backward()

            # Accumulate metrics
            epoch_loss += loss.item() * batch_size_current
            _, predictions = torch.max(outputs, 1)
            epoch_correct += (predictions == batch_labels).sum().item()

            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

        # Perform the single parameter update for the full dataset read
        optimizer.step()

        # Calculate epoch-level metrics
        avg_loss = epoch_loss / total_samples
        accuracy = (epoch_correct / total_samples) * 100

        # F1 Score Calculation
        preds_tensor = torch.tensor(all_preds)
        labels_tensor = torch.tensor(all_labels)
        classes = torch.unique(torch.cat((labels_tensor, preds_tensor)))
        f1_sum = 0.0

        for c in classes:
            tp = ((preds_tensor == c) & (labels_tensor == c)).sum().float()
            fp = ((preds_tensor == c) & (labels_tensor != c)).sum().float()
            fn = ((preds_tensor != c) & (labels_tensor == c)).sum().float()
            f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

        f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

        import math

        avg_fr = (
            float("nan")
            if math.isnan(epoch_frs[0])
            else sum(epoch_frs) / len(epoch_frs)
        )

        # Append metrics
        adam_accs.append(accuracy)
        adam_losses.append(avg_loss)
        adam_frs.append([avg_fr])
        adam_f1s.append(f1)

        elapsed_time = time.time() - start_time
        adam_times.append(elapsed_time)

        print(f"Step {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    #########################################
    # MINI-BATCH GRADIENT DESCENT TRAINING LOOP
    if run_mini_batch:
        print("\nTraining model with gradient descent (Mini-batch)...")

        # Re-initialize the model to ensure a fresh start
        match model_name:
            case "spiking-feedforward":
                model_mini = GDSpFFNet().to(device)
            case "spiking-conv":
                model_mini = GDSpConvNet().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_mini = optim.Adam(model_mini.parameters(), lr=lr_adam_mini)

        # Determine which dimension holds the batch size
        batch_dim = 1 if model_name in ["spiking-feedforward", "spiking-conv"] else 0
        total_samples = sum([b[0].size(batch_dim) for b in train_loader])

        mini_adam_accs = []
        mini_adam_losses = []
        mini_adam_frs = []
        mini_adam_f1s = []
        mini_adam_times = []

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            all_preds = []
            all_labels = []
            epoch_frs = []

            # Iterate over the data loader
            for batch_images, batch_labels in train_loader:
                current_batch_size = batch_images.size(batch_dim)
                # Create a randomized permutation of indices for shuffling data within this batch chunk
                indices = torch.randperm(current_batch_size)

                # Slice the existing batch into mini-batches based on config
                for start_idx in range(0, current_batch_size, minibatch_size):
                    end_idx = min(start_idx + minibatch_size, current_batch_size)
                    batch_indices = indices[start_idx:end_idx]

                    # Slice appropriately based on tensor geometry
                    if batch_dim == 1:
                        mini_images = batch_images[:, batch_indices, :]
                    else:
                        mini_images = batch_images[batch_indices, :]

                    mini_labels = batch_labels[batch_indices]

                    # Zero gradients
                    optimizer_mini.zero_grad()

                    # Forward pass
                    if model_name in ["spiking-feedforward", "spiking-conv"]:
                        outputs, frs = model_mini(mini_images, return_spikes=True)
                        epoch_frs.append(frs[0])
                    else:
                        outputs = model_mini(mini_images)
                        epoch_frs.append(float("nan"))

                    loss = criterion(outputs, mini_labels.long())

                    # Backward pass and update per mini-batch chunk
                    loss.backward()
                    optimizer_mini.step()

                    # Accumulate metrics
                    epoch_loss += loss.item() * mini_images.size(batch_dim)
                    _, predictions = torch.max(outputs, 1)
                    epoch_correct += (predictions == mini_labels).sum().item()

                    all_preds.extend(predictions.cpu().tolist())
                    all_labels.extend(mini_labels.cpu().tolist())

            # --- EPOCH LEVEL AGGREGATION ---
            avg_epoch_loss = epoch_loss / total_samples
            epoch_accuracy = (epoch_correct / total_samples) * 100

            # F1 Score
            preds_tensor = torch.tensor(all_preds)
            labels_tensor = torch.tensor(all_labels)
            classes = torch.unique(torch.cat((labels_tensor, preds_tensor)))
            f1_sum = 0.0

            for c in classes:
                tp = ((preds_tensor == c) & (labels_tensor == c)).sum().float()
                fp = ((preds_tensor == c) & (labels_tensor != c)).sum().float()
                fn = ((preds_tensor != c) & (labels_tensor == c)).sum().float()
                f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

            epoch_f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

            import math

            avg_fr = (
                float("nan")
                if math.isnan(epoch_frs[0])
                else sum(epoch_frs) / len(epoch_frs)
            )

            mini_adam_accs.append(epoch_accuracy)
            mini_adam_losses.append(avg_epoch_loss)
            mini_adam_frs.append([avg_fr])
            mini_adam_f1s.append(epoch_f1)

            elapsed_time = time.time() - start_time
            mini_adam_times.append(elapsed_time)

            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f}"
            )
    #########################################
    # GRADIENT DESCENT TRAINING LOOP (Full-Batch SGD)
    match model_name:
        case "spiking-feedforward":
            model_sgd = GDSpFFNet().to(device)
        case "spiking-conv":
            model_sgd = GDSpConvNet().to(device)

    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr_gd)

    sgd_accs = []
    sgd_losses = []
    sgd_frs = []
    sgd_f1s = []
    sgd_times = []

    print("\nTraining model with bare SGD (Accumulated Full-batch)...")
    start_time = time.time()

    for epoch in range(epochs):
        optimizer_sgd.zero_grad()

        epoch_loss = 0.0
        epoch_correct = 0
        all_preds = []
        all_labels = []
        epoch_frs = []

        # Iterate through batches to accumulate gradients
        for batch_images, batch_labels in train_loader:
            if model_name in ["spiking-feedforward", "spiking-conv"]:
                outputs, frs = model_sgd(batch_images, return_spikes=True)
                epoch_frs.append(frs[0])
            else:
                outputs = model_sgd(batch_images)
                epoch_frs.append(float("nan"))

            loss = criterion(outputs, batch_labels.long())

            # Scale the loss for full-batch gradient accumulation
            batch_size_current = batch_images.size(batch_dim)
            scaled_loss = loss * (batch_size_current / total_samples)
            scaled_loss.backward()

            # Accumulate metrics
            epoch_loss += loss.item() * batch_size_current
            _, predictions = torch.max(outputs, 1)
            epoch_correct += (predictions == batch_labels).sum().item()

            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

        # Perform the single parameter update
        optimizer_sgd.step()

        # Calculate epoch-level metrics
        avg_loss = epoch_loss / total_samples
        accuracy = (epoch_correct / total_samples) * 100

        preds_tensor = torch.tensor(all_preds)
        labels_tensor = torch.tensor(all_labels)
        classes = torch.unique(torch.cat((labels_tensor, preds_tensor)))
        f1_sum = 0.0
        for c in classes:
            tp = ((preds_tensor == c) & (labels_tensor == c)).sum().float()
            fp = ((preds_tensor == c) & (labels_tensor != c)).sum().float()
            fn = ((preds_tensor != c) & (labels_tensor == c)).sum().float()
            f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

        f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

        import math

        avg_fr = (
            float("nan")
            if math.isnan(epoch_frs[0])
            else sum(epoch_frs) / len(epoch_frs)
        )

        sgd_accs.append(accuracy)
        sgd_losses.append(avg_loss)
        sgd_frs.append([avg_fr])
        sgd_f1s.append(f1)
        sgd_times.append(time.time() - start_time)

        print(f"Step {epoch + 1} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    #########################################
    # MINI-BATCH GRADIENT DESCENT TRAINING LOOP (Mini-Batch SGD)
    if run_mini_batch:
        print("\nTraining model with bare SGD (Mini-batch)...")

        match model_name:
            case "spiking-feedforward":
                model_mini_sgd = GDSpFFNet().to(device)
            case "spiking-conv":
                model_mini_sgd = GDSpConvNet().to(device)

        optimizer_mini_sgd = optim.SGD(model_mini_sgd.parameters(), lr=lr_gd_mini)

        batch_dim = 1 if model_name in ["spiking-feedforward", "spiking-conv"] else 0
        total_samples = sum([b[0].size(batch_dim) for b in train_loader])

        mini_sgd_accs = []
        mini_sgd_losses = []
        mini_sgd_frs = []
        mini_sgd_f1s = []
        mini_sgd_times = []

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            all_preds = []
            all_labels = []
            epoch_frs = []

            for batch_images, batch_labels in train_loader:
                current_batch_size = batch_images.size(batch_dim)
                indices = torch.randperm(current_batch_size)

                for start_idx in range(0, current_batch_size, minibatch_size):
                    end_idx = min(start_idx + minibatch_size, current_batch_size)
                    batch_indices = indices[start_idx:end_idx]

                    if batch_dim == 1:
                        mini_images = batch_images[:, batch_indices, :]
                    else:
                        mini_images = batch_images[batch_indices, :]

                    mini_labels = batch_labels[batch_indices]

                    optimizer_mini_sgd.zero_grad()

                    if model_name in ["spiking-feedforward", "spiking-conv"]:
                        outputs, frs = model_mini_sgd(mini_images, return_spikes=True)
                        epoch_frs.append(frs[0])
                    else:
                        outputs = model_mini_sgd(mini_images)
                        epoch_frs.append(float("nan"))

                    loss = criterion(outputs, mini_labels.long())
                    loss.backward()
                    optimizer_mini_sgd.step()

                    epoch_loss += loss.item() * mini_images.size(batch_dim)
                    _, predictions = torch.max(outputs, 1)
                    epoch_correct += (predictions == mini_labels).sum().item()

                    all_preds.extend(predictions.cpu().tolist())
                    all_labels.extend(mini_labels.cpu().tolist())

            avg_epoch_loss = epoch_loss / total_samples
            epoch_accuracy = (epoch_correct / total_samples) * 100

            preds_tensor = torch.tensor(all_preds)
            labels_tensor = torch.tensor(all_labels)
            classes = torch.unique(torch.cat((labels_tensor, preds_tensor)))
            f1_sum = 0.0

            for c in classes:
                tp = ((preds_tensor == c) & (labels_tensor == c)).sum().float()
                fp = ((preds_tensor == c) & (labels_tensor != c)).sum().float()
                fn = ((preds_tensor != c) & (labels_tensor == c)).sum().float()
                f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

            epoch_f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

            import math

            avg_fr = (
                float("nan")
                if math.isnan(epoch_frs[0])
                else sum(epoch_frs) / len(epoch_frs)
            )

            mini_sgd_accs.append(epoch_accuracy)
            mini_sgd_losses.append(avg_epoch_loss)
            mini_sgd_frs.append([avg_fr])
            mini_sgd_f1s.append(epoch_f1)
            mini_sgd_times.append(time.time() - start_time)

            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f}"
            )
    #########################################
    # SAVING RESULTS AND PLOTTING
    metrics = m.get_dic()
    metrics["architecture"] = model_name
    metrics["epochs"] = epochs
    metrics["seed"] = seed

    # Full-Batch adam Metrics
    metrics["adam_accuracy"] = adam_accs
    metrics["adam_loss"] = adam_losses
    metrics["adam_firing_rate"] = adam_frs
    metrics["adam_time"] = adam_times
    metrics["adam_f1"] = adam_f1s

    # Mini-Batch GD Metrics
    if run_mini_batch:
        metrics["mini_adam_accuracy"] = mini_adam_accs
        metrics["mini_adam_loss"] = mini_adam_losses
        metrics["mini_adam_firing_rate"] = mini_adam_frs
        metrics["mini_adam_time"] = mini_adam_times
        metrics["mini_adam_f1"] = mini_adam_f1s

    metrics["sgd_accuracy"] = sgd_accs
    metrics["sgd_loss"] = sgd_losses
    metrics["sgd_firing_rate"] = sgd_frs
    metrics["sgd_time"] = sgd_times
    metrics["sgd_f1"] = sgd_f1s

    # Mini-Batch GD Metrics
    if run_mini_batch:
        metrics["mini_sgd_accuracy"] = mini_sgd_accs
        metrics["mini_sgd_loss"] = mini_sgd_losses
        metrics["mini_sgd_firing_rate"] = mini_sgd_frs
        metrics["mini_sgd_time"] = mini_sgd_times
        metrics["mini_sgd_f1"] = mini_sgd_f1s

    # ADMM Metrics
    metrics["admm_time"] = admm_times

    metrics_filename = f"experiments/3_SGD_evaluation/results/{model_name}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
