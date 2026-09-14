import configparser
import json
import os
import time

import snntorch as snn
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.dataset import get_dataset_giovanni

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

    train_loader = get_dataset_giovanni(
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
    # SAVING RESULTS AND PLOTTING
    metrics = {}
    metrics["architecture"] = model_name
    metrics["epochs"] = epochs
    metrics["seed"] = seed

    # Full-Batch adam Metrics
    metrics["adam_accuracy"] = adam_accs
    metrics["adam_loss"] = adam_losses
    metrics["adam_firing_rate"] = adam_frs
    metrics["adam_time"] = adam_times
    metrics["adam_f1"] = adam_f1s

    metrics["sgd_accuracy"] = sgd_accs
    metrics["sgd_loss"] = sgd_losses
    metrics["sgd_firing_rate"] = sgd_frs
    metrics["sgd_time"] = sgd_times
    metrics["sgd_f1"] = sgd_f1s

    metrics_filename = f"experiments/3_SGD_evaluation/results/{model_name}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
