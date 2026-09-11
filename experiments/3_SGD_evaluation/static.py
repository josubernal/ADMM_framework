import configparser
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.admm import ADMM_CrossEntropy_Taylor, ADMM_Metrics

from .utils.dataset import get_data, get_dataset
from .utils.models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("experiments/3_SGD_evaluation/config/config.ini")

seed = config.getint("config", "seed")

batch_size_static = config.getint("config", "batch_size_static")
n_batches = config.getint("config", "n_batches")
epochs = config.getint("config", "epochs")
hidden_size_static = config.getint("config", "hidden_size_static")
hidden_channels_static = config.getint("config", "hidden_channels_static")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")
lr_gd = config.getfloat("config", "learning_rate_gd")
lr_adam = config.getfloat("config", "learning_rate_adam")
lr_gd_mini = config.getfloat("config", "learning_rate_gd_mini")
lr_adam_mini = config.getfloat("config", "learning_rate_adam_mini")

run_mini_batch = config.getboolean("config", "run_mini_batch")

minibatch_size = config.getint("config", "minibatch_size")

ff_rho = config.getfloat("config", "ff_rho")
ff_beta = config.getfloat("config", "ff_beta")
conv_rho = config.getfloat("config", "conv_rho")
conv_beta = config.getfloat("config", "conv_beta")


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
class GDFFNet(nn.Module):
    def __init__(
        self, input_size=input_size, hidden_size=hidden_size_static, num_classes=10
    ):
        super(GDFFNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class GDConvNet(nn.Module):
    def __init__(
        self, hidden_channels=hidden_channels_static, k=k, s=s, p=p, num_classes=10
    ):
        super(GDConvNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        spatial = ((28 + 2 * p - k) // s) + 1
        lin_in = hidden_channels * spatial * spatial
        self.fc2 = nn.Linear(lin_in, num_classes, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc2(x)
        return x


#########################################
# AUTOMATED ITERATION OVER MODELS
#########################################
# Un-commented array to loop through all models
model_types = ["feedforward", "conv"]

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
        batch_size=batch_size_static,
        device=device,
        seed=seed,
    )

    #########################################
    # MODEL INSTANTIATION
    match model_name:
        case "feedforward":
            model = GDFFNet().to(device)
        case "conv":
            model = GDConvNet().to(device)

    match model_name:
        case "feedforward":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
                block_method="two-block",
            )
        case "conv":
            batch_size = batch_size_static
            admm_model = get_model(
                model_name=model_name,
                init="pytorch",
                train_method="vectorized",
                layer_order="backwards",
                loss=ADMM_CrossEntropy_Taylor(),
                z_first=False,
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
    # DATA
    images, labels = get_data(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        n_timesteps=n_timesteps,
        seed=seed,
    )

    if labels.dim() > 1:
        labels = labels.argmax(dim=1)

    optimizer = optim.Adam(model.parameters(), lr=lr_adam)

    adam_accs = []
    adam_losses = []
    adam_frs = []
    adam_f1s = []
    adam_times = []

    print("\nTraining model with gradient descent...")
    start_time = time.time()
    for epoch in range(epochs):
        # Extract outputs and conditionally extract firing rates

        outputs = model(images)
        frs = [float("nan")]

        loss = criterion(outputs, labels.long())

        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100

        # F1 Score Calculation
        classes = torch.unique(torch.cat((labels, predictions)))
        f1_sum = 0.0

        for c in classes:
            # Calculate True Positives, False Positives, and False Negatives for class 'c'
            tp = ((predictions == c) & (labels == c)).sum().float()
            fp = ((predictions == c) & (labels != c)).sum().float()
            fn = ((predictions != c) & (labels == c)).sum().float()

            # Calculate F1 for this class (1e-8 is an epsilon to prevent division by zero)
            f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # Macro F1 average across all found classes
        f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append metrics
        adam_accs.append(accuracy)
        adam_losses.append(loss.item())
        adam_frs.append(frs)
        adam_f1s.append(f1)
        elapsed_time = time.time() - start_time
        adam_times.append(elapsed_time)

        print(f"Step {epoch + 1} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")

    #########################################
    # MINI-BATCH GRADIENT DESCENT TRAINING LOOP
    if run_mini_batch:
        print("\nTraining model with gradient descent (Mini-batch)...")

        # Re-initialize the model to ensure a fresh start
        match model_name:
            case "feedforward":
                model_mini = GDFFNet().to(device)
            case "conv":
                model_mini = GDConvNet().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_mini = optim.Adam(model_mini.parameters(), lr=lr_adam_mini)

        # Determine which dimension holds the batch size
        batch_dim = 0
        total_samples = images.size(batch_dim)

        mini_adam_accs = []
        mini_adam_losses = []
        mini_adam_frs = []
        mini_adam_f1s = []
        mini_adam_times = []

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            all_preds = []
            all_labels = []
            epoch_frs = []

            # Create a randomized permutation of indices for shuffling data each epoch
            indices = torch.randperm(total_samples)

            # Slice the existing batch into mini-batches
            for start_idx in range(0, total_samples, minibatch_size):
                end_idx = min(start_idx + minibatch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]

                # Slice appropriately based on tensor geometry
                if batch_dim == 1:
                    batch_images = images[:, batch_indices, :]
                else:
                    batch_images = images[batch_indices, :]

                batch_labels = labels[batch_indices]

                # Forward pass
                outputs = model_mini(batch_images)
                epoch_frs.append(float("nan"))

                loss = criterion(outputs, batch_labels.long())

                # Backward pass
                optimizer_mini.zero_grad()
                loss.backward()
                optimizer_mini.step()

                # Accumulate metrics
                epoch_loss += loss.item() * batch_images.size(batch_dim)
                _, predictions = torch.max(outputs, 1)
                epoch_correct += (predictions == batch_labels).sum().item()
                epoch_total += batch_labels.size(0)

                all_preds.extend(predictions.cpu().tolist())
                all_labels.extend(batch_labels.cpu().tolist())

            # --- EPOCH LEVEL AGGREGATION ---
            avg_epoch_loss = epoch_loss / epoch_total
            epoch_accuracy = (epoch_correct / epoch_total) * 100

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
        case "feedforward":
            model_sgd = GDFFNet().to(device)
        case "conv":
            model_sgd = GDConvNet().to(device)

    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr_gd)

    sgd_accs = []
    sgd_losses = []
    sgd_frs = []
    sgd_f1s = []
    sgd_times = []

    print("\nTraining model with bare SGD (Full-batch)...")
    start_time = time.time()
    for epoch in range(epochs):
        outputs = model_sgd(images)
        frs = [float("nan")]

        loss = criterion(outputs, labels.long())

        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0)) * 100

        classes = torch.unique(torch.cat((labels, predictions)))
        f1_sum = 0.0
        for c in classes:
            tp = ((predictions == c) & (labels == c)).sum().float()
            fp = ((predictions == c) & (labels != c)).sum().float()
            fn = ((predictions != c) & (labels == c)).sum().float()
            f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

        f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()

        sgd_accs.append(accuracy)
        sgd_losses.append(loss.item())
        sgd_frs.append(frs)
        sgd_f1s.append(f1)
        sgd_times.append(time.time() - start_time)

        print(f"Step {epoch + 1} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")

    #########################################
    # MINI-BATCH GRADIENT DESCENT TRAINING LOOP (Mini-Batch SGD)
    if run_mini_batch:
        print("\nTraining model with bare SGD (Mini-batch)...")

        match model_name:
            case "feedforward":
                model_mini_sgd = GDFFNet().to(device)
            case "conv":
                model_mini_sgd = GDConvNet().to(device)

        optimizer_mini_sgd = optim.SGD(model_mini_sgd.parameters(), lr=lr_gd_mini)

        total_samples = images.size(batch_dim)

        mini_sgd_accs = []
        mini_sgd_losses = []
        mini_sgd_frs = []
        mini_sgd_f1s = []
        mini_sgd_times = []

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            all_preds = []
            all_labels = []
            epoch_frs = []

            indices = torch.randperm(total_samples)

            for start_idx in range(0, total_samples, minibatch_size):
                end_idx = min(start_idx + minibatch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]

                if batch_dim == 1:
                    batch_images = images[:, batch_indices, :]
                else:
                    batch_images = images[batch_indices, :]

                batch_labels = labels[batch_indices]

                outputs = model_mini_sgd(batch_images)
                epoch_frs.append(float("nan"))

                loss = criterion(outputs, batch_labels.long())

                # FIXED: Using the explicit mini-batch SGD optimizer
                optimizer_mini_sgd.zero_grad()
                loss.backward()
                optimizer_mini_sgd.step()

                epoch_loss += loss.item() * batch_images.size(batch_dim)
                _, predictions = torch.max(outputs, 1)
                epoch_correct += (predictions == batch_labels).sum().item()
                epoch_total += batch_labels.size(0)

                all_preds.extend(predictions.cpu().tolist())
                all_labels.extend(batch_labels.cpu().tolist())

            avg_epoch_loss = epoch_loss / epoch_total
            epoch_accuracy = (epoch_correct / epoch_total) * 100

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
