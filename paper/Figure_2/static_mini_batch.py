import configparser
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.dataset import get_dataset_static_gd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read("paper/config/config.ini")

seed = config.getint("config", "seed")

epochs = config.getint("config", "epochs")
hidden_size_static = config.getint("config", "hidden_size_static")
hidden_channels_static = config.getint("config", "hidden_channels_static")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")
lr_gd_mini = config.getfloat("config", "learning_rate_gd_mini")
lr_adam_mini = config.getfloat("config", "learning_rate_adam_mini")


minibatch_size_static = config.getint("config", "minibatch_size_static")


input_size = 784


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
    # MODEL INSTANTIATION
    match model_name:
        case "feedforward":
            model = GDFFNet().to(device)
        case "conv":
            model = GDConvNet().to(device)

    batch_size = minibatch_size_static

    # DATA
    train_loader = get_dataset_static_gd(
        model_name=model_name,
        batch_size=batch_size,
    )

    #########################################
    # MINI-BATCH GRADIENT DESCENT TRAINING LOOP

    print("\nTraining model with gradient descent (Mini-batch)...")

    # Re-initialize the model to ensure a fresh start
    match model_name:
        case "feedforward":
            model_mini = GDFFNet().to(device)
        case "conv":
            model_mini = GDConvNet().to(device)

    criterion = nn.MSELoss()
    optimizer_mini = optim.Adam(model_mini.parameters(), lr=lr_adam_mini)

    # Determine which dimension holds the batch size
    batch_dim = 0

    mini_adam_accs = []
    mini_adam_losses = []
    mini_adam_f1s = []
    mini_adam_times = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        all_preds = []
        all_labels = []

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            batch_targets = torch.nn.functional.one_hot(
                batch_labels.long(),
                num_classes=10,
            ).float()

            outputs = model_mini(batch_images)

            loss = criterion(outputs, batch_targets)

            optimizer_mini.zero_grad()
            loss.backward()
            optimizer_mini.step()
            epoch_loss += loss.item() * batch_labels.size(0)

            # Accumulate metrics
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

        mini_adam_accs.append(epoch_accuracy)
        mini_adam_losses.append(avg_epoch_loss)
        mini_adam_f1s.append(epoch_f1)

        elapsed_time = time.time() - start_time
        mini_adam_times.append(elapsed_time)

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f}"
        )

    #########################################
    # MINI-BATCH GRADIENT DESCENT TRAINING LOOP (Mini-Batch SGD)

    print("\nTraining model with bare SGD (Mini-batch)...")

    match model_name:
        case "feedforward":
            model_mini = GDFFNet().to(device)
        case "conv":
            model_mini = GDConvNet().to(device)

    optimizer_mini = optim.SGD(model_mini.parameters(), lr=lr_gd_mini)

    mini_sgd_accs = []
    mini_sgd_losses = []
    mini_sgd_f1s = []
    mini_sgd_times = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        all_preds = []
        all_labels = []

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            batch_targets = torch.nn.functional.one_hot(
                batch_labels.long(),
                num_classes=10,
            ).float()

            outputs = model_mini(batch_images)

            loss = criterion(outputs, batch_targets)

            optimizer_mini.zero_grad()
            loss.backward()
            optimizer_mini.step()

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

        mini_sgd_accs.append(epoch_accuracy)
        mini_sgd_losses.append(avg_epoch_loss)
        mini_sgd_f1s.append(epoch_f1)
        mini_sgd_times.append(time.time() - start_time)

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f}"
        )
    #########################################
    # SAVING RESULTS AND PLOTTING
    metrics = {}
    metrics["architecture"] = model_name
    metrics["epochs"] = epochs
    metrics["seed"] = seed

    metrics["mini_adam_accuracy"] = mini_adam_accs
    metrics["mini_adam_loss"] = mini_adam_losses
    metrics["mini_adam_time"] = mini_adam_times
    metrics["mini_adam_f1"] = mini_adam_f1s

    metrics["mini_sgd_accuracy"] = mini_sgd_accs
    metrics["mini_sgd_loss"] = mini_sgd_losses
    metrics["mini_sgd_time"] = mini_sgd_times
    metrics["mini_sgd_f1"] = mini_sgd_f1s

    metrics_filename = f"paper/results/mini_batch_{model_name}/results.json"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
