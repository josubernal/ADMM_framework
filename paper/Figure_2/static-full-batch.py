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

batch_size_static = config.getint("config", "batch_size_static")
epochs = config.getint("config", "epochs")
hidden_size_static = config.getint("config", "hidden_size_static")
hidden_channels_static = config.getint("config", "hidden_channels_static")
k = config.getint("config", "k")
p = config.getint("config", "p")
s = config.getint("config", "s")
lr_gd = config.getfloat("config", "learning_rate_gd")
lr_adam = config.getfloat("config", "learning_rate_adam")


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
    # MODEL INSTANTIATION
    match model_name:
        case "feedforward":
            model = GDFFNet().to(device)
        case "conv":
            model = GDConvNet().to(device)

    #########################################
    # ADAM GRADIENT DESCENT TRAINING LOOP
    criterion = nn.CrossEntropyLoss()
    # DATA
    images, labels = get_dataset_static_gd(
        model_name=model_name, batch_size=batch_size_static, device=device
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

    metrics_filename = (
        f"paper/results/admm_vs_gd/static-full-batch/{model_name}/results.json"
    )
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    # Free up memory before the next model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
