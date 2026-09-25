"""
This module is responsible for calculating all loss, energy, and constraint metrics
during ADMM training. By separating this from the main Manager, we keep the core
optimization loop clean and make it easy to add new metrics without touching the
training logic. Additionally, this module allows for quick experimentation, making it easy
to set a training loop.
"""

import json
import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn

from .dataclasses import ADMM_BatchState, ADMM_LayerState
from .functional.utils import compute_temporal_dependencies


class ADMM_Metrics:
    """Observer class that computes performance and convergence metrics for an ADMM model.

    Tracks loss, accuracy, firing rates, and the ADMM-specific constraint
    residuals (primal and dual energy) over the course of training.

    Attributes:
        model (nn.Module): The ADMM manager instance to observe.
        metrics (dict): A dictionary storing the historical arrays for each tracked metric.
    """

    def __init__(self, model=None):
        r"""Initializes the metrics tracker.

        Args:
            model (ADMM, optional): The ADMM manager instance. Passed by reference so the tracker
                can read its internal states ($z$, $a$, $\lambda$) without modifying them. Defaults to None.
        """
        self.model = model
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "f1": [],
            "lagrangian": [],
            "primal_residual": [],
            "preactivation_constraint_sum": [],
            "activation_constraint_sum": [],
            "firing_rate": [],
        }

    def __str__(self) -> str:
        """Returns a formatted string of the most recent metrics."""
        if not self.metrics["loss"]:
            return "Metrics not yet initialized."

        def format_metric(val):
            if isinstance(val, list):
                return (
                    "["
                    + ",".join(
                        [f"{v:8.2f}" for v in val if isinstance(v, (int, float))]
                    )
                    + "]"
                )
            return f"{val:10.2f}"

        loss = self.metrics["loss"][-1]
        lagr = format_metric(self.metrics["lagrangian"][-1])
        lamb = format_metric(self.metrics["primal_residual"][-1])
        pre = format_metric(self.metrics["preactivation_constraint_sum"][-1])
        act = format_metric(self.metrics["activation_constraint_sum"][-1])
        acc = format_metric(self.metrics["accuracy"][-1])

        # Safely format firing rate
        raw_fr = self.metrics["firing_rate"][-1]
        fr_str = format_metric(raw_fr) if raw_fr is not None else "N/A"

        return f"Loss: {loss:.4f} | Acc:{acc} | FR: {fr_str} | Lagr: {lagr} | Lamb: {lamb} | Pre: {pre} | Act: {act}"

    def vectorized_forward(
        self, layer: nn.Module, a_prev: torch.Tensor, state: ADMM_LayerState
    ) -> torch.Tensor:
        """Computes the forward pass for a layer, including temporal dynamics if applicable.

        Calculates the spatial forward pass and adds temporal dependencies
        if the underlying model is configured as a spiking neural network.

        Args:
            layer (nn.Module): The current network layer being evaluated.
            a_prev (torch.Tensor): The activations from the preceding layer (or input data).
            state (ADMM_LayerState): The ADMM state object containing auxiliary variables for the current layer.

        Returns:
            torch.Tensor: The computed pre-activations, including temporal dynamics if spiking.
        """
        forward = layer.spatial_forward(a_prev)

        temporal_forward = (
            forward + (compute_temporal_dependencies(state, layer.config))
            if self.model.is_spiking
            else forward
        )
        return temporal_forward

    @torch.no_grad()
    def _compute_constraints(
        self,
        inputs,
        batch_state,
    ):
        predicted_zs = []
        residuals = []
        activation_residuals = []

        for layer_idx, layer in enumerate(self.model.layers):
            layer_state = batch_state.layer_states[layer_idx]

            a_prev = (
                inputs if layer_idx == 0 else batch_state.layer_states[layer_idx - 1].a
            )

            predicted_z = self.vectorized_forward(
                layer=layer,
                state=layer_state,
                a_prev=a_prev,
            )

            residual = layer_state.z - predicted_z

            predicted_zs.append(predicted_z)
            residuals.append(residual)

            if layer_idx < self.model.L - 1:
                activation_residuals.append(layer_state.a - layer.h(layer_state.z))

        return predicted_zs, residuals, activation_residuals

    @torch.no_grad()
    def loss(self, labels: torch.Tensor, batch_state: ADMM_BatchState) -> float:
        """Computes the scalar [loss][src.admm.loss_functions.ADMM_Loss] of the final output.

        Args:
            labels (torch.Tensor): The ground truth labels.
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.

        Returns:
            float: The computed loss value.
        """
        final_layer_state = batch_state.layer_states[-1]
        final_out = (
            final_layer_state.z[-1] if self.model.is_spiking else final_layer_state.z
        )
        return self.model.loss_f(final_out, labels).item()

    @torch.no_grad()
    def lagrangian(
        self,
        labels,
        batch_state,
        residuals,
        activation_residuals,
    ):
        final_layer_state = batch_state.layer_states[-1]

        final_out = (
            final_layer_state.z[-1] if self.model.is_spiking else final_layer_state.z
        )

        cost = self.model.loss_f(final_out, labels)

        for layer_idx, layer in enumerate(self.model.layers):
            residual = residuals[layer_idx]

            # ||z - F(a)||^2
            cost = cost + (layer.config.rho / 2.0) * residual.square().sum()

            # ||a - h(z)||^2
            if layer_idx < self.model.L - 1:
                activation_residual = activation_residuals[layer_idx]
                cost = (
                    cost
                    + (layer.config.beta / 2.0) * activation_residual.square().sum()
                )

            # Lagrange multiplier term
            if (
                getattr(layer.config, "use_lagrange", False)
                and batch_state.layer_states[layer_idx].lambda_lagrange is not None
            ):
                lambda_lagrange = batch_state.layer_states[layer_idx].lambda_lagrange

                if self.model.is_spiking:
                    cost = cost + (lambda_lagrange * residual[-1]).sum()
                else:
                    cost = cost + (lambda_lagrange * residual).sum()

        return cost.item()

    @torch.no_grad()
    def primal_residual_norm(
        self,
        residuals,
    ) -> float:
        """Normalized primal residual of the final layer."""

        residual = residuals[-1]

        if self.model.is_spiking:
            residual = residual[-1]

        return torch.sqrt(residual.square().mean()).item()

    @torch.no_grad()
    def preactivation_constraint_sum(
        self,
        residuals,
    ) -> list[float]:
        """Normalized preactivation constraint for every layer."""

        return [torch.sqrt(residual.square().mean()).item() for residual in residuals]

    @torch.no_grad()
    def activation_constraint_sum(self, batch_state: ADMM_BatchState) -> list[float]:
        r"""Calculates the normalized L2 norm of the activation constraints.

        Formula evaluated per layer:

        $$ ||a_l - h_l(z_l)|| / \sqrt{N} $$

        Args:
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.

        Returns:
            list of float: The residual norms per layer.
        """
        constraints_residuals = []

        for layer_idx in range(self.model.L - 1):
            layer = self.model.layers[layer_idx]
            layer_state = batch_state.layer_states[layer_idx]

            residual = layer_state.a - layer.h(layer_state.z)
            norm_factor = residual.numel() ** 0.5
            val = (torch.norm(residual) / norm_factor).item()
            constraints_residuals.append(val)

        return constraints_residuals

    @torch.no_grad()
    def evaluate_performance(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[float, float, list[float] | float]:
        """Calculates accuracy and extracts firing rates in a single forward pass.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            tuple:
                - float: The calculated accuracy percentage.
                - float: The calculated F1 score.
                - list of float or float: Firing rates per layer (if spiking), else NaN.
        """
        raw_outputs, firing_rates = self.model.forward_model(inputs, True)
        if not self.model.is_spiking:
            firing_rates = float("nan")

        batch_size = labels.size(0)

        if batch_size == 0:
            return 0.0, 0.0, firing_rates

        flat_outputs = raw_outputs.view(batch_size, -1)
        _, predictions = flat_outputs.max(dim=1)

        if labels.dim() > 1 and labels.size(1) > 1:
            targets = labels.argmax(dim=1)
        else:
            targets = labels.view(-1)

        acc = 100.0 * (predictions == targets).sum().item() / batch_size

        # F1
        classes = torch.unique(torch.cat((targets, predictions)))
        f1_sum = 0.0

        for c in classes:
            # Calculate True Positives, False Positives, and False Negatives for class 'c'
            tp = ((predictions == c) & (targets == c)).sum().float()
            fp = ((predictions == c) & (targets != c)).sum().float()
            fn = ((predictions != c) & (targets == c)).sum().float()

            # Calculate F1 for this class (1e-8 is an epsilon to prevent division by zero)
            f1_sum += (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # Macro F1 average across all found classes
        f1 = (f1_sum / len(classes)).item() if len(classes) > 0 else 0.0

        return acc, f1, firing_rates

    def network_size_statistics(self) -> dict:
        """Calculates the footprint of parameters and auxiliary states.

        Computes the number of trainable parameters ($W$, $b$) and auxiliary
        state elements ($a$, $z$) per layer, as well as the total network footprint.

        Returns:
            dict: A dictionary mapping layers and the total network to their respective counts.
        """
        stats = {}
        total_params = 0
        total_aux = 0
        batch_state = None

        if getattr(
            self.model, "initialized", False
        ) and self.model.state_handler.is_batch_initialized(0):
            batch_state = self.model.state_handler.load_batch(0)

        for i, layer in enumerate(self.model.layers):
            W_size = (
                layer.W.numel() if hasattr(layer, "W") and layer.W is not None else 0
            )
            b_size = (
                layer.b.numel() if hasattr(layer, "b") and layer.b is not None else 0
            )

            if batch_state is not None:
                layer_state = batch_state.layer_states[i]
                a_size = layer_state.a.numel() if layer_state.a is not None else 0
                z_size = layer_state.z.numel() if layer_state.z is not None else 0
            else:
                a_size, z_size = 0, 0

            layer_params = W_size
            layer_aux = a_size + z_size

            stats[f"Layer_{i}"] = {
                "W_size": W_size,
                "b_size": b_size,
                "a_size": a_size,
                "z_size": z_size,
                "total_layer_params": layer_params,
                "total_layer_aux_states": layer_aux,
                "total_layer_elements": layer_params + layer_aux,
            }

            total_params += layer_params
            total_aux += layer_aux

        stats["Total_Network"] = {
            "Total_Trainable_Params (W)": total_params,
            "Total_Aux_States (a, z)": total_aux,
            "Total_Network_Elements": total_params + total_aux,
        }

        return stats

    @torch.no_grad()
    def save_metrics(
        self,
        dataloader,
    ) -> None:
        """Compute and append epoch-averaged metrics.

        The DataLoader provides inputs and labels, while the ADMM state handler
        provides the persistent auxiliary state for each batch.
        """

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        epoch_lagr = 0.0
        epoch_primal = 0.0

        epoch_pre = [0.0] * self.model.L
        epoch_act = [0.0] * (self.model.L - 1)
        epoch_fr = [0.0] * (self.model.L - 1) if self.model.is_spiking else None

        num_batches = 0

        for (
            batch_id,
            b_inputs,
            b_labels,
            b_state,
        ) in self.model._iterate_batches(dataloader):
            num_batches += 1

            # ---------------------------------------------------------
            # 1. Loss
            # ---------------------------------------------------------
            epoch_loss += self.loss(
                b_labels,
                b_state,
            )

            # ---------------------------------------------------------
            # 2. Performance
            # ---------------------------------------------------------
            acc, f1, fr = self.evaluate_performance(
                b_inputs,
                b_labels,
            )

            epoch_acc += acc
            epoch_f1 += f1

            if self.model.is_spiking and isinstance(fr, list):
                for i, value in enumerate(fr):
                    epoch_fr[i] += value

            # ---------------------------------------------------------
            # 3. Compute all ADMM constraints ONCE
            # ---------------------------------------------------------
            _, residuals, activation_residuals = self._compute_constraints(
                b_inputs,
                b_state,
            )

            # ---------------------------------------------------------
            # 4. Lagrangian
            # ---------------------------------------------------------
            epoch_lagr += self.lagrangian(
                b_labels,
                b_state,
                residuals,
                activation_residuals,
            )

            # ---------------------------------------------------------
            # 5. Primal residual
            # ---------------------------------------------------------
            epoch_primal += self.primal_residual_norm(
                residuals,
            )

            # ---------------------------------------------------------
            # 6. Preactivation constraints
            # ---------------------------------------------------------
            pre = self.preactivation_constraint_sum(
                residuals,
            )

            for i, value in enumerate(pre):
                epoch_pre[i] += value

            # ---------------------------------------------------------
            # 7. Activation constraints
            # ---------------------------------------------------------
            for i, residual in enumerate(activation_residuals):
                epoch_act[i] += torch.sqrt(residual.square().mean()).item()

        if num_batches == 0:
            print("Warning: DataLoader contains no batches. Metrics not saved.")
            return

        # -------------------------------------------------------------
        # Epoch averages
        # -------------------------------------------------------------
        self.metrics["loss"].append(epoch_loss / num_batches)

        self.metrics["accuracy"].append(epoch_acc / num_batches)

        self.metrics["f1"].append(epoch_f1 / num_batches)

        self.metrics["lagrangian"].append(epoch_lagr / num_batches)

        self.metrics["primal_residual"].append(epoch_primal / num_batches)

        self.metrics["preactivation_constraint_sum"].append(
            [x / num_batches for x in epoch_pre]
        )

        self.metrics["activation_constraint_sum"].append(
            [x / num_batches for x in epoch_act]
        )

        if self.model.is_spiking:
            self.metrics["firing_rate"].append([x / num_batches for x in epoch_fr])
        else:
            self.metrics["firing_rate"].append(float("nan"))

    @torch.no_grad()
    def save_distributed_metrics(
        self,
        dataloader,
    ) -> None:
        """Compute and append distributed epoch-averaged metrics."""

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        epoch_lagr = 0.0
        epoch_primal = 0.0

        epoch_pre = [0.0] * self.model.L
        epoch_act = [0.0] * (self.model.L - 1)
        epoch_fr = [0.0] * (self.model.L - 1) if self.model.is_spiking else None

        num_batches = 0

        for (
            batch_id,
            b_inputs,
            b_labels,
            b_state,
        ) in self.model._iterate_batches(dataloader):
            num_batches += 1

            epoch_loss += self.loss(
                b_labels,
                b_state,
            )

            acc, f1, fr = self.evaluate_performance(
                b_inputs,
                b_labels,
            )

            epoch_acc += acc
            epoch_f1 += f1

            if self.model.is_spiking and isinstance(fr, list):
                for i, value in enumerate(fr):
                    epoch_fr[i] += value

            epoch_lagr += self.lagrangian(
                b_inputs,
                b_labels,
                b_state,
            )

            epoch_primal += self.primal_residual_norm(
                b_inputs,
                b_state,
            )

            pre = self.preactivation_constraint_sum(
                b_inputs,
                b_state,
            )

            for i, value in enumerate(pre):
                epoch_pre[i] += value

            act = self.activation_constraint_sum(
                b_state,
            )

            for i, value in enumerate(act):
                epoch_act[i] += value

        if num_batches == 0:
            print("Warning: DataLoader contains no batches. Metrics not saved.")
            return

        # Local averages
        local_metrics = torch.tensor(
            [
                epoch_loss / num_batches,
                epoch_acc / num_batches,
                epoch_f1 / num_batches,
                epoch_lagr / num_batches,
                epoch_primal / num_batches,
                *[x / num_batches for x in epoch_pre],
                *[x / num_batches for x in epoch_act],
                *([x / num_batches for x in epoch_fr] if self.model.is_spiking else []),
            ],
            dtype=torch.float32,
            device=self.model.device,
        )

        local_num_batches = torch.tensor(
            [num_batches],
            dtype=torch.float32,
            device=self.model.device,
        )

        if dist.is_initialized():
            dist.all_reduce(
                local_metrics,
                op=dist.ReduceOp.SUM,
            )

            dist.all_reduce(
                local_num_batches,
                op=dist.ReduceOp.SUM,
            )

            local_metrics /= local_num_batches

        # Unpack
        idx = 0

        global_loss = local_metrics[idx].item()
        idx += 1

        global_acc = local_metrics[idx].item()
        idx += 1

        global_f1 = local_metrics[idx].item()
        idx += 1

        global_lagr = local_metrics[idx].item()
        idx += 1

        global_primal = local_metrics[idx].item()
        idx += 1

        global_pre = []
        for _ in range(self.model.L):
            global_pre.append(local_metrics[idx].item())
            idx += 1

        global_act = []
        for _ in range(self.model.L - 1):
            global_act.append(local_metrics[idx].item())
            idx += 1

        if self.model.is_spiking:
            global_fr = []
            for _ in range(self.model.L - 1):
                global_fr.append(local_metrics[idx].item())
                idx += 1
        else:
            global_fr = float("nan")

        self.metrics["loss"].append(global_loss)
        self.metrics["accuracy"].append(global_acc)
        self.metrics["f1"].append(global_f1)
        self.metrics["lagrangian"].append(global_lagr)
        self.metrics["primal_residual"].append(global_primal)
        self.metrics["preactivation_constraint_sum"].append(global_pre)
        self.metrics["activation_constraint_sum"].append(global_act)
        self.metrics["firing_rate"].append(global_fr)

    def get_dic(self) -> dict:
        """Returns the internal metrics dictionary."""
        return self.metrics

    def load(self, filepath: str) -> None:
        """Loads a previously saved metrics JSON file into the tracker.

        Args:
            filepath (str): The path to the saved metrics JSON.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metrics file not found at: {filepath}")
        with open(filepath, "r") as f:
            # CORRECTED: Load directly into the class attribute
            self.metrics = json.load(f)
            print(f"Metrics successfully loaded from {filepath}")

    def plot(self) -> None:
        """Generates matplotlib subplots for all recorded metrics."""
        fig, ax = plt.subplots(2, 3, figsize=(30, 5))
        ax[0, 0].semilogy(self.metrics["lagrangian"])
        ax[0, 0].set_title("Lagrangian")
        ax[0, 1].semilogy(self.metrics["primal_residual"])
        ax[0, 1].set_title("Primal Residual Norm")
        ax[0, 2].semilogy(self.metrics["preactivation_constraint_sum"])
        ax[0, 2].set_title("Preactivation Constraint (||z - F(a)||)")
        ax[1, 0].semilogy(self.metrics["activation_constraint_sum"])
        ax[1, 0].set_title("Activation Constraint (||a - h(z)||)")
        ax[1, 1].semilogy(self.metrics["loss"])
        ax[1, 1].set_title("Loss")
        ax[1, 2].plot(self.metrics["accuracy"])
        ax[1, 2].set_title("Train Accuracy")

        plt.tight_layout()
        plt.show()
