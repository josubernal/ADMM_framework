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
        self, inputs: torch.Tensor, labels: torch.Tensor, batch_state: ADMM_BatchState
    ) -> float:
        r"""Calculates the ADMM augmented Lagrangian energy to track convergence.

        This incorporates the objective [loss][src.admm.loss_functions.ADMM_Loss], the spatial affine penalties,
        the activation penalties, and the dual variable (Lagrange) multipliers.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The ground truth labels.
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.

        Returns:
            float: The calculated Lagrangian cost.
        """
        cost = 0.0

        final_layer_state = batch_state.layer_states[-1]
        final_out = (
            final_layer_state.z[-1] if self.model.is_spiking else final_layer_state.z
        )

        loss_val = self.model.loss_f(final_out, labels)
        cost += loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val

        for layer_idx, layer in enumerate(self.model.layers):
            layer_state = batch_state.layer_states[layer_idx]
            a_prev = (
                inputs if layer_idx == 0 else batch_state.layer_states[layer_idx - 1].a
            )

            predicted_z = self.vectorized_forward(
                layer=layer, state=layer_state, a_prev=a_prev
            )

            residual = layer_state.z - predicted_z
            cost += (layer.config.rho / 2.0) * torch.norm(residual).item() ** 2

            if layer_idx < self.model.L - 1:
                cost += (layer.config.beta / 2.0) * torch.norm(
                    layer_state.a - layer.h(layer_state.z)
                ).item() ** 2

            if (
                getattr(layer.config, "use_lagrange", False)
                and layer_state.lambda_lagrange is not None
            ):
                if self.model.is_spiking:
                    cost += torch.sum(layer_state.lambda_lagrange * residual[-1]).item()
                else:
                    cost += torch.sum(layer_state.lambda_lagrange * residual).item()

        return cost

    @torch.no_grad()
    def primal_residual_norm(
        self, inputs: torch.Tensor, batch_state: ADMM_BatchState
    ) -> float:
        r"""Calculates the normalized norm of the primal residual for the final layer.

        Formula evaluated: $||z_L -F_L(a_{L-1})|| / \sqrt{N}$

        Args:
            inputs (torch.Tensor): The input data tensor.
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.

        Returns:
            float: The normalized residual.
        """
        last_layer = self.model.layers[-1]
        last_state = batch_state.layer_states[-1]
        a_prev_L = batch_state.layer_states[-2].a if self.model.L > 1 else inputs

        last_out = self.vectorized_forward(
            layer=last_layer, state=last_state, a_prev=a_prev_L
        )

        residual = last_state.z - last_out
        r = residual[-1] if self.model.is_spiking else residual
        norm_factor = r.numel() ** 0.5

        return (torch.norm(r) / norm_factor).item()

    @torch.no_grad()
    def preactivation_constraint_sum(
        self, inputs: torch.Tensor, batch_state: ADMM_BatchState
    ) -> list[float]:
        r"""Calculates the normalized L2 norm of the pre-activation constraints.

        Formula evaluated per layer: $||z_l - F_l(a_{layer_idx-1})|| / \sqrt{N}$

        Args:
            inputs (torch.Tensor): The input data tensor.
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.

        Returns:
            list of float: The residual norms per layer.
        """
        constraints_residuals = []

        for layer_idx, layer in enumerate(self.model.layers):
            layer_state = batch_state.layer_states[layer_idx]
            a_prev = (
                inputs if layer_idx == 0 else batch_state.layer_states[layer_idx - 1].a
            )

            predicted_z = self.vectorized_forward(
                layer=layer, state=layer_state, a_prev=a_prev
            )

            residual = layer_state.z - predicted_z
            norm_factor = residual.numel() ** 0.5
            val = (torch.norm(residual) / norm_factor).item()
            constraints_residuals.append(val)

        return constraints_residuals

    @torch.no_grad()
    def activation_constraint_sum(self, batch_state: ADMM_BatchState) -> list[float]:
        r"""Calculates the normalized L2 norm of the activation constraints.

        Formula evaluated per layer: $||a_l - h_l(z_l)|| / \sqrt{N}$

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
    ) -> tuple[float, list[float] | float]:
        """Calculates accuracy and extracts firing rates in a single forward pass.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            tuple:
                - float: The calculated accuracy percentage.
                - list of float or float: Firing rates per layer (if spiking), else NaN.
        """
        raw_outputs, firing_rates = self.model.forward_model(inputs, True)
        if not self.model.is_spiking:
            firing_rates = float("nan")

        batch_size = labels.size(0)

        if batch_size == 0:
            return 0.0, firing_rates

        flat_outputs = raw_outputs.view(batch_size, -1)
        _, predictions = flat_outputs.max(dim=1)

        if labels.dim() > 1 and labels.size(1) > 1:
            targets = labels.argmax(dim=1)
        else:
            targets = labels.view(-1)

        acc = 100.0 * (predictions == targets).sum().item() / batch_size

        return acc, firing_rates

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

        if (
            getattr(self.model, "initialized", False)
            and self.model.state_handler.num_batches > 0
        ):
            _, _, batch_state = self.model.state_handler.load_batch(0)

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

    def save_metrics(
        self,
    ) -> None:
        """Computes and appends all tracked metrics for the current epoch.

        If inputs, labels, and batch_state are not provided, it automatically iterates
        over all batches stored in the state handler and records the epoch-averaged metrics.
        """
        if (
            not hasattr(self.model, "state_handler")
            or self.model.state_handler.num_batches == 0
        ):
            print("Warning: No batches found in state_handler. Metrics not saved.")
            return

        num_batches = self.model.state_handler.num_batches
        batch_ids = self.model.state_handler.get_batch_ids()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_lagr = 0.0
        epoch_primal = 0.0

        epoch_pre = [0.0] * self.model.L
        epoch_act = [0.0] * (self.model.L - 1)
        epoch_fr = [0.0] * (self.model.L - 1) if self.model.is_spiking else None

        # Accumulate metrics across all batches
        for batch_id in batch_ids:
            b_inputs, b_labels, b_state = self.model.state_handler.load_batch(batch_id)

            epoch_loss += self.loss(b_labels, b_state)

            acc, fr = self.evaluate_performance(b_inputs, b_labels)
            epoch_acc += acc
            if self.model.is_spiking and isinstance(fr, list):
                for i in range(len(fr)):
                    epoch_fr[i] += fr[i]

            epoch_lagr += self.lagrangian(b_inputs, b_labels, b_state)
            epoch_primal += self.primal_residual_norm(b_inputs, b_state)

            pre = self.preactivation_constraint_sum(b_inputs, b_state)
            for i in range(len(pre)):
                epoch_pre[i] += pre[i]

            act = self.activation_constraint_sum(b_state)
            for i in range(len(act)):
                epoch_act[i] += act[i]

        # Average and append
        self.metrics["loss"].append(epoch_loss / num_batches)
        self.metrics["accuracy"].append(epoch_acc / num_batches)
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
