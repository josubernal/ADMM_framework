"""
ADMM Metrics and Convergence

This module is responsible for calculating all loss, energy, and constraint metrics 
during ADMM training. By separating this from the main Manager, we keep the core 
optimization loop clean and make it easy to add new metrics without touching the 
training logic.

"""

import torch
import json
import os
import matplotlib.pyplot as plt

class ADMM_Metrics:
    """Observer class that computes performance and convergence metrics for an ADMM model."""
    def __init__(self, model=None):
        """Initializes the metrics tracker.

        Args:
            model (ADMM): The ADMM manager instance. Passed by reference so the tracker 
                can read its internal states ($z$, $a$, $\lambda$) without modifying them.
        """
        self.model = model
        self.metrics = {
            "loss":[],
            "accuracy": [],
            "lagrangian": [],
            "primal_residual": [],
            "preactivation_constraint_sum":[],
            "activation_constraint_sum": []
        }
    
    def __str__(self):
        if not self.metrics["loss"]:
            return "Metrics not yet initialized."
        def format_metric(val):
            if isinstance(val, list):
                return "[" + ",".join([f"{v:8.2f}" for v in val]) + "]"
            return f"{val:10.2f}"

        loss = self.metrics["loss"][-1]
        lagr = format_metric(self.metrics["lagrangian"][-1])
        lamb = format_metric(self.metrics["primal_residual"][-1])
        pre  = format_metric(self.metrics["preactivation_constraint_sum"][-1])
        act  = format_metric(self.metrics["activation_constraint_sum"][-1])
        acc  = format_metric(self.metrics["accuracy"][-1])
        return f" Loss: {loss:.4f} | Acc:{acc} | Lagr: {lagr} | Lamb: {lamb} | Pre: {pre} | Act: {act}"
    @torch.no_grad()
    def loss(self, labels: torch.Tensor):
        final_out = self.model.layers[-1].z[-1] if self.model.is_spiking else self.model.layers[-1].z
        return self.model.loss_f(final_out, labels).item()
    @torch.no_grad()
    def lagrangian(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Calculates the ADMM energy/cost to track convergence.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            float: The calculated Lagrangian cost.
        """
        cost = 0.0
        last_layer = self.model.layers[-1]
        
        final_out = last_layer.z[-1] if self.model.is_spiking else last_layer.z
        cost += self.model.loss_f(final_out, labels)
        
        a_prev_L = self.model.layers[-2].a if self.model.L > 1 else inputs
        last_out = last_layer.vectorized_forward(a_prev_L)
        last_layer_cost = last_layer.z - last_out
        
        lambda_term = last_layer_cost[-1] if self.model.is_spiking else last_layer_cost

        cost += torch.sum(self.model.lambda_lagrange * lambda_term)
        cost += (self.model.rho / 2.0) * torch.norm(last_layer_cost)**2
        
        for l in range(self.model.L - 1):
            a_prev = inputs if l == 0 else self.model.layers[l - 1].a
            layer = self.model.layers[l]
            
            physical_out = layer.vectorized_forward(a_prev)
            cost += (self.model.rho / 2.0) * torch.norm(layer.z - physical_out)**2
            cost += (self.model.beta / 2.0) * torch.norm(layer.a - layer.h(layer.z)) ** 2
            
        return cost.item()
    @torch.no_grad()
    def primal_residual_norm(self, inputs: torch.Tensor):
        """Calculates the normalized norm of the primal residual for the final layer.

        Args:
            inputs (torch.Tensor): The input data tensor.

        Returns:
            float: The normalized residual.
        """
        last_layer = self.model.layers[-1]
        a_prev_L = self.model.layers[-2].a if self.model.L > 1 else inputs
        last_out = last_layer.vectorized_forward(a_prev_L)
        residual = last_layer.z - last_out
        r = residual[-1] if self.model.is_spiking else residual
        norm_factor = r.numel() ** 0.5
        
        return (torch.norm(r) / norm_factor).item()
    @torch.no_grad()
    def preactivation_constraint_sum(self, inputs: torch.Tensor):
        """Calculates the normalized L2 norm of the pre-activation constraint (z vs Wx).

        Args:
            inputs (torch.Tensor): The input data tensor.

        Returns:
            list of float: The residual norms per layer.
        """     
        constraints_residuals = []
        
        with torch.no_grad():
            for l, layer in enumerate(self.model.layers):
                a_prev = inputs if l == 0 else self.model.layers[l - 1].a
                predicted_z = layer.vectorized_forward(a_prev)
                residual = layer.z - predicted_z
                norm_factor = residual.numel() ** 0.5
                val = (torch.norm(residual) / norm_factor).item()
                constraints_residuals.append(val)
                
        return constraints_residuals
    @torch.no_grad()
    def activation_constraint_sum(self):
        """Calculates the normalized L2 norm of the activation constraint (a vs h(z)).

        Returns:
            list of float: The residual norms per layer.
        """
        constraints_residuals = []
        
        with torch.no_grad():
            for l in range(self.model.L - 1):
                layer = self.model.layers[l]
                residual = layer.a - layer.h(layer.z)     
                norm_factor = residual.numel() ** 0.5
                val = (torch.norm(residual) / norm_factor).item()
                constraints_residuals.append(val)
                
        return constraints_residuals

    @torch.no_grad()
    def accuracy(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Calculates the true classification accuracy as a percentage (0-100) 
        by executing a full sequential forward pass through the network."""
        
        # 1. Perform the full, strict forward pass
        raw_outputs, _ = self.model.forward_model(inputs)
        
        batch_size = inputs.size(0)
        if batch_size == 0:
            return 0.0
            
        # 2. Flatten outputs and get predictions
        flat_outputs = raw_outputs.view(batch_size, -1) 
        _, predictions = flat_outputs.max(dim=1)
        
        # 3. Get target labels
        if labels.dim() > 1 and labels.size(1) > 1:
            targets = labels.argmax(dim=1)
        else:
            targets = labels.view(-1)
            
        # 4. Calculate percentage
        return 100. * (predictions == targets).sum().item() / batch_size
    
    def network_size_statistics(self):
        """Calculates the footprint of parameters and auxiliary states.

        Computes the number of parameters (W, b) and auxiliary state elements (a, z) 
        per layer, as well as the total network footprint.

        Returns:
            dict: A dictionary mapping layers and the total network to their respective counts.
        """
        stats = {}
        total_params = 0
        total_aux = 0

        for i, layer in enumerate(self.model.layers):
            W_size = layer.W.numel() if hasattr(layer, 'W') and layer.W is not None else 0
            # b_size = layer.b.numel() if hasattr(layer, 'b') and layer.b is not None else 0
            
            a_size = layer.a.numel() if hasattr(layer, 'a') and layer.a is not None else 0
            z_size = layer.z.numel() if hasattr(layer, 'z') and layer.z is not None else 0

            layer_params = W_size #+ b_size
            layer_aux = a_size + z_size

            stats[f"Layer_{i}"] = {
                "W_size": W_size,
               # "b_size": b_size,
                "a_size": a_size,
                "z_size": z_size,
                "total_layer_params": layer_params,
                "total_layer_aux_states": layer_aux,
                "total_layer_elements": layer_params + layer_aux
            }
            
            total_params += layer_params
            total_aux += layer_aux

        stats["Total_Network"] = {
            "Total_Trainable_Params (W)": total_params,
            "Total_Aux_States (a, z)": total_aux,
            "Total_Network_Elements": total_params + total_aux
        }
        
        return stats
    
    def save_metrics(self, inputs: torch.Tensor, labels: torch.Tensor):
        """A smart convergence tracker that conditionally returns applicable metrics.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            dict: A dictionary of all computed metrics.
        """
        self.metrics["loss"].append(self.loss(labels))
        self.metrics["accuracy"].append(self.accuracy(inputs, labels))
        self.metrics["lagrangian"].append(self.lagrangian(inputs, labels))
        self.metrics["primal_residual"].append(self.primal_residual_norm(inputs))
        self.metrics["preactivation_constraint_sum"].append(self.preactivation_constraint_sum(inputs))
        self.metrics[ "activation_constraint_sum"].append(self.activation_constraint_sum())
    
    def get_dic(self):
        return self.metrics
    
    def load(self, filepath: str):
        """Loads a previously saved metrics dictionary into self.metrics."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metrics file not found at: {filepath}")
        with open(filepath, 'r') as f:
            # CORRECTED: Load directly into the class attribute
            self.metrics = json.load(f)
            print(f"Metrics successfully loaded from {filepath}")
    
    def plot(self):
        """Generates plots for the requested metrics."""
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
