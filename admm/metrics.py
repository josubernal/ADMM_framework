"""
ADMM Metrics and Convergence

This module is responsible for calculating all loss, energy, and constraint metrics 
during ADMM training. By separating this from the main Manager, we keep the core 
optimization loop clean and make it easy to add new metrics without touching the 
training logic.

IMPORTANT:
- Original implementation had a bug in the Lagrangian cost calculation, which we have fixed here.
- Beta handling should be checked and ensured to be consistent across all files.

"""

import torch

class ADMM_Metrics:
    """
    Observer class that computes performance and convergence metrics for an ADMM model.
    """
    def __init__(self, model):
        """
        Args:
            model: The ADMM manager instance. We pass the model by reference so the 
                   tracker can read its internal states (z, a, lambda) without modifying them.
        """
        self.model = model

    def mse(self, labels: torch.Tensor):
        final_out = self.model.layers[-1].z[-1] if self.model._is_spiking() else self.model.layers[-1].z
        return torch.norm(final_out - labels).item() ** 2
    
    def lagrangian_cost(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Calculates the ADMM energy/cost to track convergence."""
        cost = 0.0
        batch_size = self.model._get_batchsize(inputs)
        last_layer = self.model.layers[-1]
        
        z_last_flat = last_layer.z[-1].view(batch_size, -1) if self.model._is_spiking() else last_layer.z.view(batch_size, -1)
        cost += torch.norm(z_last_flat - labels)**2 
        
        a_prev_L = self.model.layers[-2].a if self.model.L > 1 else inputs
        last_out = last_layer.vectorized_forward(a_prev_L)
        last_layer_cost = last_layer.z - last_out
        
        lambda_term = last_layer_cost[-1] if self.model._is_spiking() else last_layer_cost

        cost += torch.sum(self.model.lambda_lagrange * lambda_term)
        cost += (self.model.beta / 2.0) * torch.norm(last_layer_cost)**2
        
        for l in range(self.model.L - 1):
            a_prev = inputs if l == 0 else self.model.layers[l - 1].a
            layer = self.model.layers[l]
            
            physical_out = layer.vectorized_forward(a_prev)
            cost += (self.model.beta / 2.0) * torch.norm(layer.z - physical_out)**2
            cost += (self.model.gamma / 2.0) * torch.norm(layer.a - layer.h(layer.z)) ** 2
            
        return cost.item()
    
        
    def lagrangian_cost_original(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        Calculates the ADMM energy/cost exactly tracking admm_snn.py.
        Strictly for legacy SNN comparison.
        """
        if not self.model._is_spiking():
            return None 
            
        cost = 0.0

        for l in range(self.model.L - 1):
            output_spikes = inputs if l == 0 else self.model.layers[l - 1].a
            layer = self.model.layers[l]
            
            for t in range(1, self.model.T):
                term1 = layer.W @ output_spikes[t].t() - layer.z[t].t() + layer.deltas * layer.z[t - 1].t() #BUG
                #term1 = layer.W @ output_spikes[t].t() - layer.z[t].t() + layer.deltas * layer.z[t - 1].t() - layer.thetas * layer.a[t - 1].t() #BUG
                cost += (self.model.beta / 2.0) * torch.norm(term1) ** 2
                
                term2 = layer.a[t] - layer.h(layer.z[t])
                cost += (self.model.beta / 2.0) * torch.norm(term2) ** 2 #BUG
                #cost += (self.model.gamma / 2.0) * torch.norm(term2) ** 2 #BUG

            term1_t0 = layer.W @ output_spikes[0].t() - layer.z[0].t()
            cost += (self.model.beta / 2.0) * torch.norm(term1_t0) ** 2
            
            term2_t0 = layer.a[0] - layer.h(layer.z[0])
            cost += (self.model.beta / 2.0) * torch.norm(term2_t0) ** 2 #BUG
            #cost += (self.model.gamma/ 2.0) * torch.norm(term2_t0) ** 2 #BUG
            

        last_layer = self.model.layers[-1]
        a_prev_L = self.model.layers[-2].a if self.model.L > 1 else inputs
        
        for t in range(1, self.model.T):
            term1_L = last_layer.W @ a_prev_L[t].t() - last_layer.z[t].t() + last_layer.deltas * last_layer.z[t - 1].t() #BUG
            #term1_L = last_layer.W @ a_prev_L[t].t() - last_layer.z[t].t() + last_layer.deltas * last_layer.z[t - 1].t() - last_layer.thetas * last_layer.a[t - 1].t() #BUG
            cost += (self.model.beta / 2.0) * torch.norm(term1_L) ** 2

        #term1_L_t0 = last_layer.W @ a_prev_L[0].t() - last_layer.z[0].t() #BUG
        #cost += (self.model.beta / 2.0) * torch.norm(term1_L_t0) ** 2 #BUG
        
        for i in range(self.model.lambda_lagrange.size(0)):
            lam_term = last_layer.z[-1][i] - last_layer.deltas * last_layer.z[-2][i] - last_layer.W @ a_prev_L[-1][i]
            cost += self.model.lambda_lagrange[i] @ lam_term

        cost += torch.norm(last_layer.z[-1].t() - labels.t()) ** 2

        return cost.item() if isinstance(cost, torch.Tensor) else cost
    
    
    def primal_residual_norm(self, inputs: torch.Tensor):
        """Calculates the normalized L1 sum of the primal residual for the final layer."""
        last_layer = self.model.layers[-1]
        a_prev_L = self.model.layers[-2].a if self.model.L > 1 else inputs
        last_out = last_layer.vectorized_forward(a_prev_L)
        residual = last_layer.z - last_out
        r = residual[-1] if self.model._is_spiking() else residual
        norm_factor = r.numel() ** 0.5
        
        return (torch.norm(r) / norm_factor).item()

    def preactivation_constraint_sum(self, inputs: torch.Tensor):
        """Calculates the normalized L2 norm of the pre-activation constraint (z vs Wx)."""
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
  
    def activation_constraint_sum(self):
        """Calculates the normalized L2 norm of the activation constraint (a vs h(z))."""
        constraints_residuals = []
        
        with torch.no_grad():
            for l in range(self.model.L - 1):
                layer = self.model.layers[l]
                residual = layer.a - layer.h(layer.z)     
                norm_factor = residual.numel() ** 0.5
                val = (torch.norm(residual) / norm_factor).item()
                constraints_residuals.append(val)
                
        return constraints_residuals


    def network_size_statistics(self):
        """
        Calculates the number of parameters (W, b) and auxiliary state elements (a, z) 
        per layer, as well as the total network footprint.
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
    
    def get_all_metrics(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        A smart convergence tracker that conditionally returns applicable metrics.
        """
        metrics = {
            "mse": self.mse(labels),
            "lagrangian_cost": self.lagrangian_cost(inputs, labels),
            "primal_residual": self.primal_residual_norm(inputs),
            "preactivation_constraint_sum": self.preactivation_constraint_sum(inputs),
            "activation_constraint_sum": self.activation_constraint_sum()
        }
        
        is_linear_network = all(layer.W.dim() == 2 for layer in self.model.layers)
        if self.model._is_spiking() and is_linear_network:
            metrics["lagrangian_cost_original"] = self.lagrangian_cost_original(inputs, labels)
            
        return metrics