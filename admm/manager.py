"""
ADMM Universal Manager

This module contains the core orchestrator for the network and ADMM optimizer.

ADMMI breaks the network down into layer-wise sub-problems and updates in a loop:
1. Weight Updates 
2. Activation & Pre-activation Updates 
3. Dual Variable / Lagrange Multiplier Updates (Gradient Ascent)

INDEX:
- ADMM (Main Module)

IMPORTANT:
- Lambda update is beta not 2*beta, to match Cesare's implementation and ensure convergence.
This should be discussed in the future as it differs from the standard ADMM formulation.
"""

import torch
import torch.nn as nn
import random
import warnings

class ADMM(nn.Module):
    """
    Universal Manager.
    Handles both Static and Spiking ADMM networks automatically.
    """      
    def __init__(self, layers: nn.ModuleList, beta: float = 1.0, gamma: float = 1.0, 
                 init: str = "zeros", bias: bool = False, device=None, 
                 train_method: str = "unrolled", **kwargs):
        super().__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.L = len(self.layers)
        self.initialized = False
        
        self.lambda_lagrange = None
        self.init = init
        self.train_method = train_method
        self.beta = beta
        self.gamma = gamma
        self.bias = bias

        if self.bias:
            raise NotImplementedError("Bias usage is not available for now. It will be added in future versions.")
        
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        valid_methods = ["vectorized", "unrolled-random", "unrolled-sequential"]
        if self.train_method not in valid_methods:
            raise ValueError(f"Invalid train_method. Please select from: {valid_methods}")
        
        if not self._is_spiking() and self.train_method.startswith("unrolled"):
            warnings.warn(
                "Non-spiking networks can only be trained using the 'vectorized' method. "
                "Automatically switching train_method to 'vectorized'.", 
                UserWarning
            )
            self.train_method = "vectorized"
        
        if self._is_spiking() and self.train_method == "vectorized":
            raise NotImplementedError("Spiking networks currently require 'unrolled' training methods. Vectorized SNNs coming soon.")
    

        config = {'beta': self.beta, 'gamma': self.gamma, 'init': self.init, 'train_method': self.train_method}
        config.update(kwargs)
        self._configure_layers(config)

    def _configure_layers(self, config_dict: dict):
        """Pushes global hyperparameters down to the individual layer setup methods."""
        for i, layer in enumerate(self.layers):
            layer.to(self.device)
            is_last = (i == self.L - 1)
            layer.setup(config_dict, is_last_layer=is_last)
    
    def _is_spiking(self):
        """Helper to determine if the network has a temporal dimension (SNN)."""
        return hasattr(self, 'T') and self.T is not None           
    
    def _get_batchsize(self, inputs: torch.Tensor):
        """Extracts batch size dynamically (dim 1 for Spiking, dim 0 for Static)."""
        return inputs.shape[1] if self._is_spiking() else inputs.shape[0]                     
    
    def _init_states(self, inputs: torch.Tensor):
        """
        Warm-starts the ADMM auxiliary variables 'z' (pre-activations) and 'a' (activations) 
        using a standard forward pass. Initializes the Lagrange multiplier to zero.
        """
        x = inputs.to(self.device)
        self.initialized = True        
        
        with torch.no_grad():
            if self.init == "zeros-rng":
                warnings.warn(
                    "The 'zeros-rng' initialization is deprecated. Use 'zeros' for production.",
                    UserWarning
                )
                z_preds, a_preds = [], []
                for layer in self.layers:
                    z_pred = layer.forward(x)
                    a_pred = layer.h(z_pred)
                    z_preds.append(z_pred)
                    a_preds.append(a_pred)
                    x = a_pred 
                for i, layer in enumerate(self.layers):
                    layer.z = torch.rand(z_preds[i].shape).to(self.device)
                        
                for i, layer in enumerate(self.layers):
                    if i < self.L - 1: 
                        layer.a = torch.rand(a_preds[i].shape).to(self.device)
                    else:
                        layer.a = torch.zeros_like(a_preds[i])

            else:
                for layer in self.layers:
                    z_pred = layer.forward(x)
                    a_pred = layer.h(z_pred)     
                    layer.z = torch.rand_like(z_pred)
                    layer.a = torch.rand_like(a_pred)
                    x = a_pred 
                        
            last_z = self.layers[-1].z
            target_shape = last_z[-1] if self._is_spiking() else last_z
            self.lambda_lagrange = torch.zeros_like(target_shape, device=self.device)
    
    def forward_model(self, inputs: torch.Tensor):
        """
        Standard Feed-Forward pass used strictly for inference/evaluation.
        Returns the final network output and the activation firing rates per layer.
        """
        x = inputs.to(self.device)
        final_z = None
        firing_rates = [] 
        
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                z_pred = layer.forward(x)
                x = layer.h(z_pred) 
                
                if i < len(self.layers) - 1:
                    # Sum of all spikes divided by the total number of elements
                    layer_firing_rate = x.sum().item() / x.numel()
                    firing_rates.append(layer_firing_rate)
                    
                final_z = z_pred
                
        final_out = final_z[-1] if self._is_spiking() else final_z
        batch_size = inputs.size(1) if self._is_spiking() else inputs.size(0)
            
        if final_out.dim() > 2:
            final_out = final_out.view(batch_size, -1)

        return final_out, firing_rates
    
    def _lambda_update(self, last_layer, a_prev_L):
        """
        Updates the Lagrange multiplier (lambda) based on the final layer constraint.
        Formula: lambda_new = lambda_old + beta * (z - Wx) 
        """
        if self._is_spiking():
            z_T = last_layer.z[-1]
            z_T_minus_1 = last_layer.z[-2]
            F_a_T = last_layer.spatial_forward(a_prev_L[-1].unsqueeze(0)).squeeze(0)
            residual = z_T - (last_layer.deltas * z_T_minus_1) - F_a_T
        else:
            spatial_out = last_layer.spatial_forward(a_prev_L)
            residual = last_layer.z - spatial_out

        self.lambda_lagrange += self.beta * residual
    
    def fit(self, inputs: torch.Tensor, labels: torch.Tensor, warming: bool = False):
        """Orchestrates the fitting loop."""
        with torch.no_grad():
            if self.lambda_lagrange is None or self.lambda_lagrange.shape[0] != self._get_batchsize(inputs):
                self._init_states(inputs)
            
            time_steps = None
            if self._is_spiking():
                if self.train_method == "unrolled-random":
                    time_steps = random.sample(range(self.T - 1), self.T - 1)
                elif self.train_method == "unrolled-sequential":
                    time_steps = list(range(self.T - 1))
                    
            random_layers = random.sample(range(self.L - 1), self.L - 1)
            for l in random_layers:  
                    layer = self.layers[l]
                    next_layer = self.layers[l+1]
                    a_prev = inputs if l == 0 else self.layers[l - 1].a

                    cache_pinv= True if l == 0 else False
                    layer.update_weights(a_prev, cache_pinv=cache_pinv)
                    if self.bias: layer.update_bias(a_prev)
                    
                    lagrange = self.lambda_lagrange if l == self.L - 2 else None
                    if getattr(layer, 'train_method', '').startswith("unrolled") and getattr(layer, 'spiking', False):
                        layer.update_az_interleaved(next_layer, a_prev, lagrange, time_steps)
                    else:
                        layer.update_a(next_layer, lagrange)
                        layer.update_z(a_prev)

            # Update last layer
            last_layer = self.layers[-1]
            a_prev_L = self.layers[-2].a if len(self.layers) > 1 else inputs
             
            last_layer.update_weights(a_prev_L, self.lambda_lagrange)
            if self.bias: last_layer.update_bias(a_prev_L, self.lambda_lagrange)
            last_layer.update_z_last(a_prev_L, labels, self.lambda_lagrange, time_steps)

            # Lambda update
            if not warming:
                self._lambda_update(last_layer, a_prev_L)
