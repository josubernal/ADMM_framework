"""
ADMM Universal Manager

This module contains the core orchestrator for the network and ADMM optimizer.
ADMM breaks the network down into layer-wise sub-problems and updates in a loop:
1. Weight Updates 
2. Activation & Pre-activation Updates 
3. Dual Variable / Lagrange Multiplier Updates

IMPORTANT:
- Lambda update is rho not 2*rho, to match Cesare's implementation and ensure convergence.
This should be discussed in the future as it differs from the standard ADMM formulation.
"""

import torch
import torch.nn as nn
import random
import warnings
from .initializers import get_initializer
from .loss_functions import ADMM_SSE


class ADMM(nn.Module):
    """Universal Manager for ADMM networks.

    Handles both Static and Spiking ADMM networks automatically, orchestrating
    layer-wise optimization loops.
    """   
    def __init__(self, layers: nn.ModuleList, rho: float = 1.0, beta: float = 1.0, 
                 init: str = "s-uniform", bias: bool = False, device=None, loss_f=None,
                 train_method: str = "decoupled-backwards", layer_order:str="backwards", use_cholesky=True, **kwargs):
        super().__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_f = loss_f if loss_f is not None else ADMM_SSE()
        self.layers = layers
        self.L = len(self.layers)
        self.initialized = False
        
        self.lambda_lagrange = None
        self.init = init
        self.train_method = train_method
        self.layer_order = layer_order
        self.rho = rho
        self.beta = beta
        self.bias = bias
        
        
        self.use_cholesky=use_cholesky
        
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        self.T = kwargs.get('T', getattr(self, 'T', None))
        self.is_spiking = self.T is not None
        
        valid_methods = ["vectorized", "unrolled-random", "unrolled-sequential", "decoupled-random", "decoupled-sequential", "unrolled-backwards", "decoupled-backwards"]
        if self.train_method not in valid_methods:
            raise ValueError(f"Invalid train_method. Please select from: {valid_methods}")
        
        if not self.is_spiking and self.train_method!="vectorized":
            warnings.warn(
                "Non-spiking networks can only be trained using the 'vectorized' method. "
                "Automatically switching train_method to 'vectorized'.", 
                UserWarning
            )
            self.train_method = "vectorized"
            
        config = {'rho': self.rho, 'beta': self.beta, 'init': self.init, 'device':self.device, 'loss_f':self.loss_f,'use_cholesky': self.use_cholesky}
        config.update(kwargs)
        self._configure_layers(config)

    def _configure_layers(self, config_dict: dict):
        """Pushes global hyperparameters down to the individual layer setup methods.

        Args:
            config_dict (dict): Dictionary containing configuration parameters: rho, beta and init strategy.
        """
        for i, layer in enumerate(self.layers):
            layer.to(self.device)
            is_last = (i == self.L - 1)
            layer.setup(config_dict, is_last_layer=is_last)
          
    
    def _get_batchsize(self, inputs: torch.Tensor):
        """Extracts batch size dynamically based on the network type.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            int: The batch size (dim 1 for Spiking, dim 0 for Static).
        """
        return inputs.shape[1] if self.is_spiking else inputs.shape[0]  
    
    def _get_time_steps(self):
        """Helper that returns time-steps depending on the selected training method.

        Returns:
            list or None: A list of time steps if applicable, otherwise None.
        """
        time_steps = None
        if self.is_spiking:
            if self.train_method.endswith("random"):
               time_steps = random.sample(range(self.T), self.T)
            elif self.train_method.endswith("sequential"):
                time_steps = list(range(self.T)) 
            elif self.train_method.endswith("backwards"):                  
                time_steps = list(range(self.T - 1, -1, -1))
        return time_steps
    
    def _get_layers(self):
        """Helper that returns layer order depending on the selected method.

        Returns:
            list or None: A list of time steps if applicable, otherwise None.
        """
        if self.layer_order== "backwards":
            layer_indices = list(range(self.L -1, -1, -1))
        elif self.layer_order=="random-last":
            layer_indices = random.sample(range(self.L - 1), self.L - 1)
            layer_indices.append(self.L - 1)
        elif self.layer_order == "random":
            layer_indices = list(range(self.L))
            random.shuffle(layer_indices)
        elif self.layer_order == "sequential":                  
            layer_indices = list(range(self.L))
        else: 
            raise ValueError(f"Invalid layer order. Selected method {self.layer_order} does not exist, please read the documentation.")
        return layer_indices
    
    def _init_states(self, inputs: torch.Tensor):
        """Warm-starts the ADMM auxiliary variables 'z' and 'a'.

        Utilizes the selected initialization strategy and initializes the Lagrange multiplier.

        Args:
            inputs (torch.Tensor): The initial input tensor.
        """
        self.initialized = True        
        initializer = get_initializer(self.init)

        initializer.init_states(self.layers, inputs, self.device)
        
        # Initialize the Lagrange multiplier
        last_z = self.layers[-1].z
        target_shape = last_z[-1] if self.is_spiking else last_z
        self.lambda_lagrange = torch.zeros_like(target_shape, device=self.device)
        
    def forward_model(self, inputs: torch.Tensor):
        """Standard Feed-Forward pass used strictly for inference/evaluation.

        Args:
            inputs (torch.Tensor): The input tensor to the network.

        Returns:
            tuple:
                - torch.Tensor: The final network output.
                - list of float: The activation firing rates per layer.
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
                    layer_firing_rate = x.sum().item() / x.numel() if self.is_spiking else float("NaN")
                    firing_rates.append(layer_firing_rate)
                    
                final_z = z_pred
                
        final_out = final_z[-1] if self.is_spiking else final_z
        batch_size = inputs.size(1) if self.is_spiking else inputs.size(0)
            
        if final_out.dim() > 2:
            final_out = final_out.view(batch_size, -1)

        return final_out, firing_rates
    
    def _lambda_update(self, last_layer, a_prev_L):
        """Updates the Lagrange multiplier (lambda) based on the final layer constraint.

        Formula: lambda_new = lambda_old + rho * (z - forward(a_prev_L))

        Args:
            last_layer (nn.Module): The final layer of the network.
            a_prev_L (torch.Tensor): The activations from the penultimate layer or inputs.
        """
        if self.is_spiking:
            z_T = last_layer.z[-1]
            z_T_minus_1 = last_layer.z[-2]
            forward = last_layer.spatial_forward(a_prev_L[-1].unsqueeze(0)).squeeze(0)
            self.lambda_lagrange.add_(z_T, alpha=self.rho)
            self.lambda_lagrange.add_(z_T_minus_1, alpha=-self.rho*last_layer.deltas)
            self.lambda_lagrange.add_(forward, alpha=-self.rho)
        else:
            forward = last_layer.spatial_forward(a_prev_L)
            self.lambda_lagrange.add_(last_layer.z, alpha=self.rho)
            self.lambda_lagrange.add_(forward, alpha=-self.rho)

    def _optimize_w_and_b(self, layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None, cache_pinv: bool = False):
        """Unified interface for updating all trainable parameters (W, b).

        Args:
            layer (nn.Module): The layer to update.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
            cache_pinv (bool, optional): Whether to cache the pseudoinverse. Defaults to False.
        """
        layer.update_weights(a_prev, cache_pinv=cache_pinv, lambda_lagrange=lambda_lagrange)
        if self.bias: layer.update_bias(a_prev, lambda_lagrange)

    def _optimize_a_and_z(self, layer: nn.Module, next_layer: nn.Module, a_prev: torch.Tensor, lagrange: torch.Tensor = None, time_steps=None):
        """Unified interface for updating activations and pre-activations variables (a, z).

        Args:
            layer (nn.Module): The current layer being optimized.
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        if self.train_method.startswith("unrolled") and getattr(layer, 'spiking', False):
            layer.update_az_interleaved(next_layer, a_prev, lagrange, time_steps)
        elif self.train_method.startswith("decoupled") and getattr(layer, 'spiking', False):
            layer.update_a(next_layer, a_prev, lagrange)
            layer.update_z_decoupled(a_prev, time_steps)
        else:
            layer.update_a(next_layer, a_prev, lagrange)
            layer.update_z(a_prev)

    def _optimize_z_last(self, layer: nn.Module, a_prev: torch.Tensor, labels: torch.Tensor, time_steps=None):
        """Default static state optimization for the final layer's pre-activations.

        Args:
            layer (nn.Module): The final layer of the network.
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        if self.train_method != 'vectorized':
            layer.update_z_last_unrolled(a_prev, labels, self.lambda_lagrange, time_steps)
        else:
            layer.update_z_last(a_prev, labels, self.lambda_lagrange)
   
    @torch.no_grad()
    def fit(self, inputs: torch.Tensor, labels: torch.Tensor, warming: bool = False):
        """Orchestrates the fitting loop for the ADMM optimization process.

        Args:
            inputs (torch.Tensor): The input data tensor.
            labels (torch.Tensor): The target labels tensor.
            warming (bool, optional): If True, bypasses the lambda update. Defaults to False.
        """
        with torch.no_grad():
            if self.lambda_lagrange is None or self.lambda_lagrange.shape[0] != self._get_batchsize(inputs):
                self._init_states(inputs)
            
            time_steps= self._get_time_steps()
            layer_indices = self._get_layers()
            print(layer_indices)
            for l in layer_indices:  
                layer = self.layers[l]
                a_prev = inputs if l == 0 else self.layers[l - 1].a
                if l < self.L - 1:
                    next_layer = self.layers[l+1]
                    cache_pinv= True if l == 0 else False
                    lagrange = self.lambda_lagrange if l == self.L - 2 else None
                    self._optimize_w_and_b(layer, a_prev, cache_pinv=cache_pinv)
                    self._optimize_a_and_z(layer, next_layer, a_prev, lagrange, time_steps)
                    del a_prev
                    del lagrange
                elif l==self.L-1:
                    self._optimize_w_and_b(layer, a_prev, self.lambda_lagrange)
                    self._optimize_z_last(layer, a_prev, labels, time_steps)
            
            last_layer = self.layers[-1]
            a_prev_L = self.layers[-2].a if len(self.layers) > 1 else inputs
            if not warming:
                self._lambda_update(last_layer, a_prev_L)
