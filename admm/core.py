"""
ADMM Core Interface

This module defines the absolute base for all ADMM layers. 
It manages the fundamental auxiliary variables ('a' and 'z'), handles the 
configuration setup, and dictates the generic forward passes.

It is strictly non-parametric (no weights/biases) and non-temporal.
"""

import torch
import torch.nn as nn
from .activations import ADMM_Identity

####################################################################################################
# Base Layer Interface
####################################################################################################           

class ADMM_Layer(nn.Module):
    """The core Base Layer Interface for all ADMM modules.
    
    Manages the fundamental auxiliary variables, handles the configuration setup.
    """
    def __init__(self, h: nn.Module = None):
        super().__init__()
        self.device = None
        self.use_cholesky= None
        self.rho = None
        self.beta = None
        self.deltas = None
        self.thetas = None
        
        self.z = None
        self.a = None
                
        self.h = h if h is not None else ADMM_Identity()
    
    def setup(self, config: dict, is_last_layer: bool = False):
        """Receives global ADMM hyperparameters from the manager and cascades them.

        Args:
            config (dict): Dictionary containing global hyperparameters (e.g., rho, beta).
            is_last_layer (bool, optional): Flag indicating if this is the final layer. Defaults to False.
        """
        for key, val in config.items():
            setattr(self, key, val)
            
        if is_last_layer and hasattr(self, 'use_reset'):
            self.use_reset = False
            
        if hasattr(self, 'h') and hasattr(self.h, 'setup'):
            self.h.setup(config)
    
    def _broadcast_to_match(self, tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """Helper method to safely align tensor dimensions for element-wise operations.

        Args:
            tensor (torch.Tensor): The tensor to be broadcasted.
            target_tensor (torch.Tensor): The reference tensor whose dimensions are matched.

        Returns:
            torch.Tensor: The broadcasted tensor matching the target's dimensionality.
        """
        if tensor.dim() < target_tensor.dim():
            missing_dims = target_tensor.dim() - tensor.dim()
            return tensor.view(*tensor.shape, *([1] * missing_dims))
        return tensor 
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        """Standard sequential pass for initialization or inference.

        - For Static Networks: Simply returns the spatial transformation (y = Wx).
        - For Spiking Networks (SNNs): Simulates the Leaky Integrate-and-Fire (LIF) 
        mechanics step-by-step over the time dimension (T).

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the spatial (and temporal, if spiking) pass.
        """
        y = self.spatial_forward(x)             
        return y
    def vectorized_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized ADMM pass for optimization and constraint evaluation.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            torch.Tensor: The evaluated constraints including temporal dependencies.
        """
        y = self.spatial_forward(x)             
        return y
   
    def update_z_last(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps=None):
        """Solves the proximal update for the 'z' variable for the last layer.

        Formula:
        z_last= numerator / denominator, where
        numerator = rho * forward(a_prev) + (2*labels - lambda)
        denominator = 2 + rho

        For spiking networks, this also incorporates temporal penalties into the 
        numerator and denominator.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        forward = self.vectorized_forward(a_prev) 
        labels = self._broadcast_to_match(labels, forward)
        lambda_lagrange = self._broadcast_to_match(lambda_lagrange, forward)

        forward.mul_(self.rho)
        forward.add_(labels, alpha=2.0)
        forward.sub_(lambda_lagrange)
        forward.div_(2.0 + self.rho)
        
        self.z.copy_(forward)        
        

    