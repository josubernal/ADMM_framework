"""
ADMM Initializers Module

This module is responsible for initializing the weights and biases of ADMM layers.
By separating this from the main training logic, we keep the core optimization loop
clean and make it easy to experiment with different initialization strategies.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import warnings
import math

class ADMM_Initializer(ABC):
    """Abstract Base class for ADMM Initialization Strategies."""
    def init_weights(self, weight_shape: tuple, device:torch.device) -> torch.Tensor:
        """Initializes the weights for a layer.

        Args:
            weight_shape (tuple): The dimensions of the weight tensor.

        Returns:
            torch.Tensor: The initialized weight parameter without gradient tracking.
        """
        return torch.zeros(*weight_shape, device=device)
        
    def init_bias(self, bias_shape: tuple,  device:torch.device) -> torch.Tensor:
        """Initializes the bias for a layer.

        Args:
            bias_shape (tuple): The dimensions of the bias tensor.

        Returns:
            torch.Tensor: The initialized bias parameter (zeros) without gradient tracking.
        """
        return torch.zeros(*bias_shape, device=device)

    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        """Warm-starts the auxiliary variables 'z' and 'a' with random values.

        Args:
            layers (nn.ModuleList): The list of layers in the network.
            inputs (torch.Tensor): The initial input tensor to the network.
            device (torch.device): The device on which the tensors should be allocated.
        """
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                
                layer.z = z_pred.clone()
                layer.a = z_pred.clone() 

                x = layer.a

class WeightsZerosInitializer(ADMM_Initializer):
    """Zero weights (Baseline)."""  
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        return torch.zeros(*weight_shape, device=device)
    
class WeightsXavierInitializer(ADMM_Initializer):
    """Xavier weights."""
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        nn.init.xavier_normal_(w)
        return w

class WeightsRandomInitializer(ADMM_Initializer):
    """Standard Normal Random weights."""
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        return torch.randn(*weight_shape, device=device)

class WeightsPytorchDefaultInitializer(ADMM_Initializer):
    """
    Exact replication of PyTorch's default initialization 
    for nn.Linear and nn.Conv2d.
    """
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        # This is the exact source-code configuration used by PyTorch
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        return w
              
class ZUniform(ADMM_Initializer):
    """
    Pure Random Initialization. 
    Forces a massive, chaotic constraint violation on the ADMM solver.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x) 
                layer.z = torch.randn_like(z_pred)
                a_pred = layer.h(layer.z) if layer.h is not None else layer.z
                layer.a = a_pred.clone()
                
                x = layer.a 

class StatesUniform(ADMM_Initializer):
    """
    Pure Random Initialization. 
    Forces a massive, chaotic constraint violation on the ADMM solver.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        """Warm-starts the auxiliary variables 'z' and 'a' with random values.

        Args:
            layers (nn.ModuleList): The list of layers in the network.
            inputs (torch.Tensor): The initial input tensor to the network.
            device (torch.device): The device on which the tensors should be allocated.
        """
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred)     
                layer.z = torch.rand_like(z_pred)
                layer.a = torch.rand_like(a_pred)
                x = a_pred 

class RelaxedSpikeInitializer(ADMM_Initializer):
    """
    Capitalizes on the 'Energy Shock' discovery. 
    Weights are initialized to zero to allow algebraic overwriting.
    Pre-activations (z) are given Gaussian noise to simulate membrane potential variance.
    Activations (a) are forced into a dense Bernoulli distribution (p=0.5) to provide 
    maximum information density to the first ADMM weight projection.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                # 1. Do a forward pass just to get the exact tensor geometries
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred

                layer.z = torch.randn_like(z_pred)
                layer.a = torch.randint(0, 2, size=a_pred.shape, dtype=a_pred.dtype, device=device)
                
                x = a_pred

def get_initializer(init_type: str) -> ADMM_Initializer:
    """Factory function to retrieve the correct initializer strategy.
    Args:
        init_type (str): The string identifier for the initialization strategy.

    Returns:
        ADMM_Initializer: An instance of the requested initialization strategy.

    Raises:
        ValueError: If the provided init_type is not found in the defined strategies.
    """
    strategies = {
        "zeros": WeightsZerosInitializer(),
        "xavier": WeightsXavierInitializer(),
        "wrandom": WeightsRandomInitializer(),
        "pytorch": WeightsPytorchDefaultInitializer(),
        "z-uniform": ZUniform(),
        "s-uniform": StatesUniform(),
        "relaxed":  RelaxedSpikeInitializer(),
    }
    if init_type not in strategies:
        raise ValueError(f"Initialization method '{init_type}' not defined. Options: {list(strategies.keys())}")
    return strategies[init_type]
