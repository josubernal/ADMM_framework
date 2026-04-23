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

class ADMM_Initializer(ABC):
    """Abstract Base class for ADMM Initialization Strategies."""
    @abstractmethod
    def init_weights(self, weight_shape: tuple, device:torch.device) -> torch.Tensor:
        """Initializes the weights for a layer.

        Args:
            weight_shape (tuple): The dimensions of the weight tensor.

        Returns:
            torch.Tensor: The initialized weight parameter without gradient tracking.
        """
        pass
        
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
                a_pred = layer.h(z_pred)     
                layer.z = torch.rand_like(z_pred)
                layer.a = torch.rand_like(a_pred)
                x = a_pred 

class ZerosInitializer(ADMM_Initializer):
    """Initialization strategy that sets weights to zero."""
    def init_weights(self, weight_shape: tuple,  device:torch.device=None) -> torch.Tensor:
        return torch.zeros(*weight_shape,  device=device)

class ZerosPassInitializer(ZerosInitializer):
    """Initialization strategy that uses a forward pass with zeroed weights."""
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        """Warm-starts the auxiliary variables 'z' and 'a'. 'z' with random values and 'a' using a forward pass.

        Args:
            layers (nn.ModuleList): The list of layers in the network.
            inputs (torch.Tensor): The initial input tensor to the network.
            device (torch.device): The device on which the tensors should be allocated.
        """
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                layer.z = torch.rand_like(z_pred) #CHANGE THIS TO MEAN 1 VAR 0.1
                layer.a = layer.h(layer.z)
                x = layer.a

class XavierInitializer(ADMM_Initializer):
    """Initialization strategy that applies Xavier uniform initialization to weights."""
    def init_weights(self, weight_shape: tuple,  device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape,  device=device)
        nn.init.xavier_uniform_(w)
        return w

class ZerosRNGInitializer(ZerosInitializer):
    """Deprecated initialization strategy kept for legacy support."""
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        """Warm-starts the auxiliary variables using random number generation.

        Args:
            layers (nn.ModuleList): The list of layers in the network.
            inputs (torch.Tensor): The initial input tensor to the network.
            device (torch.device): The device on which the tensors should be allocated.
        """
        warnings.warn("The 'zeros-rng' initialization is deprecated. Use 'zeros' for production.", UserWarning)
        x = inputs.to(device)
        z_preds, a_preds = [], []
        
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred)
                z_preds.append(z_pred)
                a_preds.append(a_pred)
                x = a_pred 
                
            for i, layer in enumerate(layers):
                    layer.z = torch.rand(z_preds[i].shape, device=device)
                        
            for i, layer in enumerate(layers):
                if i <  len(layers) - 1: 
                    layer.a = torch.rand(a_preds[i].shape, device=device)
                else:
                    layer.a = torch.zeros_like(a_preds[i])
                    

class HeWarmStartInitializer(ADMM_Initializer):
    """Kaiming (He) weights + Exact Forward Pass States for ReLU networks."""
    
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        # Kaiming Normal is the gold standard for ReLU-based networks
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        return w

    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred)     
                
                # Use the EXACT forward pass, not random noise
                layer.z = z_pred.clone() 
                layer.a = a_pred.clone()
                x = a_pred
class GaussianPassInitializer(ADMM_Initializer):
    """Weights to Xavier, states initialized to specific Gaussian distribution."""
    
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        nn.init.xavier_normal_(w)
        return w

    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        variance = 0.1
        std_dev = variance ** 0.5
        mean = 1.0
        
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                
                # Gaussian distribution with Mean 1, Variance 0.1
                noise = torch.randn_like(z_pred) * std_dev + mean
                layer.z = noise
                layer.a = layer.h(layer.z)
                x = layer.a

class DataDrivenInitializer(ADMM_Initializer):
    """Scales weights dynamically based on the variance of the first batch."""
    
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        # Start with standard normal
        w = torch.randn(*weight_shape, device=device)
        return w

    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                # Get raw pre-activation
                z_pred = layer.forward(x)
                
                # Calculate standard deviation of the batch
                std = z_pred.std() + 1e-5
                mean = z_pred.mean()
                # Scale the weights of the layer so the output variance is 1
                layer.W.data = layer.W.data / std
                if hasattr(layer, 'b') and layer.b is not None:
                    layer.b.data = layer.b.data - (mean / std)
                
                # Recalculate with corrected weights
                z_pred_corrected = layer.forward(x)
                a_pred = layer.h(z_pred_corrected)
                
                layer.z = z_pred_corrected.clone()
                layer.a = a_pred.clone()
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
        "zeros": ZerosInitializer(),
        "zeros-rng": ZerosRNGInitializer(),
        "zeros-pass": ZerosPassInitializer(),
        "xavier": XavierInitializer(),
        "kaiming": HeWarmStartInitializer(),
        "gaussian": GaussianPassInitializer(),
        "data": DataDrivenInitializer()       
        
    }
    if init_type not in strategies:
        raise ValueError(f"Initialization method '{init_type}' not defined. Options: {list(strategies.keys())}")
    return strategies[init_type]
