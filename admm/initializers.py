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

#ABLATION STUDY INITS
class BaseWeightTester(ADMM_Initializer):
    """
    Base class for Phase 1 testing. 
    Forces the Exact Warm Start for states so ADMM starts with 0 error.
    """
    def init_bias(self, bias_shape: tuple, device:torch.device) -> torch.Tensor:
        # Fixed: Bias is 1D and cannot use Kaiming/Xavier. Zero is standard.
        return torch.zeros(*bias_shape, device=device)

    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred
                layer.z = z_pred.clone()
                layer.a = a_pred.clone()
                x = a_pred

class WeightsZerosInitializer(BaseWeightTester):
    """Zero weights (Baseline)."""  
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        return torch.zeros(*weight_shape, device=device)
    
class WeightsKaimingInitializer(BaseWeightTester):
    """Kaiming (He) weights. Gold standard for Static ReLU networks."""
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        return w

class WeightsXavierInitializer(BaseWeightTester):
    """Xavier weights."""
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        nn.init.xavier_normal_(w)
        return w

class WeightsRandomInitializer(BaseWeightTester):
    """Standard Normal Random weights."""
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        return torch.randn(*weight_shape, device=device)

class WeightsPytorchDefaultInitializer(BaseWeightTester):
    """
    Exact replication of PyTorch's default initialization 
    for nn.Linear and nn.Conv2d.
    """
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        # This is the exact source-code configuration used by PyTorch
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        return w
        
    def init_bias(self, bias_shape: tuple, device:torch.device) -> torch.Tensor:
        # PyTorch calculates a bound based on fan_in and uses uniform distribution
        # For simplicity in ADMM, starting default biases at 0 is still acceptable,
        # but to be strictly faithful to PyTorch:
        return torch.zeros(*bias_shape, device=device)
    
class WeightsDataDrivenInitializer(BaseWeightTester):
    """
    Data-Driven Initialization (LSUV).
    Dynamically scales weights based on the first batch of data 
    to force pre-activation variance to exactly 1.0.
    """
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        # Start with standard normal, it will be corrected instantly in init_states
        w = torch.empty(*weight_shape, device=device)
        nn.init.normal_(w, mean=0.0, std=0.1)
        return w

    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                # 1. Do a test forward pass
                z_pred = layer.forward(x)
                
                # 2. Measure actual standard deviation of the data
                std = z_pred.std() + 1e-5
                
                # 3. Force variance to 1.0 by dividing the weights
                layer.W.data = layer.W.data / std
                
                # 4. Re-calculate the corrected forward pass
                z_pred_corrected = layer.forward(x)
                a_pred = layer.h(z_pred_corrected) if layer.h is not None else z_pred_corrected
                
                # 5. Exact Warm Start clone (from Base class)
                layer.z = z_pred_corrected.clone()
                layer.a = a_pred.clone()
                x = a_pred
                
class WeightsSNNThresholdInitializer(BaseWeightTester):
    """
    Threshold-scaled Normal weights. 
    Gold standard for Spiking networks (SNNs).
    """
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        
        # CORRECTED: Handle both 2D Linear and 4D Conv shapes
        if len(weight_shape) == 4:
            # fan_in = in_channels * kernel_h * kernel_w
            fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3]
        elif len(weight_shape) == 2:
            # fan_in = in_features
            fan_in = weight_shape[1]
        else:
            fan_in = weight_shape[0]
            
        # Scale std dev so membrane potentials gently reach the threshold
        std = (self.threshold / math.sqrt(fan_in)) * 0.5 
        nn.init.normal_(w, mean=0.0, std=std)
        return w

class StatesStaticBase(ADMM_Initializer):
    """Locks in Kaiming Weights for Static Networks."""
    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        return w
        
    def init_bias(self, bias_shape: tuple, device:torch.device) -> torch.Tensor:
        return torch.zeros(*bias_shape, device=device)
        
class StatesSpikingBase(ADMM_Initializer):
    """Locks in Threshold Weights for Spiking Networks."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def init_weights(self, weight_shape: tuple, device:torch.device=None) -> torch.Tensor:
        w = torch.empty(*weight_shape, device=device)
        if len(weight_shape) == 4:
            fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3]
        elif len(weight_shape) == 2:
            fan_in = weight_shape[1]
        else:
            fan_in = weight_shape[0]
            
        std = (self.threshold / math.sqrt(fan_in)) * 0.5 
        nn.init.normal_(w, mean=0.0, std=std)
        return w

    def init_bias(self, bias_shape: tuple, device:torch.device) -> torch.Tensor:
        return torch.zeros(*bias_shape, device=device)

class StaticStateWarmStart(StatesStaticBase):
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred
                layer.z = z_pred.clone()
                layer.a = a_pred.clone()
                x = a_pred

class SpikingStateWarmStart(StatesSpikingBase):
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred
                layer.z = z_pred.clone()
                layer.a = a_pred.clone()
                x = a_pred

class StaticStateZeros(StatesStaticBase):
    """
    Weights are Kaiming. 
    States are forced to 0, creating maximum ADMM constraint shock.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                # We do the forward pass ONLY to get the mathematical shapes
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred
                
                # Overwrite states with absolute zero
                layer.z = torch.zeros_like(z_pred)
                layer.a = torch.zeros_like(a_pred)
                
                # We must pass the zeroed 'a' to the next layer to simulate the dead signal
                x = layer.a 

class SpikingStateZeros(StatesSpikingBase):
    """
    Weights are Threshold-Scaled. 
    States are forced to 0, simulating a completely dead SNN.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred) if layer.h is not None else z_pred

                layer.z = torch.zeros_like(z_pred)
                layer.a = torch.zeros_like(a_pred)
                
                x = layer.a
     
class StaticStateNoisy(StatesStaticBase):
    """
    Adds 5% Gaussian noise to the forward pass to break symmetry 
    and encourage ADMM exploration.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                
                # Add 5% noise relative to the standard deviation
                noise_scale = z_pred.std() * 0.05
                layer.z = z_pred + (torch.randn_like(z_pred) * noise_scale)
                
                # Calculate 'a' from the noisy 'z'
                a_pred = layer.h(layer.z) if layer.h is not None else layer.z
                layer.a = a_pred.clone()
                
                x = a_pred

class SpikingStateNoisy(StatesSpikingBase):
    """
    Adds 5% Gaussian noise to the forward pass. 
    Maintains strict SNN binary constraints via Heaviside.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x)
                
                noise_scale = z_pred.std() * 0.05
                layer.z = z_pred + (torch.randn_like(z_pred) * noise_scale)
                
                a_pred = layer.h(layer.z) if layer.h is not None else layer.z
                layer.a = a_pred.clone()
                
                x = a_pred   
                
class StaticStateRandom(StatesStaticBase):
    """
    Pure Random Initialization. 
    Forces a massive, chaotic constraint violation on the ADMM solver.
    """
    def init_states(self, layers: nn.ModuleList, inputs: torch.Tensor, device: torch.device):
        x = inputs.to(device)
        with torch.no_grad():
            for layer in layers:
                z_pred = layer.forward(x) # Only used to get the tensor shape
                
                # Pure unscaled random noise
                layer.z = torch.randn_like(z_pred)
                
                a_pred = layer.h(layer.z) if layer.h is not None else layer.z
                layer.a = a_pred.clone()
                
                x = layer.a 

class SpikingStateRandom(StatesSpikingBase):
    """
    Pure Random Initialization.
    Tests if the SNN can recover from chaotic, unscaled binary spikes.
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
        "wzeros": WeightsZerosInitializer(),
        "wkaiming": WeightsKaimingInitializer(),
        "wxavier": WeightsXavierInitializer(),
        "wrandom": WeightsRandomInitializer(),
        "wpytorch": WeightsPytorchDefaultInitializer(),
        "wdata": WeightsDataDrivenInitializer(),
        "wthreshold": WeightsSNNThresholdInitializer(),
        "szeros": StaticStateZeros(),
        "swarm": StaticStateWarmStart(),
        "snoisy":StaticStateNoisy(),
        "srandom": StaticStateRandom(),
        "spiking-szeros": SpikingStateZeros(),
        "spiking-swarm": SpikingStateWarmStart(),
        "spiking-snoisy":SpikingStateNoisy(),
        "spiking-srandom": SpikingStateRandom()        
    }
    if init_type not in strategies:
        raise ValueError(f"Initialization method '{init_type}' not defined. Options: {list(strategies.keys())}")
    return strategies[init_type]
