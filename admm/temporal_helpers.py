"""
ADMM Temporal Math

This module isolates the stateless mathematical operations required for 
Spiking Neural Networks (SNNs)´, helping to mantain the code clean and modular.
"""

import torch
from dataclasses import dataclass

@dataclass
class TemporalCache:
    """
    Holds precomputed tensors for the unrolled SNN temporal loops.
    Replaces the opaque dictionary to provide type safety and IDE auto-completion.
    """
    forward_pass: torch.Tensor
    denominator_main: torch.Tensor
    denominator_last: torch.Tensor
    term2: torch.Tensor
    @classmethod
    def build(cls, layer, next_layer, a_prev, lambda_lagrange):
        """Precomputes and distributes operations to accelerate the unrolled loop.
        
            This factory method calculates the static portions of the ADMM activation 
            update that do not change during the unrolled sequence.

            Precomputes:
            1- Forward pass
            2- a update denominator (for both t<T and t=T)
            3- Term2 = next_layer(rho * W^T * adjoint(Y + lambda_penalty) + temporal penalties) 

            Args:
                layer (nn.Module): The current layer.
                next_layer (nn.Module): The subsequent layer in the network.
                a_prev (torch.Tensor): The previous layer's activations.
                lambda_lagrange (torch.Tensor): The Lagrange multiplier.

            Returns:
                TemporalCache: A typed data class containing the precomputed matrices.
        """
        #1- Forward Pass
        forward_pass = layer.spatial_forward(a_prev)
        
        #2- Denominator      
        temp_num, temp_den= layer._get_temporal_a_penalties(forward_pass)
        denominator_main, denominator_last, _= next_layer._get_a_denominator(
            beta_current=layer.beta,
            a_shape=layer.a.shape,
            temp_den=temp_den,
            unrolled=True
        )

        #3- Term 2
        target_z = next_layer._get_Y(include_reset=False)
            
        if lambda_lagrange is not None:
            target_z = next_layer._apply_lagrange_to_Y(target_z, lambda_lagrange)
                
        projected_target = next_layer.adjoint_operator(target_z, original_input_shape=layer.a.shape)
        base_RHS = next_layer.rho * projected_target
        TERM2 = base_RHS + temp_num if getattr(layer, 'use_reset', False) and isinstance(temp_num, torch.Tensor) else base_RHS          

        term2 = TERM2
         
        return cls(
                forward_pass=forward_pass,
                denominator_main=denominator_main,
                denominator_last=denominator_last,
                term2=term2
        )



def fold_time(x: torch.Tensor):
    """Folds the Time and Batch dimensions together for spatial operations.

        Args:
            x (torch.Tensor): The input tensor, potentially 5D or 3D.

        Returns:
            tuple:
                - torch.Tensor: The flattened tensor ready for spatial operations.
                - tuple or None: The original `(Time, Batch)` shape to reconstruct the tensor later.
    """
    if x.dim() in [3, 5]: 
        return x.reshape(x.size(0) * x.size(1), *x.shape[2:]), x.shape[:2]
    return x, None

def unfold_time(x_flat: torch.Tensor, tb_shape: tuple):
    """Unfolds the Time and Batch dimensions back out after spatial operations.

        Args:
            x_flat (torch.Tensor): The spatially processed flat tensor.
            tb_shape (tuple): The `(Time, Batch)` shape returned by `_fold_time`.

        Returns:
            torch.Tensor: The reconstructed temporal tensor.
    """
    if tb_shape is None: 
        return x_flat
    return x_flat.reshape(tb_shape[0], tb_shape[1], *x_flat.shape[1:])
    
def compute_temporal_dependencies(z: torch.Tensor, a: torch.Tensor, deltas: float, thetas: float, 
                                  use_reset: bool = True, include_reset: bool = True) -> torch.Tensor:     
    """Computes the physical voltage leakage and threshold reset over time.

        Args:
            include_reset (bool, optional): Whether to include the spike reset penalty. Defaults to True.

        Returns:
            torch.Tensor: The calculated temporal dependencies tensor.
    """
    out = torch.zeros_like(z)
    out[1:] = deltas * z[:-1]
    
    if use_reset and include_reset:
        out[1:] -= thetas * a[:-1]
        
    return out  


def get_temporal_a_penalties(z: torch.Tensor, forward_pass: torch.Tensor, deltas: float, 
                             thetas: float, rho: float, use_reset: bool = True):
    """Calculates the temporal penalties for the $a$ update.

        Mathematical formulation:
        - Numerator (main timesteps): -  theta * rho * (z - delta * z_t-1 - forward)$
        - Numerator (last timestep): 0
        - Denominator (main timesteps): rho * theta^2
        - Denominator (last timestep): 0

        Args:
            forward_pass (torch.Tensor): The computed spatial forward pass.

        Returns:
            tuple:
                - torch.Tensor: The numerator penalty tensor.
                - torch.Tensor: The denominator penalty tensor.
    """
    if not use_reset: 
        return 0.0, 0.0
    
    num = torch.zeros_like(z)
    den = torch.zeros_like(z)        
    
    num[:-1] = -thetas * rho * (z[1:] - deltas * z[:-1] - forward_pass[1:])
    den[:-1] = rho * (thetas ** 2)
    
    return num, den

def get_temporal_z_penalties(z: torch.Tensor, forward_pass: torch.Tensor, labels: torch.Tensor, 
                             lamb: torch.Tensor, deltas: float, rho: float):
    """Calculates the temporal penalties for the z update.

        Mathematical formulation:
        - Numerator (main timesteps): rho * delta * (z - forward)_t+1
        - Numerator (second last timestep): adds (lambda * delta) / 2
        - Numerator (last timestep): adds labels - (lambda / 2)
        - Denominator (main timesteps): rho * delta^2
        - Denominator (last timestep): adds 1.0

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lamb (torch.Tensor): The Lagrange multiplier.

        Returns:
            tuple:
                - torch.Tensor: The numerator penalty tensor.
                - torch.Tensor: The denominator penalty tensor.
    """
    num = torch.zeros_like(z)
    den = torch.zeros_like(z)
    
    num[:-1] = rho * deltas * (z - forward_pass)[1:]
    den[:-1] = rho * (deltas ** 2) 
    
    num[-2] += (lamb * deltas) / 2
    num[-1] += labels - (lamb / 2)
    den[-1] += 1.0
    
    return num, den
