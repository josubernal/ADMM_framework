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
    adjoint: torch.Tensor
    @classmethod
    def build(cls, layer, next_layer, a_prev, lambda_lagrange:torch.Tensor=None):
        """Precomputes and distributes operations to accelerate the unrolled loop.
        
            This factory method calculates the static portions of the ADMM activation 
            update that do not change during the unrolled sequence.

            Precomputes:
            1- Forward pass
            2- a update denominator (for both t<T and t=T)
            3- a update numerator without the h term 

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
        denominator_main, denominator_last, _= next_layer._get_a_denominator(
            beta_current=layer.beta,
            rho_current = layer.rho,
            thetas_current  = layer.thetas,
            a_shape=layer.a.shape,
            unrolled=True
        )

        #3- Term 2
        adjoint = layer._get_a_adjoint(next_layer=next_layer, lambda_lagrange=lambda_lagrange)# forward_pass=forward_pass)  DEPRECATED   

        return cls(
                forward_pass=forward_pass,
                denominator_main=denominator_main,
                denominator_last=denominator_last,
                adjoint=adjoint
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
    out = torch.empty_like(z)
    out[0].zero_()
    out_slice = out[1:]
    out_slice.copy_(z[:-1])
    out_slice.mul_(deltas)
    if use_reset and include_reset:
        out[1:].add_(a[:-1], alpha=-thetas)
        
    return out  

def get_spiking_v(z, bias, temporal_dependencies, rho, lambda_lagrange=None, broadcast_func=None) -> torch.Tensor:
    """Returns the spiking target, including temporal leakage and reset penalties.

        Args:
            include_reset (bool, optional): Whether to include the spike reset penalty. Defaults to True.

        Returns:
            torch.Tensor: The calculated spiking target tensor.
    """
    v = z.clone()
    if isinstance(bias, torch.Tensor):
        v.sub_(bias)
    v.sub_(temporal_dependencies)
    if lambda_lagrange is not None and broadcast_func is not None:
        lam_sp = broadcast_func(lambda_lagrange, z[-1])
        v[-1].add_(lam_sp, alpha=1.0 / rho)
    return v
    
def get_spiking_a_denominator(WtW, in_features, beta_current, rho_next, temporal_penalty, unrolled=False):
    """
    Computes the denominator matrix for the activation (a) update step.

        Formula:
        result= beta*I + rho*W^TW + rho*theta^2*I if t<T
        result= beta*I + rho*W^TW if t=T

    
    Args:
        W: The expanded/flattened weights of the NEXT layer.
        beta_current: Beta of the current layer.
        rho_next: Rho of the NEXT layer (for spatial term).
        temporal_penalty: Precomputed rho_curr * theta_curr^2 (for temporal term).
    
    Returns:
        tuple:
            - torch.Tensor: The computed denominator_main matrix (for t < T).
            - torch.Tensor: The computed denominator_last matrix (for t = T).
            - int: The number of input features.
    """
    
    denominator_last = WtW * rho_next
    
    denominator_last.diagonal().add_(beta_current)
   
    denominator_main = denominator_last.clone()
    denominator_main.diagonal().add_(temporal_penalty)
    
    if unrolled:
        denominator_main = torch.linalg.inv(denominator_main).transpose(-2, -1)          
        denominator_last = torch.linalg.inv(denominator_last).transpose(-2, -1)
            
    return denominator_main, denominator_last, in_features


def get_spiking_a_adjoint(layer, next_layer, lambda_lagrange): #forward_pass):DEPRECATED
    """
        Computes the linear components of the ADMM numerator for activation updates.

        Formula:
        result= rho * adjoint(v_{l+1}) if t=T
        result= rho * adjoint(v_{l+1}) - thetas * rho * (z_t+1 - delta * z_t - forward ) otherwise

        Args:
            layer (ADMM_Layer): The current layer being updated.
            next_layer (ADMM_Layer): The subsequent layer in the network.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            forward_pass (torch.Tensor): The pre-computed spatial forward pass.

        Returns:
            torch.Tensor: A tensor shaped like `layer.z`.
    """
    v = next_layer._get_v(include_reset=False, lambda_lagrange=lambda_lagrange)     
    adjoint = next_layer.adjoint_operator(v, original_input_shape=layer.a.shape)
    # temporal_penalty = torch.zeros_like(layer.z)
    # temporal_penalty[:-1] = -layer.thetas * layer.rho * (layer.z[1:] - layer.deltas * layer.z[:-1] - forward_pass[1:]) DEPRECATED
    return  adjoint.mul_(next_layer.rho) #+ temporal_penalty  