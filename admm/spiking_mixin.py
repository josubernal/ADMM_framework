"""
ADMM Spiking Mixin

This module provides temporal modeling and unrolled solvers for Spiking 
Neural Networks (SNNs). It is designed to be used as a Mixin, overriding 
standard spatial hooks with temporal dependencies (leakage, reset).
"""

import torch
import torch.nn as nn
from .solvers import solve_spiking_system, solve_woodbury_system
from types import SimpleNamespace
from .temporal_helpers import TemporalCache, fold_time, unfold_time, compute_temporal_dependencies, get_spiking_v, get_spiking_a_denominator, get_spiking_a_adjoint

############################################################################################################
#Spiking mixing
############################################################################################################ 

class ADMM_Spiking:
    """Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs)."""
    def _fold_time(self, x: torch.Tensor):
        """Delegates to temporal_helpers.fold_time."""
        return fold_time(x=x)

    def _unfold_time(self, x_flat: torch.Tensor, tb_shape: tuple):
        """Delegates to temporal_helpers.unfold_time."""
        return unfold_time(x_flat=x_flat, tb_shape=tb_shape)
    
    def _compute_temporal_dependencies(self, include_reset: bool = True) -> torch.Tensor:     
        """Delegates to temporal_helpers.compute_temporal_dependencies."""
        return compute_temporal_dependencies(
            self.z, self.a, self.deltas, self.thetas, 
            getattr(self, 'use_reset', False), include_reset
        )
    
    def _get_v(self, include_reset: bool = True, lambda_lagrange: torch.Tensor = None) -> torch.Tensor:
        """Delegates to temporal_helpers.get_spiking_v."""
        return get_spiking_v(
            z=self.z,
            bias=self._format_bias(),
            temporal_dependencies=self._compute_temporal_dependencies(include_reset),
            rho=self.rho,
            lambda_lagrange=lambda_lagrange,
            broadcast_func=self._broadcast_to_match
        )
    
    def _get_a_denominator(self, beta_current, rho_current, thetas_current, a_shape, unrolled=False):
        """Delegates to temporal_helpers.get_spiking_a_denominator."""
        temporal_penalty = rho_current * (thetas_current ** 2)
        W = self._get_expanded_weights(a_shape)
        out_features, in_features = W.shape

        if getattr(self, 'W', None) is not None and self.W.dim() == 2 and in_features > out_features:
                main_dict = {'W': W, 'beta': beta_current + temporal_penalty, 'rho': self.rho}
                last_dict = {'W': W, 'beta': beta_current, 'rho': self.rho}
                return main_dict, last_dict, in_features
            
        WtW, in_features = self._get_WtW(a_shape=a_shape)
       
        
        return get_spiking_a_denominator(
            WtW=WtW,
            in_features=in_features,
            beta_current=beta_current,
            rho_next=self.rho,
            temporal_penalty=temporal_penalty,
            unrolled=unrolled
        )
    
    def _get_a_adjoint(self, next_layer: nn.Module, lambda_lagrange: torch.Tensor) -> torch.Tensor: # forward_pass:torch.Tensor DEPRECATED
        """Delegates to temporal_helpers.get_spiking_a_adjoint."""
        return get_spiking_a_adjoint(
            layer=self,
            next_layer=next_layer,
            lambda_lagrange=lambda_lagrange
            #forward_pass=forward_pass DEPRECATED
        )
        
    def _create_cache(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Delegates cache construction to the TemporalCache factory.
            
        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.

        Returns:
            TemporalCache: A typed data class containing the precomputed matrices."""
        return TemporalCache.build(self, next_layer, a_prev, lambda_lagrange)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        """Standard sequential pass for initialization or inference.

        Simulates the spiking mechanics step-by-step over the time dimension.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the full temporal simulation.
        """
        y = self.spatial_forward(x)     
        z = torch.zeros_like(y)
        z_prev = torch.zeros_like(y[0])
        a_prev = torch.zeros_like(y[0])   
        for t in range(self.T):
            reset = self.thetas * a_prev if (t > 0 and getattr(self, 'use_reset', False)) else 0.0
            z_t = y[t] + self.deltas * z_prev - reset
            z[t] = z_t
            z_prev = z_t
            a_prev = self.h(z_t)
        return z 
    
    def vectorized_forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Vectorized ADMM pass for optimization and constraint evaluation.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            torch.Tensor: The evaluated constraints including temporal dependencies.
        """
        z = self.spatial_forward(a_prev) + self._compute_temporal_dependencies()
        return z

    def update_a(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Manages the vectorized activation (a) update for a specific timestep.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        
        # self.beta * self.h(self.z) + adjoint + temporal penalty
        numerator = self._get_a_adjoint(
            next_layer=next_layer,
            lambda_lagrange=lambda_lagrange
        )
        numerator.add_(self.h(self.z), alpha=self.beta)
        
        #Temporal penalty  =  -rho*thetas(z_t+1 -forward_t+1 -delta*z)
        num_slice = numerator[:-1]
        forward_pass = self.spatial_forward(a_prev)
        num_slice.sub_(forward_pass[1:], alpha=-self.thetas * self.rho)
        del forward_pass
        num_slice.add_(self.z[1:], alpha=-self.thetas * self.rho)
        num_slice.add_(self.z[:-1], alpha=self.deltas*self.thetas * self.rho)
                
        denominator_main, denominator_last, in_features = next_layer._get_a_denominator(  
            a_shape=self.a.shape,
            beta_current=self.beta,
            rho_current=self.rho,
            thetas_current=self.thetas,
            unrolled=False)
        
        new_a = next_layer.solve_activation_system(
            numerator=numerator,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            a_shape=self.a.shape,
            in_features=in_features
        )           
        
        new_a = torch.clamp(new_a, min=0.0, max=1.0)
        self.a.copy_(new_a)   

    def solve_activation_system(self, numerator: torch.Tensor,  denominator_main:torch.Tensor, denominator_last:torch.Tensor,a_shape: tuple, in_features:int) -> torch.Tensor:
        return solve_spiking_system(
            A_main=denominator_main, 
            A_last=denominator_last, 
            B=numerator, 
            a_shape=a_shape, 
            in_features=in_features,
            T=numerator.size(0) 
        )
        
    def update_bias(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Averages the residual errors across spatial and temporal dimensions to update the bias vector.

        Formula:
        b = mean(z-A(a_prev)-temporal_dependencies)
        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        in_mean = self.spatial_forward(a_prev, use_bias=False)
        in_mean.add_(self._compute_temporal_dependencies())
        in_mean.neg_().add_(self.z)
        
        if lambda_lagrange is not None:
            lambda_lagrange= self._broadcast_to_match(lambda_lagrange, self.z[-1])
            in_mean[-1].add_(lambda_lagrange, alpha=1.0 / self.rho)
            
        new_bias = torch.mean(in_mean, dim=self._get_bias_reduction_dims())
        self.b.copy_(new_bias)

        
    def update_z_last(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Solves the proximal update for the $z$ variable for the final layer, incorporating 
        spiking temporal penalties into the numerator and denominator.

        Computes z_L= numerator / denominator 
        where: 
            numerator = rho * (temporal_forward + delta * (z - forward)) if t<T-1
                      = rho * (temporal_forward + delta * (z - forward)) + (lamda*delta) if t=T-1
                      = rho * (temporal_forward) + (2y-lambda) if t=T
                       
            numerator = rho + (delta^2 * rho) if t<T
                      = rho + 2  if t=T

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
        """
        forward = self.spatial_forward(a_prev)
        temporal_forward = self.vectorized_forward(a_prev)
        shape = self.z[-1] 
            
        labels = self._broadcast_to_match(labels, shape)
        lambda_lagrange = self._broadcast_to_match(lambda_lagrange, shape)

        numerator = temporal_forward.clone().mul_(self.rho)
        z_minus_fwd = self.z.clone().sub_(forward)
        
        numerator[:-1].add_(z_minus_fwd[1:], alpha=self.rho * self.deltas)
        numerator[-2].add_(lambda_lagrange, alpha=self.deltas)
        numerator[-1].add_(labels, alpha=2.0).sub_(lambda_lagrange)
        
        denominator_main = self.rho * (self.deltas ** 2) + self.rho
        denominator_last = 2.0 + self.rho
        
        numerator[:-1].div_(denominator_main)
        numerator[-1].div_(denominator_last)

        self.z.copy_(numerator)   

    def update_az_interleaved(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list):
        """Orchestrates the interleaved updates of a and z over time using caching.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            time_steps (list): The list of sequence time steps to update.
        """
        cache = self._create_cache(next_layer, a_prev, lambda_lagrange)
        
        for t in time_steps:
            self.update_a_unrolled(t, cache, next_layer)
            self.update_z_unrolled(t, cache)
            
        # The final timestep is always updated last
        t_final = self.z.size(0) - 1
        self.update_a_unrolled(t_final, cache, next_layer)
        self.update_z_unrolled(t_final, cache)
         
    def update_a_unrolled(self, t: int, cache: TemporalCache, next_layer: nn.Module):
        """Manages the unrolled activation (a) update for a specific timestep.
        
        Utilizes precomputed terms from `TemporalCache`.

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices.
            next_layer (nn.Module): The subsequent layer in the network.
            
        """
        h_t = self.beta * self.h(self.z[t])
        temporal_penalty_numerator_t = 0.0
        if t < self.z.size(0) - 1:
            temporal_penalty_numerator_t = -self.thetas * self.rho * (self.z[t+1] - self.deltas * self.z[t] - cache.forward_pass[t+1])
        numerator = cache.adjoint[t] + h_t + temporal_penalty_numerator_t
        
        is_last = (t == self.z.size(0) - 1)
        denominator = cache.denominator_last if is_last else cache.denominator_main 
        new_a_t = next_layer.solve_activation_system_unrolled(numerator=numerator, denominator=denominator)
     
        self.a[t].copy_(torch.clamp(new_a_t, min=0.0, max=1.0))

    def solve_activation_system_unrolled(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """
        Executes the unrolled step using dense matrix multiplication.
        Reshaping is aligned with solve_linear_system in solvers.py.
        """
        if isinstance(denominator, dict):
            return solve_woodbury_system(
                W=denominator['W'], 
                B=numerator, 
                beta=denominator['beta'], 
                rho=denominator['rho'], 
                a_shape=numerator.shape
            )
        original_shape = numerator.shape 
        in_features = denominator.size(1)
        numerator_flat = numerator.reshape(-1, in_features)
        a_t_flat = torch.matmul(numerator_flat, denominator)   
        return a_t_flat.view(original_shape)

    def update_z_unrolled(self, t, cache, z_to_use=None):
        """Performs the unrolled pre-activation (z) update for hidden layers.

        Defines the proximal update based on two main terms:
         1. Temporal Forward pass  (Forward pass + delta*z_prev - theta*a_prev)
         2. z_next - Forward Pass

        Applies the proximal operator to these terms via the activation function's 
        specialized unrolled solver.

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices.
        """
        z_to_use = z_to_use if z_to_use is not None else self.z
        T = self.z.size(0)
        temporal_forward = cache.forward_pass[t].clone()
        if t > 0:
            temporal_forward.add_(z_to_use[t-1], alpha=self.deltas)
            temporal_forward.add_(self.a[t-1], alpha=-self.thetas)
        
        if t < T - 1:
            z_minus_forward = z_to_use[t+1].sub(cache.forward_pass[t+1])
        else:
            z_minus_forward = None
        new_z_t = self.h.activation_z_unrolled(temporal_forward=temporal_forward, z_minus_forward=z_minus_forward,a_t=self.a[t])  
        self.z[t].copy_(new_z_t)

    def update_z_last_unrolled(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list, jacobi:bool=False):
        """
        Urolled z update for the final layer (L).
        Computes z_L,t = numerator / denominator 
        where: 
            numerator = rho * (temporal_forward + delta * (z - forward)) if t<T-1
                      = rho * (temporal_forward + delta * (z - forward)) + (lamda*delta) if t=T-1
                      = rho * (temporal_forward) + (2y-lambda) if t=T
                       
            numerator = rho + (delta^2 * rho) if t<T
                      = rho + 2  if t=T
        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            time_steps (list): The list of sequence time steps to update.               
        """
        T = self.z.size(0)
        labels = self._broadcast_to_match(labels, self.z[-1]) 
        lambda_lagrange = self._broadcast_to_match(lambda_lagrange, self.z[-1]) 
        forward= self.spatial_forward(a_prev)
        denominator_main =  (self.rho * self.deltas ** 2) + self.rho
        z_to_use = self.z.clone() if jacobi else self.z
        buffer = torch.empty_like(self.z[0])
        for t in time_steps:
            if t == T - 1:
                continue
            buffer.copy_(forward[t])
            if t >= 1:
                buffer.add_(z_to_use[t-1], alpha=self.deltas)

            buffer.add_(z_to_use[t+1], alpha=self.deltas)
            buffer.add_(forward[t+1], alpha=-self.deltas)
            buffer.mul_(self.rho)
            if t == T - 2:
                buffer.add_(lambda_lagrange, alpha=self.deltas)
                
            buffer.div_(denominator_main)
            self.z[t].copy_(buffer)
        t = T - 1
        buffer.copy_(forward[t])
        del forward
        if t >= 1:
            buffer.add_(z_to_use[t-1], alpha=self.deltas)
    
        buffer.mul_(self.rho)
        buffer.add_(labels, alpha=2.0)
        buffer.sub_(lambda_lagrange)
        buffer.div_(2.0 + self.rho)
            
        self.z[t].copy_(buffer)
        
        if jacobi:
            del z_to_use
    
    def update_z_decoupled(self, a_prev: torch.Tensor, time_steps: list):
        """Decoupled causal sweep for the z update.

        Reuses the unrolled logic across all timesteps without redefining the math by 
        creating a lightweight mock cache.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            time_steps (list): The list of sequence time steps to update.
        """
        forward_pass = self.spatial_forward(a_prev)
        mock_cache = SimpleNamespace(forward_pass=forward_pass)
        for t in time_steps:
            self.update_z_unrolled(t, mock_cache)
        t_final = self.z.size(0) - 1
        self.update_z_unrolled(t_final, mock_cache)
        del forward_pass
        del mock_cache


