"""
ADMM Affine Layer Module

This module manages layers with trainable parameters (Weights and Biases) 
such as Linear and Conv2d layers. It contains the optimization logic 
to update these parameters using standard spatial mathematics.
"""

import torch
import torch.nn as nn
from .core import ADMM_Layer
from .initializers import get_initializer
from .solvers import solve_least_squares_weights, solve_linear_system

############################################################################################################
#Affine Layer Manager
############################################################################################################           

class ADMM_AffineLayer(ADMM_Layer):
    """Manages affine layers with trainable parameters like Linear and Conv2d.

    Handles the initialization of weights and biases and contains the optimization 
    solvers to update them using ADMM principles.
    """
    def __init__(self, h: nn.Module = None, bias: bool = False):
        super().__init__(h)
        self.bias = bias
        self.pinv = None
        self.W = None 
        self.b = None     

    def _init_weights_and_bias(self, weight_shape: tuple, bias_shape: tuple):
        """Initializes weights and biases based on the specified strategy.

        Args:
            weight_shape (tuple): The dimensions of the weight tensor.
            bias_shape (tuple): The dimensions of the bias tensor.
        """
        initializer = get_initializer(self.init)
        
        self.W = initializer.init_weights(weight_shape)
        self.b = initializer.init_bias(bias_shape) if self.bias else 0
    
    def _format_bias(self):
        """Reshapes the bias vector to match the dimensionality of the layer's output.

        Returns:
            torch.Tensor or int: The reshaped bias tensor, or 0 if bias is disabled.
        """
        if not self.bias:
            return 0
        target_shape = [1] * self.z.dim()
        target_shape[self.channel_dim] = self.b.size(0)
        return self.b.view(*target_shape)
    
    def _compute_covariances(self, Y: torch.Tensor, a_prev: torch.Tensor):
        """Computes the numerator and denominator for the weight update.

        Calculates the numerator (Y^T @ P) and denominator (P^T @ P) required for 
        the weight update step.

        Args:
            Y (torch.Tensor): The base target tensor.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            tuple:
                - torch.Tensor: The computed numerator matrix.
                - torch.Tensor: The computed denominator matrix.
                - torch.Tensor: The computed $P$ matrix.
        """
        P = self._compute_P(a_prev)
        Y_flat = Y.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])
        
        numerator = Y_flat.t() @ P
        denominator= P.t() @ P
        
        return numerator, denominator, P
    
    def _get_Y(self) -> torch.Tensor:
        """Returns Y (z - bias) for parameter updates.

        Returns:
            torch.Tensor: The tensor Y, calculated as z - bias.
        """
        return self.z - self._format_bias()

    def _apply_lagrange_to_weights(self, lambda_lagrange: torch.Tensor, a_prev: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Calculates the Lagrange penalty for the weight numerator.

        Args:
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            a_prev (torch.Tensor): The previous layer's activations.
            P (torch.Tensor): The computed P matrix from the covariance step.

        Returns:
            torch.Tensor: The computed penalty term to be added to the weight numerator.
        """
        lamb = self._broadcast_to_match(lambda_lagrange, self.z)
        lamb = lamb.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])
        return (lamb.t() / self.beta) @ P

    def _apply_lagrange_to_Y(self, Y: torch.Tensor, lambda_lagrange: torch.Tensor) -> torch.Tensor:
        """Applies the Lagrange multiplier to the activation or bias target.

        Args:
            Y (torch.Tensor): The target tensor to apply the penalty to.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.

        Returns:
            torch.Tensor: The penalized target tensor.
        """
        lam_spatial = self._broadcast_to_match(lambda_lagrange, Y)
        return Y + (lam_spatial / self.beta)
    
    def _get_expanded_weights(self, next_layer: nn.Module):
        """Delegates weight expansion to the next layer's pooling operator.

        If no spatial pooling is used, it simply flattens the weights.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.

        Returns:
            torch.Tensor: The expanded or flattened weight matrix.
        """
        if hasattr(next_layer, 'pool_op') and self.a.dim() > 3:
            return next_layer.pool_op.expand_weights(next_layer.W, original_a_shape=self.a.shape)
            
        return next_layer.W.view(next_layer.W.size(0), -1)  

    def update_weights(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None, cache_pinv: bool = False):
        """Solves the regularized least-squares problem for the Weight matrix W.

        Mathematical formulation: 
        W= numerator @ denominator^{-1}, where
        numerator = (Y^T @ P) + Lagrange_penalty
        denominator = P^T @ P
        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
            cache_pinv (bool, optional): Whether to cache the pseudoinverse. Defaults to False.
        """
        Y = self._get_Y()            
        numerator, denominator, P = self._compute_covariances(Y, a_prev)
        if lambda_lagrange is not None:
            numerator += self._apply_lagrange_to_weights(lambda_lagrange, a_prev, P)

        new_W, self.pinv = solve_least_squares_weights(
            numerator, 
            denominator, 
            cached_pinv=self.pinv if cache_pinv else None
        )
        self.W.data.copy_(new_W.reshape(self.W.shape))
          
    def update_bias(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Averages the residual errors to update the bias vector.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        target = self.z - self.spatial_forward(a_prev, use_bias=False)
        
        if lambda_lagrange is not None:
            lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z)
            #target = target + (lam_spatial / (2 * self.beta))
            target = target + (lam_spatial / (self.beta))
                
        new_bias = torch.mean(target, dim=self._get_bias_reduction_dims())
        self.b.data.copy_(new_bias)

    
    def update_z(self, a_prev: torch.Tensor, time_steps=None):
        """Applies the z update using the activation function's operator.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        res = self.spatial_forward(a_prev)
        new_z = self.h.activation_z_update(a=self.a, res=res, z=self.z, time_steps=time_steps)
        self.z.data.copy_(new_z)  

    def update_a(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Universal Exact Solver for the a update.

        Solves the linear system: 
        denominator * a = numerator, where
        numerator = gamma * h(z) + beta * adjoint(Y)
        denominator = gamma * I + beta * W^T * W

        Args:
            target (torch.Tensor): The target tensor from the next layer.
            next_layer (nn.Module): The subsequent layer in the network.
            temp_num (float or torch.Tensor, optional): Temporal penalty for the numerator. Defaults to 0.0.
            temp_den (float or torch.Tensor, optional): Temporal penalty for the denominator. Defaults to 0.0.

        Returns:
            torch.Tensor: The exact updated activations.
        """
        inside_adjoint= next_layer._get_Y()
        if lambda_lagrange is not None:
             inside_adjoint = next_layer._apply_lagrange_to_Y(inside_adjoint, lambda_lagrange / 2.0)
        h_z = self.h(self.z)
        adjoint = next_layer.adjoint_operator(inside_adjoint, original_input_shape=self.a.shape)
        numerator = self.gamma * h_z + next_layer.beta * adjoint 
        
        W = self._get_expanded_weights(next_layer)  
        I = torch.eye(W.size(1), device=next_layer.W.device, dtype=W.dtype)
        WtW = torch.matmul(W.t(), W) 
        denominator = self.gamma * I + next_layer.beta * WtW
        
        new_a = solve_linear_system(denominator, numerator, self.a.shape, W.size(1))
        self.a.data.copy_(new_a)



         