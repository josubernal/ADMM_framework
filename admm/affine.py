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

from .pooling import ADMM_Flatten

############################################################################################################
#Affine Layer Manager
############################################################################################################           

class ADMM_AffineLayer(ADMM_Layer):
    """Manages affine layers with trainable parameters like Linear and Conv2d.

    Handles the initialization of weights and biases and contains the optimization 
    solvers to update them using ADMM principles.
    """
    def __init__(self, h: nn.Module = None, bias: bool = False, pool_op= None):
        super().__init__(h)
        self.bias = bias
        self.pinv = None
        self.W = None 
        self.b = None  
        self.pool_op = pool_op if pool_op is not None else ADMM_Flatten()   

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
        return (lamb.t() / self.rho) @ P

    def _apply_lagrange_to_Y(self, Y: torch.Tensor, lambda_lagrange: torch.Tensor) -> torch.Tensor:
        """Applies the Lagrange multiplier to the activation or bias target.

        Args:
            Y (torch.Tensor): The target tensor to apply the penalty to.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.

        Returns:
            torch.Tensor: The penalized target tensor.
        """
        lam_spatial = self._broadcast_to_match(lambda_lagrange, Y)
        return Y + (lam_spatial / self.rho)
    
    def _get_expanded_weights(self, a_shape):
        """Delegates weight expansion to the next layer's pooling operator.

        If no spatial pooling is used, it simply flattens the weights.

        Args:
            next_layer (nn.Module): The subsequent layer in the network.

        Returns:
            torch.Tensor: The expanded or flattened weight matrix.
        """
        if self.pool_op and len(a_shape) > 3:
            return self.pool_op.expand_weights(self.W, original_a_shape=a_shape)
            
        return self.W.view(self.W.size(0), -1)
    
        
    def _get_a_denominator(self, beta_current, a_shape, temp_den=0.0, unrolled=False):
        """Computes the denominator matrix for the activation (a) update step.

        Formula:
        denomitator = beta I + rho W^TW
        Args:
            beta_current (float): The penalty parameter beta for the activation update.
            a_shape (tuple): The a shape of the activation tensor, 
                used to determine how weights should be expanded or flattened.
            temp_den (float or torch.Tensor, optional): Temporal penalty term 
                used in Spiking Neural Networks to account for leakage/reset 
                dependencies. Defaults to 0.0.
            unrolled (bool, optional): If True, indicates the call is from an 
                unrolled temporal solver. In the base implementation, this is 
                a placeholder; spiking overrides use this to return inverted 
                matrices. Defaults to False.

        Returns:
            tuple:
                - torch.Tensor: The computed denominator matrix (LHS of the system).
                - int: The number of input features (dimensionality of the expanded 
                  weight space).
        """
        W = self._get_expanded_weights(a_shape=a_shape)   
        in_features = W.size(1)
        I = torch.eye(in_features, device=self.W.device, dtype=W.dtype)
        WtW = torch.matmul(W.t(), W) 
        denominator =  beta_current * I + self.rho * WtW
        return  denominator, in_features
    
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
            #target = target + (lam_spatial / (2 * self.rho))
            target = target + (lam_spatial / (self.rho))
                
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
        """Universal Solver for the a update. 

        Solves the linear system: 
        denominator * a = numerator, where
        numerator = beta * h(z) + rho * adjoint(Y)
        denominator = beta * I + rho * W^T * W

        Args:
            target (torch.Tensor): The target tensor from the next layer.
            next_layer (nn.Module): The subsequent layer in the network.
            temp_num (float or torch.Tensor, optional): Temporal penalty for the numerator. Defaults to 0.0.
            temp_den (float or torch.Tensor, optional): Temporal penalty for the denominator. Defaults to 0.0.

        Returns:
            torch.Tensor: The exact updated activations.
        """
        inside_adjoint = next_layer._get_Y()
        if lambda_lagrange is not None:
            inside_adjoint = next_layer._apply_lagrange_to_Y(inside_adjoint, lambda_lagrange / 2.0)
        
        h_z = self.h(self.z)
        adjoint = next_layer.adjoint_operator(inside_adjoint, original_input_shape=self.a.shape)
        numerator = self.beta * h_z + next_layer.rho * adjoint 
        
        new_a = next_layer.solve_activation_system(
            numerator=numerator,
            a_shape=self.a.shape,
            beta_current=self.beta
        )
        
        self.a.data.copy_(new_a)
        
    def solve_activation_system(self, numerator: torch.Tensor, a_shape: tuple, beta_current: float, temp_den=0.0) -> torch.Tensor:
        """Universal Solver for the a update.

        Solves the linear system: 
        denominator * a = numerator, where
        numerator = beta * h(z) + rho * adjoint(Y)
        denominator = beta * I + rho * W^T * W

        Args:
            target (torch.Tensor): The target tensor from the next layer.
            next_layer (nn.Module): The subsequent layer in the network.
            temp_num (float or torch.Tensor, optional): Temporal penalty for the numerator. Defaults to 0.0.
            temp_den (float or torch.Tensor, optional): Temporal penalty for the denominator. Defaults to 0.0.

        Returns:
            torch.Tensor: The exact updated activations.
        """
        denominator , in_features = self._get_a_denominator( beta_current, a_shape)
        return solve_linear_system(denominator, numerator, a_shape, in_features)
    