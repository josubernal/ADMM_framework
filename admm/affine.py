
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
        
        self.W = initializer.init_weights(weight_shape, device=self.device)
        self.b = initializer.init_bias(bias_shape, device=self.device) if self.bias else 0
    
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
        
        return numerator, denominator
    
    
    def _get_v(self, lambda_lagrange: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Returns v for parameter updates.
        Formula:
        v = z-b if l<L
        v = z-b+(lambda_lagrange/rho)

        Returns:
            torch.Tensor: The tensor v, calculated as z - bias.
        """
        v = self.z.clone()
        bias_formatted = self._format_bias()
        if isinstance(bias_formatted, torch.Tensor):
            v.sub_(bias_formatted)
        if lambda_lagrange is not None:
            lambda_lagrange = self._broadcast_to_match(lambda_lagrange, v)
            v.add_(lambda_lagrange, alpha=1.0 / self.rho)
        return v

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
    
    def _get_a_numerator(self, beta_current, a_shape, h_z:torch.Tensor, lambda_lagrange: torch.Tensor= None):
        inside_adjoint = self._get_v(lambda_lagrange=lambda_lagrange)
        adjoint = self.adjoint_operator(inside_adjoint, original_input_shape=a_shape)
        numerator = h_z.clone().mul_(beta_current)
        numerator.add_(adjoint, alpha=self.rho)
        return numerator
        
    def _get_WtW(self, a_shape: tuple):
        """Template method to compute W^T W."""
        W = self._get_expanded_weights(a_shape=a_shape)   
        in_features = W.size(1)
        WtW = torch.matmul(W.t(), W)
        return WtW, in_features
    
    def _get_a_denominator(self, beta_current, a_shape):
        """Computes the denominator matrix for the activation (a) update step.

        Formula:
        denominator = beta * I + rho * W^T * W
        
        Args:
            beta_current (float): The penalty parameter beta for the activation update.
            a_shape (tuple): The shape of the activation tensor, used to determine 
                how weights should be expanded or flattened.

        Returns:
            tuple:
                - torch.Tensor: The computed denominator matrix (LHS of the system).
                - int: The number of input features.
        """
        W= self._get_expanded_weights(a_shape=a_shape)
        out_features, in_features = W.shape
        if getattr(self, 'W', None) is not None and self.W.dim() == 2 and in_features > out_features:
            return self._get_woodbury_params(beta_current, a_shape)
        
        denominator, in_features = self._get_WtW(a_shape)
        denominator.mul_(self.rho)
        denominator.diagonal().add_(beta_current)
        return  denominator, denominator, in_features
    
    def _get_woodbury_params(self, beta_current, a_shape):
         W = self._get_expanded_weights(a_shape=a_shape) 
         _, in_features = W.shape
         dic= {'W':W, 'beta':beta_current, 'rho': self.rho}
         return dic,dic, in_features
          
    def update_weights(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None, cache_pinv: bool = False):
        """Maneges th update for the layer's weights by solving a regularized least-squares problem.

        Args:
            a_prev (torch.Tensor): The activations from the previous layer. For 
                standard layers, this is a 4D tensor [B, C, H, W]; for spiking 
                layers, it is a 5D tensor [T, B, C, H, W].
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier used 
                to enforce the ADMM consensus constraint. If provided, it is 
                incorporated into the target 'v' to penalize constraint violations. 
                Defaults to None.
            cache_pinv (bool, optional): If True, reuses the previously computed 
                pseudoinverse (self.pinv) to solve the system, significantly 
                accelerating updates.
                Defaults to False.

        Returns:
            None: The weights (self.W) are updated in-place.
        """
        v = self._get_v(lambda_lagrange=lambda_lagrange) 
        numerator, denominator = self._compute_covariances(v, a_prev)
        new_W, temp_pinv = solve_least_squares_weights(
            numerator, 
            denominator, 
            use_cholesky=self.use_cholesky,
            cached_pinv=self.pinv if cache_pinv else None   
        )
        self.W.copy_(new_W.detach().reshape(self.W.shape))
        if cache_pinv and temp_pinv is not None:
            self.pinv = temp_pinv.detach()
        else:
            self.pinv = None
          
    def update_bias(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Averages the residual errors to update the bias vector.
        
        Formula:
        b = mean(z-A(a_prev))
        
        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        in_mean = self.spatial_forward(a_prev, use_bias=False)
        in_mean.neg_().add_(self.z)
        
        if lambda_lagrange is not None:
            lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z)
            in_mean.add_(lam_spatial, alpha=1.0 / self.rho)
                
        new_bias = torch.mean(in_mean, dim=self._get_bias_reduction_dims())
        self.b.copy_(new_bias)

    
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
        """Manages the activation (a) update for standard spatial layers.
        
        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        numerator = next_layer._get_a_numerator(beta_current=self.beta, a_shape=self.a.shape, h_z=self.h(self.z), lambda_lagrange=lambda_lagrange)
        denominator_main, denominator_last , in_features = next_layer._get_a_denominator(beta_current=self.beta, a_shape=self.a.shape)
        new_a = next_layer.solve_activation_system(
            numerator=numerator,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            a_shape=self.a.shape,
            in_features=in_features
        )
        
        self.a.copy_(new_a)
        
    def solve_activation_system(self, numerator: torch.Tensor, denominator_main:torch.Tensor, denominator_last:torch.Tensor, a_shape: tuple, in_features:int) -> torch.Tensor:
        """Universal Solver for the a update.
        Args:
            numerator (torch.Tensor): The precomputed numerator tensor.
            a_shape (tuple): The shape of the activation tensor.
            beta_current (float): The penalty parameter beta.

        Returns:
            torch.Tensor: The exact updated activations.
        """
        return solve_linear_system(denominator_last, numerator, a_shape, in_features)
