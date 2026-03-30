"""
ADMM Base Layer Definitions

This module provides the foundational classes for all ADMM layers. 

INDEX:
- ADMM_Layer: The absolute base class managing states, setups, and generic passes.
- ADMM_AffineLayer: Manages updates of affine layers.
- ADMM_Spiking: A mixin handling exact unrolled solvers for the temporal dimension.

IMPORTANT:
- Beta should be discussed.
- Bias need to be reworked
"""

import torch
import torch.nn as nn
from .activations import ADMM_Identity
from .initializers import get_initializer 



####################################################################################################
# Base Layer Interface
####################################################################################################           

class ADMM_Layer(nn.Module):
    """
    The core Base Layer Interface for all ADMM modules.
    Manages the fundamental auxiliary variables, handles the configuration setup, 
    and dictates the standard vs. vectorized forward passes.
    """
    def __init__(self, h: nn.Module = None):
        super().__init__()
        self.device = None
        self.beta = None
        self.gamma = None
        self.deltas = None
        self.thetas = None
        
        self.z = None
        self.a = None
        
        self.spiking = False 
        self.train_method = None
        
        self.h = h if h is not None else ADMM_Identity()
    
    def _broadcast_to_match(self, tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """Helper method to safely align tensor dimensions for element-wise operations."""
        if tensor.dim() < target_tensor.dim():
            missing_dims = target_tensor.dim() - tensor.dim()
            return tensor.view(*tensor.shape, *([1] * missing_dims))
        return tensor 


    def setup(self, config: dict, is_last_layer: bool = False):
        """
        Receives global ADMM hyperparameters from the manager and 
        cascades them to the activation function.
        """
        for key, val in config.items():
            setattr(self, key, val)
            
        if is_last_layer and hasattr(self, 'use_reset'):
            self.use_reset = False
            
        if hasattr(self, 'h') and hasattr(self.h, 'setup'):
            self.h.setup(config)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        """
        STANDARD SEQUENTIAL PASS (Initialization / Inference)
        
        - For Static Networks: Simply returns the spatial transformation (y = Wx).
        - For Spiking Networks (SNNs): Simulates the Leaky Integrate-and-Fire (LIF) 
          mechanics step-by-step over the time dimension (T).
        """
        y = self.spatial_forward(x)     
        
        if self.spiking:
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
        
        return y
   
    def vectorized_forward(self, a_prev: torch.Tensor) -> torch.Tensor:
        """
        DECOUPLED ADMM PASS (Optimization / Constraint Evaluation)
        
        Evaluates the linear constraint across all dimensions simultaneously.
        In SNNs, this natively includes the temporal leakage and reset dependencies.
        """
        z = self.spatial_forward(a_prev) 
        if self.spiking: 
            z += self._compute_temporal_dependencies()
        return z
    
    def update_z_last(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps=None):
        """
        Solves the proximal update for the 'z' variable for the LAST layer.
        
        z =  beta * (forward(a_prev)) + (labels - (lamb / 2)) / (1 + beta)
        Spiking Case: Incorporates temporal penalties into the numerator and denominator.
        """
       
        forward = self.vectorized_forward(a_prev) #This implicitly computes temporal dependencies if spiking=True
        shape = self.z[-1] if self.spiking else forward
        

        labels = self._broadcast_to_match(labels, shape)
        lamb = self._broadcast_to_match(lambda_lagrange, shape)
        
        if self.spiking:
            num, den = self._get_temporal_z_penalties(a_prev, labels, lamb)
        else:
            num = labels - (lamb / 2.0)
            den = 1.0
            
        numerator = (self.beta * forward) + num
        denominator = self.beta + den
        
        self.z.data.copy_(numerator / denominator)       

 
############################################################################################################
#Affine Layer Manager
############################################################################################################           


class ADMM_AffineLayer(ADMM_Layer):
    """
    Manages affine layers with trainable parameters (Weights and Biases) like Linear and Conv2d.
    Handles the initialization of these parameters and contains the optimization solvers to update them.
    """
    def __init__(self, h: nn.Module = None, bias: bool = False):
        super().__init__(h)
        self.bias = bias
        self.pinv = None
        self.W = None 
        self.b = None     

    def _init_weights_and_bias(self, weight_shape: tuple, bias_shape: tuple):
        """Initializes weights based on the specified strategy."""
        initializer = get_initializer(self.init)
        
        self.W = initializer.init_weights(weight_shape)
        self.b = initializer.init_bias(bias_shape) if self.bias else 0
    
    def _format_bias(self):
        """Reshapes the bias vector to match the dimensionality of the layer's output."""
        if not self.bias:
            return 0
        target_shape = [1] * self.z.dim()
        target_shape[self.channel_dim] = self.b.size(0)
        return self.b.view(*target_shape)
    
    def _compute_covariances(self, Y: torch.Tensor, a_prev: torch.Tensor):
        """
        Computes the numerator (Y^T @ P) and denominator (P^T @ P) for the least squares update.
        """
        P = self._compute_P(a_prev)
        Y_flat = Y.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])
        
        numerator = Y_flat.t() @ P
        denominator= P.t() @ P
        
        return numerator, denominator, P
    
    def update_weights(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None, cache_pinv: bool = False):
        """
        Solves the regularized least-squares problem for the Weight matrix W.
        W_new = ((P^T @ P)^-1 @ (P^T @ Y))^T = (Y^T @ P) @ (P^T @ P)^-1
        """
        Y = self.z - self._format_bias()
        if self.spiking:
            Y = Y - self._compute_temporal_dependencies()
            
        numerator, denominator, P = self._compute_covariances(Y, a_prev)

        if lambda_lagrange is not None:
            z = self.z[-1] if self.spiking else self.z
            P = self._compute_P(a_prev[-1].unsqueeze(0)) if self.spiking else P
            lamb = self._broadcast_to_match(lambda_lagrange, z)
            lamb = lamb.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])
            numerator += (lamb.t() / self.beta) @ P

        if cache_pinv:
            if getattr(self, 'pinv', None) is None: 
                    self.pinv = torch.linalg.pinv(denominator)
            pinv= self.pinv
        else:
            pinv= torch.linalg.pinv(denominator)
            
        new_W = numerator @ pinv
        self.W.data.copy_(new_W.reshape(self.W.shape))
        
        
    def update_bias(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Averages the residual errors to update the bias vector."""
        target = self.z - self.spatial_forward(a_prev, use_bias=False)
        
        if self.spiking:
            target = target - self._compute_temporal_dependencies()
            if lambda_lagrange is not None:
                lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z[-1])
                #target[-1] = target[-1] + (lam_spatial / (2 * self.beta))
                target[-1] = target[-1] + (lam_spatial / (self.beta))
        else:
            if lambda_lagrange is not None:
                lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z)
                #target = target + (lam_spatial / (2 * self.beta))
                target = target + (lam_spatial / (self.beta))
                
        new_bias = torch.mean(target, dim=self._get_bias_reduction_dims())
        self.b.data.copy_(new_bias)

    
    #P
    def update_z(self, a_prev: torch.Tensor, time_steps=None):
        """Applies the z update using the activation function's operator."""
        res = self.spatial_forward(a_prev)
        new_z = self.h.activation_z_update(a=self.a, res=res, z=self.z)
        self.z.data.copy_(new_z)  
      
    def _get_expanded_weights(self, next_layer: nn.Module):
        """
        Delegates weight expansion to the next layer's pooling operator.
        If no spatial pooling is used, it simply flattens the weights.
        """
        if hasattr(next_layer, 'pool_op') and self.a.dim() > 3:
            return next_layer.pool_op.expand_weights(next_layer.W, original_a_shape=self.a.shape)
            
        return next_layer.W.view(next_layer.W.size(0), -1)  
     
    #THE TWO FOLLOWING FUNCTIONS SHOULD BE REWORKED WHEN VECTORIZED MODE IS CREATED TO BE CLEANER.
    def _compute_exact_a(self, target: torch.Tensor, next_layer: nn.Module, temp_num=0.0, temp_den=0.0):
        """
        Universal Exact Solver for the Activation (a) update.
        Solves the linear system: (gamma * I + beta * W^T W) * a = gamma * h(z) + beta * adjoint + temporal_penalties (if spiking)
        """

        W= self._get_expanded_weights(next_layer)  
        in_features = W.size(1)
        I = torch.eye(in_features, device=next_layer.W.device, dtype=W.dtype)
        WtW = torch.matmul(W.t(), W) 
        
        h_z = self.h(self.z)
        adjoint = next_layer.adjoint_operator(target, original_input_shape=self.a.shape)
        
        numerator = self.gamma * h_z + next_layer.beta * adjoint + temp_num
        denominator = self.gamma * I + next_layer.beta * WtW
        
        def solve_system(lhs_matrix, rhs_slice, out_shape):
            rhs_flat = rhs_slice.reshape(-1, in_features)
            solved_flat = torch.linalg.solve(lhs_matrix, rhs_flat.t()).t()
            return solved_flat.view(out_shape)

        if not getattr(next_layer, 'spiking', False):
            return solve_system(denominator, numerator, self.a.shape)
            
        else:
            T = self.z.size(0)
            a_last = solve_system(denominator, numerator[-1], (1, *self.a.shape[1:]))
            
            if T > 1:
                td_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
                LHS_main = denominator + (td_scalar * I)
                
                a_main = solve_system(LHS_main, numerator[:-1], (T - 1, *self.a.shape[1:]))
            else:
                a_main = torch.empty((0, *self.a.shape[1:]), device=self.z.device)
                
            return torch.cat([a_main, a_last], dim=0)

     
    def update_a(self, next_layer: nn.Module, lambda_lagrange: torch.Tensor = None):
        """
        Updates the activation 'a'.
        a = gamma * h(z) + beta * adjoint(next_layer.z - next_layer._format_bias() - temporal_penalties + (lambda/2*beta)) / gamma + beta * W^T W
        
        Spiking Case: Incorporates temporal penalties and handles the lambda term based on the time step.
        """
        numerator = next_layer.z - next_layer._format_bias()
        
        if getattr(next_layer, 'spiking', False):
            numerator = numerator - next_layer._compute_temporal_dependencies()
            if lambda_lagrange is not None:
                lam_sp = next_layer._broadcast_to_match(lambda_lagrange, next_layer.z[-1])
                numerator[-1] = numerator[-1] + (lam_sp / (2 * next_layer.beta))
        else:
            if lambda_lagrange is not None:
                lam_sp = next_layer._broadcast_to_match(lambda_lagrange, numerator)
                numerator = numerator + (lam_sp / (2 * next_layer.beta))
                
        new_a = self._compute_exact_a(numerator, next_layer)
        new_a = torch.clamp(new_a, min=0.0, max=1) if self.spiking else new_a
        self.a.data.copy_(new_a)

############################################################################################################
#Spiking mixing
############################################################################################################ 

class ADMM_Spiking:
    """
    Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs).
    """

    def _fold_time(self, x: torch.Tensor):
        """Folds the Time and Batch dimensions together for spatial operations."""
        if x.dim() in [3, 5]: 
            return x.reshape(x.size(0) * x.size(1), *x.shape[2:]), x.shape[:2]
        return x, None

    def _unfold_time(self, x_flat: torch.Tensor, tb_shape: tuple):
        """Unfolds the Time and Batch dimensions back out after spatial operations."""
        if tb_shape is None: 
            return x_flat
        return x_flat.reshape(tb_shape[0], tb_shape[1], *x_flat.shape[1:])
    
    
    def _compute_temporal_dependencies(self):     
        """Computes the physical voltage leakage and threshold reset over time."""
        z = torch.zeros_like(self.z)
        z[1:] = self.deltas * self.z[:-1]
        if getattr(self, 'use_reset', False):
            z[1:] -= self.thetas * self.a[:-1]
        return z
           
    def _get_temporal_z_penalties(self, a_prev: torch.Tensor, labels: torch.Tensor, lamb: torch.Tensor):
        """
        Calculates the temporal penalties for the z update.
        num = beta * deltas * (z - forward)_t+1 + (lamb * deltas) / 2 (second last timestep) and labels - (lamb / 2) (last timestep)
        den = beta * (deltas^2) + 1 (last timestep)
        """
        num, den = torch.zeros_like(self.z), torch.zeros_like(self.z)
        forward = self.spatial_forward(a_prev)
       
        num[:-1] = self.beta * self.deltas * (self.z - forward)[1:]
        den[:-1] = self.beta * (self.deltas ** 2) 
        
        num[-2] += (lamb * self.deltas) / 2
        num[-1] = labels - (lamb / 2)
        den[-1] = 1.0
        return num, den
    
    def _get_temporal_a_penalties(self, cache: dict):
        """
        Calculates the temporal penalties for the a update.
        num = -beta * thetas * (z - deltas * z_t-1 - forward)(main timesteps) and 0 (last timestep)
        den = beta * (thetas^2) (main timesteps) and 0 (last timestep)
        """
        if not getattr(self, 'use_reset', False): 
            return 0.0, 0.0
        
        num, den = torch.zeros_like(self.z), torch.zeros_like(self.z)        
        num[:-1] = - self.thetas * self.beta * (self.z[1:] - self.deltas * self.z[:-1] - cache["FORWARD"][1:])
        den[:-1] =  self.beta * (self.thetas ** 2)
        return num, den
    
    def _create_cache(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Precomputes and distributes operations to accelerate the unrolled loop."""
        T, batch_size = self.z.shape[:2]
        cache = {}

        cache["FORWARD"] = self.spatial_forward(a_prev)
        target_z = next_layer.z - next_layer._format_bias()
        
        if getattr(next_layer, 'spiking', False):
            z_shifted = torch.zeros_like(next_layer.z)
            z_shifted[1:] = next_layer.z[:-1]
            target_z = target_z - next_layer.deltas * z_shifted
        
        if lambda_lagrange is not None:
            target_z[-1] = target_z[-1] + (lambda_lagrange / next_layer.beta)

        projected_target = next_layer.adjoint_operator(target_z, original_input_shape=self.a.shape)
        base_RHS = next_layer.beta * projected_target

        temp_num, _ = self._get_temporal_a_penalties(cache)
        TERM2 = base_RHS + temp_num if getattr(self, 'use_reset', False) and isinstance(temp_num, torch.Tensor) else base_RHS          
               
        W_flat = self._get_expanded_weights(next_layer)
        
        I = torch.eye(W_flat.size(1), device=next_layer.W.device, dtype=W_flat.dtype)
        WtW = torch.matmul(W_flat.t(), W_flat) 
                
        LHS_last = self.gamma * I + next_layer.beta * WtW
        LHS_main = LHS_last + self.beta * (self.thetas ** 2) * I if getattr(self, 'use_reset', False) else LHS_last
            
        TERM1_a_main = torch.linalg.inv(LHS_main).t()          
        TERM1_a_last = torch.linalg.inv(LHS_last).t()
            
        cache["TERM1_a_main"] = TERM1_a_main
        cache["TERM1_a_last"] = TERM1_a_last

        TERM2_flat = TERM2.view(T * batch_size, -1)
        A_main_cache_flat = torch.matmul(TERM2_flat, TERM1_a_main)
         
        cache["TERM1*TERM2_a_main"] = A_main_cache_flat.view(T, batch_size, -1)
        cache["TERM1*TERM2_a_last"] = torch.matmul(TERM2[-1].view(batch_size, -1), TERM1_a_last)
            
        return cache
    
            
    def _update_a_unrolled(self, t, cache):
        """
        a update: (beta*W^T_l+1*W_l+1+gamma*I)^-1 (beta*W^T_l+1*v_l+1 + gamma*h)
        We call this TERM1 * (TERM2 + TERM3)= TERM1*TERM2 + TERM1*TERM3= A+ TERM1*TERM3
        Note that TERM1*TERM2 and TERM1 can be cached since does not depend on a_l nor z_l
        We need to compute just TERM3
        """ 
        h = self.h(self.z[t])

        TERM3 = self.gamma * h.view(h.size(0), -1) 
            
        if t < self.z.size(0) - 1:
            a_t_flat = torch.matmul(TERM3, cache["TERM1_a_main"]) + cache["TERM1*TERM2_a_main"][t]
        else:
            a_t_flat = torch.matmul(TERM3, cache["TERM1_a_last"]) + cache["TERM1*TERM2_a_last"]
            
        self.a[t].data.copy_(torch.clamp(a_t_flat.view_as(self.a[t]), min=0.0, max=1.0))
    
        
    def _update_z_unrolled(self, t, cache):
        """
         z update for hidden layers:
         We define the proximal update based on two main terms:
         TERM_1: Spiking Forward pass  (Forward pass + delta*z_prev - theta*a_prev)
         TERM_2: z_next - Forward Pass
         Then we apply the proximal operator to these terms.
        """
        T = self.z.size(0) #Time = z size
        TERM_1 = cache["FORWARD"][t]
        if t > 0:
            TERM_1 = TERM_1 + (self.deltas * self.z[t-1] - self.thetas * self.a[t-1])
        
        TERM_2 = self.z[t+1] - cache["FORWARD"][t+1] if t < T - 1 else None
        
        new_z_t = self.h.activation_z_unrolled(TERM_1, TERM_2, self.a[t])  
        self.z[t].data.copy_(new_z_t)

    def _update_z_last_unrolled(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list):
        """
        z update for the LAST layer (L):
        TERM_LAMBDA: Lagrange multiplier (only active at T-2)
        
        For t < T - 1: z_t = (TERM_1 + delta*TERM_2 + delta*TERM_LAMBDA/beta) / (1 + delta^2)
        For t = T - 1: z_T = (beta*TERM_3 + 2*labels - TERM_LAMBDA) / (2 + beta)
        
        We precompute FORWARD_PASS for all timesteps to avoid redundant calculations.
        """
        T = self.z.size(0) #Time = z size
        labels_sp = self._broadcast_to_match(labels, self.z[-1]) # reshaped labels
        lam_sp = self._broadcast_to_match(lambda_lagrange, self.z[-1]) #reshaped lambda
        
        FORWARD_PASS = self.spatial_forward(a_prev) - self._format_bias()
        
        for t in time_steps:
            TERM_1 = FORWARD_PASS[t]
            if t >= 1:
                TERM_1 = TERM_1 + self.deltas * self.z[t-1]
            
            TERM_2 = self.z[t+1] - FORWARD_PASS[t+1]
            TERM_LAMBDA = lam_sp if t == T - 2 else torch.zeros_like(lam_sp)
            z_t = (TERM_1 + self.deltas * TERM_2 + self.deltas * TERM_LAMBDA / self.beta) / (1 + self.deltas ** 2)
            self.z[t].data.copy_(z_t)

        t = T - 1
        TERM_3 = FORWARD_PASS[t]
        if t >= 1:
            TERM_3 = TERM_3 + self.deltas * self.z[t-1]
    
        z_T = (self.beta * TERM_3 + 2 * labels_sp - lam_sp) / (2 + self.beta)
        self.z[t].data.copy_(z_T)


    def update_z_last(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps=None):
        """Manager that reroutes the target z_last update depending on the chosen training method."""
        if getattr(self, 'train_method', 'vectorized').startswith('unrolled'):
            self._update_z_last_unrolled(a_prev, labels, lambda_lagrange, time_steps)
        else:
            super().update_z_last(a_prev, labels, lambda_lagrange)
            
    def update_az_interleaved(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list):
        """Orchestrates the interleaved updates of 'a' and 'z' over time with caching."""
        cache = self._create_cache(next_layer, a_prev, lambda_lagrange)
        
        for t in time_steps:
            self._update_a_unrolled(t, cache)
            self._update_z_unrolled(t, cache)
            
        # The final timestep is always updated last
        t_final = self.z.size(0) - 1
        self._update_a_unrolled(t_final, cache)
        self._update_z_unrolled(t_final, cache)
        