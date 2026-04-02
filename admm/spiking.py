"""
ADMM Spiking Mixin

This module provides temporal modeling and unrolled solvers for Spiking 
Neural Networks (SNNs). It is designed to be used as a Mixin, overriding 
standard spatial hooks with temporal dependencies (leakage, reset).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from .solvers import solve_spiking_system
from types import SimpleNamespace

@dataclass
class TemporalCache:
    """
    Holds precomputed tensors for the unrolled SNN temporal loops.
    Replaces the opaque dictionary to provide type safety and IDE auto-completion.
    """
    forward_pass: torch.Tensor
    term1_a_main: torch.Tensor
    term1_a_last: torch.Tensor
    term1_term2_a_main: torch.Tensor
    term1_term2_a_last: torch.Tensor

############################################################################################################
#Spiking mixing
############################################################################################################ 

class ADMM_Spiking:
    """
    Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs).
    """
    #================= Temporal Modeling Hooks =================#
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
    
    def _compute_temporal_dependencies(self, include_reset: bool = True) -> torch.Tensor:     
        """Computes the physical voltage leakage and threshold reset over time."""
        z = torch.zeros_like(self.z)
        z[1:] = self.deltas * self.z[:-1]
        if getattr(self, 'use_reset', False) and include_reset:
            z[1:] -= self.thetas * self.a[:-1]
        return z
    
    def _get_temporal_a_penalties(self, forward_pass: torch.Tensor):
        """
        Calculates the temporal penalties for the a update.
        num = -beta * thetas * (z - deltas * z_t-1 - forward)(main timesteps) and 0 (last timestep)
        den = beta * (thetas^2) (main timesteps) and 0 (last timestep)
        """
        if not getattr(self, 'use_reset', False): 
            return 0.0, 0.0
        
        num, den = torch.zeros_like(self.z), torch.zeros_like(self.z)        
        num[:-1] = - self.thetas * self.beta * (self.z[1:] - self.deltas * self.z[:-1] - forward_pass[1:])
        den[:-1] =  self.beta * (self.thetas ** 2)
        return num, den
   
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
           
    #================= Overridden Spatial Hooks with Temporal Logic =================#
    def _get_Y(self, include_reset: bool = True) -> torch.Tensor:
        """Spiking target includes temporal leakage and reset penalties."""
        return self.z - self._format_bias() - self._compute_temporal_dependencies(include_reset)

    def _apply_lagrange_to_weights(self, lambda_lagrange, a_prev, P) -> torch.Tensor:
        """Spiking Lagrange penalty only applies to the final timestep."""
        P = self._compute_P(a_prev[-1].unsqueeze(0))
        lamb = self._broadcast_to_match(lambda_lagrange, self.z[-1])
        lamb = lamb.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])
        return (lamb.t() / self.beta) @ P

    def _apply_lagrange_to_Y(self, target, lambda_lagrange) -> torch.Tensor:
        """Applies the Lagrange multiplier strictly to the final timestep of the target."""
        lam_sp = self._broadcast_to_match(lambda_lagrange, self.z[-1])
        target_clone = target.clone() 
        target_clone[-1] = target_clone[-1] + (lam_sp / self.beta)
        return target_clone
    
    #================= Overridden vectorized operations =================#
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        """
        STANDARD SEQUENTIAL PASS (Initialization / Inference)
        Simulates the Leaky Integrate-and-Fire (LIF) mechanics step-by-step over the time dimension (T).
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
        """
        VECTORIZED ADMM PASS (Optimization / Constraint Evaluation)
        """
        z = self.spatial_forward(a_prev) + self._compute_temporal_dependencies()
        return z
        
    def update_a(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """
        Updates the activation 'a'.
        a = gamma * h(z) + beta * adjoint(next_layer.Y- temporal_penalties + (lambda/2*beta)) / gamma + beta * W^T W + temporal penalties 
        
        """
        inside_adjoint = next_layer._get_Y()
        if lambda_lagrange is not None:
            inside_adjoint = next_layer._apply_lagrange_to_Y(inside_adjoint, lambda_lagrange / 2.0)
                
        temp_num, temp_den = 0.0, 0.0

        forward_pass = self.spatial_forward(a_prev)
        temp_num, temp_den = self._get_temporal_a_penalties(forward_pass)

        W = self._get_expanded_weights(next_layer)  
        in_features = W.size(1)
        I = torch.eye(in_features, device=next_layer.W.device, dtype=W.dtype)
        WtW = torch.matmul(W.t(), W) 
        
        h_z = self.h(self.z)
        adjoint = next_layer.adjoint_operator(inside_adjoint, original_input_shape=self.a.shape)
        
        numerator = self.gamma * h_z + next_layer.beta * adjoint + temp_num
        denominator = self.gamma * I + next_layer.beta * WtW
        
        # Spiking Case
        T = self.z.size(0)
        td_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
        LHS_main = denominator + (td_scalar * I)
        new_a = solve_spiking_system(LHS_main, denominator, numerator, self.a.shape, in_features, T)    
        new_a = torch.clamp(new_a, min=0.0, max=1)
        self.a.data.copy_(new_a)



    def update_bias(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Averages the residual errors to update the bias vector."""
        target = self.z - self.spatial_forward(a_prev, use_bias=False)
        
        target = target - self._compute_temporal_dependencies()
        if lambda_lagrange is not None:
            lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z[-1])
            #target[-1] = target[-1] + (lam_spatial / (2 * self.beta))
            target[-1] = target[-1] + (lam_spatial / (self.beta))
       
                
        new_bias = torch.mean(target, dim=self._get_bias_reduction_dims())
        self.b.data.copy_(new_bias)

    def update_z_last(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Manager that reroutes the target z_last update depending on the chosen training method."""
        """
        Solves the proximal update for the 'z' variable for the LAST layer.
       
        z =  beta * (forward(a_prev)) + (labels - (lamb / 2)) / (1 + beta)
        Spiking Case: Incorporates temporal penalties into the numerator and denominator.
        """ 

        forward = self.vectorized_forward(a_prev) #This implicitly computes temporal dependencies if spiking=True
        shape = self.z[-1] 
            
        labels = self._broadcast_to_match(labels, shape)
        lamb = self._broadcast_to_match(lambda_lagrange, shape)
        num, den = self._get_temporal_z_penalties(a_prev, labels, lamb)
                
        numerator = (self.beta * forward) + num
        denominator = self.beta + den
            
        self.z.data.copy_(numerator / denominator)     
    
     #================= Unrolled operations =================#

    def _create_cache(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Precomputes and distributes operations to accelerate the unrolled loop."""
        T, batch_size = self.z.shape[:2]

        forward_pass = self.spatial_forward(a_prev)
        target_z = next_layer._get_Y(include_reset=False)
        
        if lambda_lagrange is not None:
            target_z = next_layer._apply_lagrange_to_Y(target_z, lambda_lagrange)
            
        projected_target = next_layer.adjoint_operator(target_z, original_input_shape=self.a.shape)
        base_RHS = next_layer.beta * projected_target

        temp_num, _ = self._get_temporal_a_penalties(forward_pass)
        TERM2 = base_RHS + temp_num if getattr(self, 'use_reset', False) and isinstance(temp_num, torch.Tensor) else base_RHS          
               
        W_flat = self._get_expanded_weights(next_layer)
        
        I = torch.eye(W_flat.size(1), device=next_layer.W.device, dtype=W_flat.dtype)
        WtW = torch.matmul(W_flat.t(), W_flat) 
                
        LHS_last = self.gamma * I + next_layer.beta * WtW
        LHS_main = LHS_last + self.beta * (self.thetas ** 2) * I if getattr(self, 'use_reset', False) else LHS_last
            
        term1_a_main = torch.linalg.inv(LHS_main).t()          
        term1_a_last = torch.linalg.inv(LHS_last).t()

        term2_flat = TERM2.view(T * batch_size, -1)
        A_main_cache_flat = torch.matmul(term2_flat, term1_a_main)
         
        term1_term2_a_main = A_main_cache_flat.view(T, batch_size, -1)
        term1_term2_a_last = torch.matmul(TERM2[-1].view(batch_size, -1), term1_a_last)
            
        return TemporalCache(
            forward_pass=forward_pass,
            term1_a_main=term1_a_main,
            term1_a_last=term1_a_last,
            term1_term2_a_main=term1_term2_a_main,
            term1_term2_a_last=term1_term2_a_last
        )
    
    
    def update_az_interleaved(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list):
        """Orchestrates the interleaved updates of 'a' and 'z' over time with caching."""
        cache = self._create_cache(next_layer, a_prev, lambda_lagrange)
        
        for t in time_steps:
            self._update_a_unrolled(t, cache)
            self.update_z_unrolled(t, cache)
            
        # The final timestep is always updated last
        t_final = self.z.size(0) - 1
        self._update_a_unrolled(t_final, cache)
        self.update_z_unrolled(t_final, cache)
         
            
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
            a_t_flat = torch.matmul(TERM3, cache.term1_a_main) + cache.term1_term2_a_main[t]
        else:
            a_t_flat = torch.matmul(TERM3, cache.term1_a_last) + cache.term1_term2_a_last
            
        self.a[t].data.copy_(torch.clamp(a_t_flat.view_as(self.a[t]), min=0.0, max=1.0))
    
        
    def update_z_unrolled(self, t, cache):
        """
         z update for hidden layers:
         We define the proximal update based on two main terms:
         TERM_1: Spiking Forward pass  (Forward pass + delta*z_prev - theta*a_prev)
         TERM_2: z_next - Forward Pass
         Then we apply the proximal operator to these terms.
        """
        T = self.z.size(0) #Time = z size
        TERM_1 = cache.forward_pass[t]
        if t > 0:
            TERM_1 = TERM_1 + (self.deltas * self.z[t-1] - self.thetas * self.a[t-1])
        
        TERM_2 = self.z[t+1] - cache.forward_pass[t+1] if t < T - 1 else None
        
        new_z_t = self.h.activation_z_unrolled(TERM_1, TERM_2, self.a[t])  
        self.z[t].data.copy_(new_z_t)

        
    def update_z_hybrid(self, a_prev: torch.Tensor, time_steps: list):
        """
        Hybrid causal sweep for the z update.
        Reuses the unrolled logic across all timesteps without redefining the math.
        """
        # 1. Compute the forward pass on the fly
        forward_pass = self.spatial_forward(a_prev)
        
        # 2. Create a lightweight mock cache to satisfy the unrolled method signature
        mock_cache = SimpleNamespace(forward_pass=forward_pass)

        # 3. Execute the sweep using the single-timestep mathematical operator
        for t in time_steps:
            self.update_z_unrolled(t, mock_cache)
        
 

    def update_z_last_unrolled(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list):
        """
        Urolled z update for the LAST layer (L):
        Computes z_L,t = denominator / numerator
        where: 
            denominator = beta * (forward + delta * z_t-1) + (beta * delta * (z - forward)) if t<T-1
                        = beta * (forward + delta * z_t-1) + (beta * delta * (z - forward)) + (lamda*(delta/2)) if t=T-1
                        = beta * (forward + delta * z_t-1) + (y- (lambda/2)) if t=T
                       
            numerator = beta + (delta^2 * beta) if t<T
                      = beta + 1  if t=T
                      
        """
        T = self.z.size(0) #Time = z size
        labels = self._broadcast_to_match(labels, self.z[-1]) # reshaped labels
        lamb= self._broadcast_to_match(lambda_lagrange, self.z[-1]) #reshaped lambda
        
        forward= self.spatial_forward(a_prev) - self._format_bias()
        
        for t in time_steps:
            term1 = forward[t]
            if t >= 1:
                term1 = term1 + self.deltas * self.z[t-1]
            
            term2 = self.deltas * (self.z[t+1] - forward[t+1])
            term_lambda = lamb if t == T - 2 else torch.zeros_like(lamb)
            z_t = (term1 + term2 + self.deltas * term_lambda / self.beta) / (1 + self.deltas ** 2)
            self.z[t].data.copy_(z_t)

        t = T - 1
        forward_T = forward[t]
        if t >= 1:
            forward_T = forward_T + self.deltas * self.z[t-1]
    
        z_T = (self.beta * forward_T + 2 * labels - lamb) / (2 + self.beta)
        self.z[t].data.copy_(z_T)


