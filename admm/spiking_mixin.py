"""
ADMM Spiking Mixin

This module provides temporal modeling and unrolled solvers for Spiking 
Neural Networks (SNNs). It is designed to be used as a Mixin, overriding 
standard spatial hooks with temporal dependencies (leakage, reset).
"""


import torch
import torch.nn as nn
from .solvers import solve_spiking_system
from types import SimpleNamespace
from .temporal_helpers import TemporalCache, fold_time, unfold_time, compute_temporal_dependencies, get_temporal_a_penalties, get_temporal_z_penalties

############################################################################################################
#Spiking mixing
############################################################################################################ 

class ADMM_Spiking:
    """Mixin class that provides temporal modeling and unrolled solvers for Spiking Neural Networks (SNNs)."""
    def _fold_time(self, x: torch.Tensor):
        return fold_time(x=x)

    def _unfold_time(self, x_flat: torch.Tensor, tb_shape: tuple):
        return unfold_time(x_flat=x_flat, tb_shape=tb_shape)
    
    def _compute_temporal_dependencies(self, include_reset: bool = True) -> torch.Tensor:     
        return compute_temporal_dependencies(
            self.z, self.a, self.deltas, self.thetas, 
            getattr(self, 'use_reset', False), include_reset
        )
    
    def _get_temporal_a_penalties(self, forward_pass: torch.Tensor):
        return get_temporal_a_penalties(
            self.z, forward_pass, self.deltas, self.thetas, 
            self.rho, getattr(self, 'use_reset', False)
        )
   
    def _get_temporal_z_penalties(self, a_prev: torch.Tensor, labels: torch.Tensor, lamb: torch.Tensor):
        forward = self.spatial_forward(a_prev)
        return get_temporal_z_penalties(
            self.z, forward, labels, lamb, self.deltas, self.rho
        )
    

    #================= Overridden Spatial Hooks with Temporal Logic =================#
    def _get_Y(self, include_reset: bool = True) -> torch.Tensor:
        """Returns the spiking target, including temporal leakage and reset penalties.

        Args:
            include_reset (bool, optional): Whether to include the spike reset penalty. Defaults to True.

        Returns:
            torch.Tensor: The calculated spiking target tensor.
        """
        return self.z - self._format_bias() - self._compute_temporal_dependencies(include_reset)

    def _apply_lagrange_to_weights(self, lambda_lagrange, a_prev, P) -> torch.Tensor:
        """Spiking Lagrange penalty only applies to the final timestep's weights.

        Args:
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            a_prev (torch.Tensor): The previous layer's activations.
            P (torch.Tensor): The computed $P$ matrix.

        Returns:
            torch.Tensor: The penalty to be added to the weight update.
        """
        P = self._compute_P(a_prev[-1].unsqueeze(0))
        lamb = self._broadcast_to_match(lambda_lagrange, self.z[-1])
        lamb = lamb.movedim(self.channel_dim, -1).reshape(-1, self.W.shape[0])
        return (lamb.t() / self.rho) @ P

    def _apply_lagrange_to_Y(self, target, lambda_lagrange) -> torch.Tensor:
        """Applies the Lagrange multiplier strictly to the final timestep of the target.

        Args:
            target (torch.Tensor): The target tensor.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.

        Returns:
            torch.Tensor: The penalized target tensor.
        """
        lam_sp = self._broadcast_to_match(lambda_lagrange, self.z[-1])
        target_clone = target.clone() 
        target_clone[-1] = target_clone[-1] + (lam_sp / self.rho)
        return target_clone

    def _get_a_denominator(self, beta_current, a_shape, temp_den=0.0, unrolled=False):
        """Computes the denominator matrix for the activation (a) update step.

        Formula:
        denomitator = beta I + rho W^TW + temporal_penalties
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
        
        denominator_last =  beta_current * I + self.rho * WtW
        td_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
        denominator_main = denominator_last + (td_scalar * I)
        if unrolled:
            denominator_main = torch.linalg.inv(denominator_main).t()          
            denominator_last = torch.linalg.inv(denominator_last).t()
        return  denominator_main, denominator_last, in_features

    
    #================= Overridden vectorized operations =================#
    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        """Standard sequential pass for initialization or inference.

        Simulates the Leaky Integrate-and-Fire (LIF) mechanics step-by-step 
        over the time dimension ($T$).

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
        """Updates the activation $a$ for spiking networks.

        Mathematical formulation:
         a = beta * h(z) + rho * adjoint(next_layer.Y- temporal_penalties + (lambda/2*rho)) / beta + rho * W^T W + temporal penalties 

        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        inside_adjoint = next_layer._get_Y()
        if lambda_lagrange is not None:
            inside_adjoint = next_layer._apply_lagrange_to_Y(inside_adjoint, lambda_lagrange / 2.0)

        forward_pass = self.spatial_forward(a_prev)
        temp_num, temp_den = self._get_temporal_a_penalties(forward_pass)

        h_z = self.h(self.z)
        adjoint = next_layer.adjoint_operator(inside_adjoint, original_input_shape=self.a.shape)
        
        numerator = self.beta * h_z + next_layer.rho * adjoint + temp_num
        
        new_a = next_layer.solve_activation_system(
            numerator=numerator,
            a_shape=self.a.shape,
            beta_current=self.beta,
            temp_den=temp_den
        )           
        
        new_a = torch.clamp(new_a, min=0.0, max=1.0)
        self.a.data.copy_(new_a)
        
    
    # In spiking.py (ADMM_Spiking)
    def solve_activation_system(self, numerator: torch.Tensor, a_shape: tuple, beta_current: float, temp_den=0.0) -> torch.Tensor:
        # Correctly unpack all 3 values
        den_main, den_last, in_features = self._get_a_denominator(beta_current, a_shape, temp_den)
        return solve_spiking_system(
            A_main=den_main, 
            A_last=den_last, 
            B=numerator, 
            out_shape=a_shape, 
            in_features=in_features,
            T=numerator.size(0) 
        )
    
        

    def update_bias(self, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor = None):
        """Averages the residual errors across spatial and temporal dimensions to update the bias vector.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor, optional): The Lagrange multiplier. Defaults to None.
        """
        target = self.z - self.spatial_forward(a_prev, use_bias=False)
        
        target = target - self._compute_temporal_dependencies()
        if lambda_lagrange is not None:
            lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z[-1])
            target[-1] = target[-1] + (lam_spatial / (self.rho))
       
                
        new_bias = torch.mean(target, dim=self._get_bias_reduction_dims())
        self.b.data.copy_(new_bias)

    def update_z_last(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Solves the proximal update for the $z$ variable for the final layer, incorporating 
        spiking temporal penalties into the numerator and denominator.

        Mathematical formulation:
        z =  rho * (forward(a_prev)) + (labels - (lamb / 2)) / (1 + rho)

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
        """
        forward = self.vectorized_forward(a_prev) #This implicitly computes temporal dependencies if spiking=True
        shape = self.z[-1] 
            
        labels = self._broadcast_to_match(labels, shape)
        lamb = self._broadcast_to_match(lambda_lagrange, shape)
        num, den = self._get_temporal_z_penalties(a_prev, labels, lamb)
                
        numerator = (self.rho * forward) + num
        denominator = self.rho + den
            
        self.z.data.copy_(numerator / denominator)     
    
     #================= Unrolled operations =================#
    
    def _create_cache(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Delegates cache construction to the TemporalCache factory.
            
        Args:
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.

        Returns:
            TemporalCache: A typed data class containing the precomputed matrices."""
        return TemporalCache.build(self, next_layer, a_prev, lambda_lagrange)

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
            self._update_a_unrolled(t, cache, next_layer)
            self.update_z_unrolled(t, cache)
            
        # The final timestep is always updated last
        t_final = self.z.size(0) - 1
        self._update_a_unrolled(t_final, cache, next_layer)
        self.update_z_unrolled(t_final, cache)
         
    
    def _update_a_unrolled(self, t: int, cache: TemporalCache, next_layer: nn.Module):
        """Performs the unrolled activation (a) update for a specific timestep.

        a update: (rho*W^T_l+1*W_l+1+beta*I)^-1 (rho*W^T_l+1*v_l+1 + beta*h)
        We call this TERM1 * (TERM2 + TERM3)
        
        Utilizes precomputed TERM1 and TERM2 from the `TemporalCache` 
        to execute pure matrix multiplication without calling a solver.

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices (TERM1 and TERM2).
        """
        h_t = self.h(self.z[t])
        TERM3 = self.beta * h_t
        
        RHS_t = cache.term2[t] + TERM3
        
        # Select the correct matrix/denominator from the cache
        is_last = (t == self.z.size(0) - 1)
        math_box = cache.denominator_last if is_last else cache.denominator_main
        
        # POLYMORPHIC CALL: Hand the RHS and the math_box back to the layer!
        new_a_t = next_layer.solve_unrolled_step(RHS_t, math_box)
            
        self.a[t].data.copy_(torch.clamp(new_a_t, min=0.0, max=1.0))
    
    # In spiking.py (ADMM_Spiking)
    def solve_unrolled_step(self, RHS: torch.Tensor, math_box: torch.Tensor) -> torch.Tensor:
        """
        Executes the unrolled step using dense matrix multiplication.
        Reshaping is aligned with solve_linear_system in solvers.py.
        """
        # math_box is [in_features, in_features]. Use that for the reshape.
        in_features = math_box.size(1)
        
        # Correctly flatten all dimensions except the one matching math_box
        RHS_flat = RHS.reshape(-1, in_features)
        
        # Multiply by the cached inverse matrix
        a_t_flat = torch.matmul(RHS_flat, math_box)
        
        # Return to original 4D shape [Batch, C, H, W]
        return a_t_flat.view_as(RHS)

             
    def update_z_unrolled(self, t, cache):
        """Performs the unrolled pre-activation (z) update for hidden layers.

        Defines the proximal update based on two main terms:
         TERM_1: Spiking Forward pass  (Forward pass + delta*z_prev - theta*a_prev)
         TERM_2: z_next - Forward Pass

        Applies the proximal operator to these terms via the activation function's 
        specialized unrolled solver.

        Args:
            t (int): The current timestep index.
            cache (TemporalCache): The precomputed matrices.
        """
        T = self.z.size(0) #Time = z size
        TERM_1 = cache.forward_pass[t]
        if t > 0:
            TERM_1 = TERM_1 + (self.deltas * self.z[t-1] - self.thetas * self.a[t-1])
        
        TERM_2 = self.z[t+1] - cache.forward_pass[t+1] if t < T - 1 else None
        
        new_z_t = self.h.activation_z_unrolled(TERM_1, TERM_2, self.a[t])  
        self.z[t].data.copy_(new_z_t)

        
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
        
 

    def update_z_last_unrolled(self, a_prev: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, time_steps: list):
        """
        Urolled z update for the final layer (L).
        Computes z_L,t = denominator / numerator
        where: 
            denominator = rho * (forward + delta * z_t-1) + (rho * delta * (z - forward)) if t<T-1
                        = rho * (forward + delta * z_t-1) + (rho * delta * (z - forward)) + (lamda*(delta/2)) if t=T-1
                        = rho * (forward + delta * z_t-1) + (y- (lambda/2)) if t=T
                       
            numerator = rho + (delta^2 * rho) if t<T
                      = rho + 1  if t=T
        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            time_steps (list): The list of sequence time steps to update.               
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
            z_t = (term1 + term2 + self.deltas * term_lambda / self.rho) / (1 + self.deltas ** 2)
            self.z[t].data.copy_(z_t)

        t = T - 1
        forward_T = forward[t]
        if t >= 1:
            forward_T = forward_T + self.deltas * self.z[t-1]
    
        z_T = (self.rho * forward_T + 2 * labels - lamb) / (2 + self.rho)
        self.z[t].data.copy_(z_T)


