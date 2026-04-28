"""
ADMM Activation Functions 

This module contains the activation functions designed for ADMM.

In ADMM, activations are treated as constraints or non-linear penalties. 
Therefore, each activation function must provide a z-update.

INDEX:
- ADMMActivationBase
- ADMM_Identity
- ADMM_ReLU
- ADMM_Heaviside
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ADMMActivationBase(nn.Module, ABC):
    """Abstract Base Class for ADMM Activation Functions."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def setup(self, config: dict,  parent_layer=None):
        """Receives and stores ADMM hyperparameters from the parent layer.

        Args:
            config (dict): Dictionary containing hyperparameters like beta, rho, etc.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the standard non-linear activation.
        
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The activated output tensor.
        """
        pass

    @abstractmethod
    def activation_z_update(self, **kwargs) -> torch.Tensor:
        """Solves the optimization step for the auxiliary variable z.

        Args:
            **kwargs: Variable keyword arguments depending on the specific activation.

        Returns:
            torch.Tensor: The updated z tensor.
        """
        pass


####################################################################################################
# ACTIVATION FUNCTIONS
####################################################################################################
    

class ADMM_Identity(ADMMActivationBase):
    """A Pass-Through (Identity) activation function. """
    def __init__(self):
        super().__init__()
        
    def setup(self, config: dict,  parent_layer=None):
        pass 

    def forward(self, x):
        return x 
   
    def activation_z_update(self, a, forward, **kwargs):
        return forward
    
class ADMM_ReLU(ADMMActivationBase):
    """ADMM implementation of the Rectified Linear Unit (ReLU).

    How it works:
    - Forward: Standard max(0, x).
    - z-update: Solves a constrained quadratic minimization. It calculates an 
    unconstrained weighted average between the pre-activation state (a) and the 
    residual state (res), and then simply clips negative values to 0.
    """
    def __init__(self):
        super().__init__()  
        self._layer_ref = [None] 
        
    def setup(self, config: dict, parent_layer=None):
        self._layer_ref[0] = parent_layer 

    @property
    def rho(self):
        return self._layer_ref[0].rho # Pull from index 0
        
    @property
    def beta(self):
        return self._layer_ref[0].beta

    def forward(self, x):
        return torch.relu(x)
   
    def activation_z_update(self, a, forward, **kwargs):
        """Calculates the proximal update for ReLU.

        Formula: z = max(0, (beta * a + rho * forward) / (beta + rho))

        Args:
            a (torch.Tensor): The pre-activation tensor.
            forward (torch.Tensor): The residual tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The updated z tensor.
        """
        z = (self.beta * a + self.rho * forward) / (self.beta + self.rho)
        return torch.where(z > 0, z, forward)


class ADMM_Heaviside(ADMMActivationBase):
    """ADMM implementation of a Heaviside (Step) Function.

    Often used in Spiking Neural Networks.

    How it works:
    - Forward: A binary step function. Outputs 1 if x > theta, otherwise 0.
    - z-update: Highly specialized. Because the Heaviside function is 
    non-differentiable and non-convex, the step evaluates distinct energy 
    states (spiking vs. not spiking) and incorporates temporal dependencies 
    (leakage and spike reset) over sequence steps.
    """        
    def __init__(self, thetas=1.0):
        super().__init__()  
        self.thetas = thetas
        self.deltas = None 
        self._layer_ref = [None]
        
    def setup(self, config: dict, parent_layer=None):
        self.thetas = config.get('thetas', self.thetas)
        self.deltas = config.get('deltas', self.deltas)
        self._layer_ref[0] = parent_layer
        
    @property
    def rho(self):
        return self._layer_ref[0].rho # Pull from index 0
        
    @property
    def beta(self):
        return self._layer_ref[0].beta
        
    def forward(self, x):
        return (x > self.thetas).to(x.dtype)
    
    def check_entries(self, z: torch.Tensor, temporal_forward: torch.Tensor, a: torch.Tensor,  z_minus_forward: torch.Tensor = None, is_vectorized: bool = False):
        """Universal boolean masking for ADMM discrete energy evaluation.

        Enforces the non-convex Heaviside domain constraints by explicitly evaluating 
        the discrete Lagrangian energy difference between the spiking and resting states. 

        Mathematical Formulation:
      
        delta_1 = beta * (1 - 2a)
        delta_2 = rho * ((z - temporal_forward)^2 - (theta - temporal_forward)^2)
        delta_3 = rho ((z - forward - delta * z + theta * a)^2 - ( z - forward - delta * theta + theta * a)^2)

        z = theta if z > theta and delta_1 + delta_2 + delta_3 > 0
        z = theta + epsilon if z < theta and delta_2 + delta_3 - delta_1 > 0

        Args:
            z (torch.Tensor): The unconstrained pre-activation minimizer (z^*).
            temporal_forward (torch.Tensor): The forward pass target.
            a (torch.Tensor): The current activation state.
            z_minus_forward (torch.Tensor, optional): The residual of the next timestep 
                (z_{t+1} - F_{t+1}), used for calculating delta_3. Defaults to None.
            is_vectorized (bool, optional): Flag indicating if this is a sequence update 
                (applies the indicator function to zero out delta_3 at t=T). Defaults to False.

        Returns:
            torch.Tensor: The projected and bounded $z$ tensor.
        """
        delta1 = a.mul(-2.0).add_(1.0).mul_(self.beta)

        total_delta = z.sub(temporal_forward).pow_(2)
        tmp = temporal_forward.sub(self.thetas).pow_(2)
        total_delta.sub_(tmp).mul_(self.rho)
        total_delta.add_(delta1) # total_delta is now delta1 + delta2
    
        if z_minus_forward is not None:
            res_term = z_minus_forward.add(a, alpha=self.thetas) 
            
            d3 = res_term.sub(self.deltas * z).pow_(2)
            d3_sub = res_term.sub(self.deltas * self.thetas).pow_(2)
            d3.sub_(d3_sub).mul_(self.rho)
            
            if is_vectorized:
                d3[-1].zero_()
                
            total_delta.add_(d3)

        mask1 = (z > self.thetas).logical_and_(total_delta > 0)
        
        total_delta.add_(delta1, alpha=-2.0) 
        mask2 = (z <= self.thetas).logical_and_(total_delta > 0)

        z.masked_fill_(mask1, self.thetas)
        z.masked_fill_(mask2, self.thetas + 1e-5)
        return z

    def activation_z_unrolled(self, temporal_forward, z_minus_forward, a_t):
        """Unrolled version of the update, handling specific time-step logic.

        Formula: 
        z = (temporal_forward + deltas * (z - forward + thetas * a_t)) / (1.0 + deltas^2) if t<T
        z = temporal_forward if t=T
        
        Args:
            q (torch.Tensor): The residual target tensor.
            r_ltnext (torch.Tensor): The temporal penalty tensor from the next timestep.
            a_t (torch.Tensor): The pre-activation tensor at the current timestep.

        Returns:
            torch.Tensor: The updated z tensor for the current timestep.
        """
        if z_minus_forward is not None:
            z_res = z_minus_forward.add(a_t, alpha=self.thetas)
            z_res.mul_(self.deltas).add_(temporal_forward)
            z_res.div_(1.0 + self.deltas ** 2)
        else:
            z_res = temporal_forward.clone()
        return self.check_entries(z=z_res, temporal_forward=temporal_forward, a= a_t,  z_minus_forward= z_minus_forward, is_vectorized=False)
    
    def activation_z_update(self, forward, z, a, **kwargs):
        """Executes a Pure Vectorized (Jacobi) block of the z-update.

        Args:
            res (torch.Tensor): The residual tensor.
            z (torch.Tensor): The current $z$ tensor.
            a (torch.Tensor): The pre-activation tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The updated $z$ tensor.
        """

        q = forward.clone()
        q[1:].add_(z[:-1], alpha= self.deltas)
        q[1:].add_(a[:-1], alpha=-self.thetas)

        r = torch.zeros_like(z)
        r[:-1].copy_(z[1:]).sub_(forward[1:])
            
        numerator = q.mul(self.rho)
        denominator = torch.full_like(numerator, self.rho)

        num_slice = numerator[:-1]
        # temp_term = (z_next - res_next) + thetas * a_curr
        temp_term = z[1:].sub(forward[1:]).add_(a[:-1], alpha=self.thetas)
        num_slice.add_(temp_term, alpha=self.deltas * self.rho)
        
        denominator[:-1].add_( (self.deltas**2) * self.rho )
        z_new = numerator.div_(denominator)

        return self.check_entries(z_new,  q, a, r, is_vectorized=True)
