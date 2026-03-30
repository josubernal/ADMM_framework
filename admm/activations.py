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

IMPORTANT:
The ADMM_Heaviside has a bug to be discussed.

"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ADMMActivationBase(nn.Module, ABC):
    """
    Abstract Base Class for ADMM Activation Functions.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def setup(self, config: dict):
        """Receives and stores ADMM hyperparameters (gamma, beta, etc.) from the parent layer."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the standard non-linear activation (used in the forward pass)."""
        pass

    @abstractmethod
    def activation_z_update(self, **kwargs) -> torch.Tensor:
        """Solves the optimization step for the auxiliary variable 'z'."""
        pass


####################################################################################################
# ACTIVATION FUNCTIONS
####################################################################################################
    

class ADMM_Identity(ADMMActivationBase):
    """
    A Pass-Through (Identity) activation function. 
    """
    def __init__(self):
        super().__init__()
        
    def setup(self, config: dict):
        pass 

    def forward(self, x):
        return x 
   
    def activation_z_update(self, a, res, **kwargs):
        return res 
    
class ADMM_ReLU(ADMMActivationBase):
    """
    ADMM implementation of the Rectified Linear Unit (ReLU).
    
    How it works:
    - Forward: Standard max(0, x).
    - Proximal Z-Update: Solves a constrained quadratic minimization. It calculates an 
      unconstrained weighted average between the pre-activation state ('a') and the 
      residual state ('res'), and then simply clips negative values to 0.
    """
    def __init__(self):
        super().__init__()  
        self.beta = None
        self.gamma = None 
        
    def setup(self, config: dict):
        self.beta = config.get('beta', self.beta)
        self.gamma = config.get('gamma', self.gamma)   

    def forward(self, x):
        return torch.relu(x)
   
    def activation_z_update(self, a, res, **kwargs):
        """
        Formula: z = max(0, (gamma * a + beta * res) / (gamma + beta))
        """
        z = (self.gamma * a + self.beta * res) / (self.gamma + self.beta)
        return torch.where(z > 0, z, res)


class ADMM_Heaviside(ADMMActivationBase):
    """
    ADMM implementation of a Heaviside (Step) Function, often used in Spiking Neural Networks.
    
    How it works:
    - Forward: A binary step function. Outputs 1 if x > theta, otherwise 0.
    - Proximal Z-Update: Highly specialized. Because the Heaviside function is non-differentiable 
      and non-convex, the proximal step evaluates distinct energy states (spiking vs. not spiking) 
      and incorporates temporal dependencies (leakage and spike reset) over sequence steps.
    """
    def __init__(self, thetas=1.0):
        super().__init__()  
        self.thetas = thetas
        self.beta = None
        self.gamma = None
        self.deltas = None 
        
    def setup(self, config: dict):
        self.thetas = config.get('thetas', self.thetas)
        self.deltas = config.get('deltas', self.deltas)
        self.beta = config.get('beta', self.beta)
        self.gamma = config.get('gamma', self.gamma)
        
    def forward(self, x):
        return (x > self.thetas).to(x.dtype)
    
    def check_entries(self, z: torch.Tensor, q: torch.Tensor, a: torch.Tensor, r_next: torch.Tensor = None, is_sequence: bool = False):
        """
        Universal zero-allocation boolean masking for the ADMM energy penalties.
        Handles both single-timestep (unrolled) and full-sequence (vectorized) updates.
        """
        delta1 = self.gamma * (1.0 - 2.0 * a)
        delta2 = self.beta * ((z - q) ** 2 - self.beta * (self.thetas - q) ** 2) #BUG
        #delta2 = self.beta * ((z - q)**2 - (self.thetas - q)**2) #BUG

        if r_next is not None:
            temp_penalty = self.beta * ((r_next - self.deltas * z + self.thetas * a)**2 - 
                                        (r_next - self.deltas * self.thetas + self.thetas * a)**2)
            
            if is_sequence:
                temp_penalty[-1] = 0.0
                
            delta2 += temp_penalty

        mask_z_greater = z > self.thetas
        mask_deltas1 = (delta1 + delta2) > 0
        mask_deltas2 = (delta2 - delta1) > 0  
            
        z[mask_z_greater & mask_deltas1] = self.thetas
        z[(~mask_z_greater) & mask_deltas2] = self.thetas + 1e-5
        
        return z

    def activation_z_unrolled(self, q, r_ltnext, a_t):
        """Unrolled version of the proximal update, handling specific time-step logic."""
        if r_ltnext is not None:
            z_res = (q + self.deltas * (r_ltnext + self.thetas * a_t)) / (1.0 + self.deltas ** 2)
        else:
            z_res = q.clone() 
        return self.check_entries(z_res, q, a_t, r_ltnext, is_sequence=False)
    
    def activation_z_update(self, res, z, a, **kwargs):
        """Specialized z-update using ADMM temporal penalties."""
        q = res.clone()
        q[1:] += self.deltas * z[:-1] 
        q[1:] -= self.thetas * a[:-1] 

        r = z - res
        
        numerator = self.beta * q
        denominator = self.beta

        temporal_penalty_num = torch.zeros_like(numerator)
        temporal_penalty_den = torch.zeros_like(numerator)
        temporal_penalty_num[:-1] = self.deltas * self.beta * (r[1:] + self.thetas * a[:-1]) 
        temporal_penalty_den[:-1] = (self.deltas**2) * self.beta  

        numerator = numerator + temporal_penalty_num
        denominator = denominator + temporal_penalty_den

        return self.check_entries(numerator / denominator,  q, a, r, is_sequence=True)

 