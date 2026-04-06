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
    def setup(self, config: dict):
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
        
    def setup(self, config: dict):
        pass 

    def forward(self, x):
        return x 
   
    def activation_z_update(self, a, res, **kwargs):
        return res 
    
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
        self.rho = None
        self.beta = None 
        
    def setup(self, config: dict):
        self.rho = config.get('rho', self.rho)
        self.beta = config.get('beta', self.beta)   

    def forward(self, x):
        return torch.relu(x)
   
    def activation_z_update(self, a, res, **kwargs):
        """Calculates the proximal update for ReLU.

        Formula: z = max(0, (beta * a + rho * res) / (beta + rho))

        Args:
            a (torch.Tensor): The pre-activation tensor.
            res (torch.Tensor): The residual tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The updated z tensor.
        """
        z = (self.beta * a + self.rho * res) / (self.beta + self.rho)
        return torch.where(z > 0, z, res)


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
        self.rho = None
        self.beta = None
        self.deltas = None 
        
    def setup(self, config: dict):
        self.thetas = config.get('thetas', self.thetas)
        self.deltas = config.get('deltas', self.deltas)
        self.rho = config.get('rho', self.rho)
        self.beta = config.get('beta', self.beta)
        
    def forward(self, x):
        return (x > self.thetas).to(x.dtype)
    
    def check_entries(self, z: torch.Tensor, q: torch.Tensor, a: torch.Tensor, r_next: torch.Tensor = None, is_sequence: bool = False):
        """Universal zero-allocation boolean masking for ADMM energy penalties.

        Handles both single-timestep (unrolled) and full-sequence (vectorized) updates.

        Args:
            z (torch.Tensor): The current z tensor.
            q (torch.Tensor): The residual target tensor.
            a (torch.Tensor): The pre-activation tensor.
            r_next (torch.Tensor, optional): The next time step's residual. Defaults to None.
            is_sequence (bool, optional): Flag indicating if this is a vectorized sequence update. Defaults to False.

        Returns:
            torch.Tensor: The updated z tensor after evaluating energy penalties.
        """
        delta1 = self.beta * (1.0 - 2.0 * a)
        delta2 = self.rho * ((z - q) ** 2 - self.rho * (self.thetas - q) ** 2) #BUG
        #delta2 = self.rho * ((z - q)**2 - (self.thetas - q)**2) #BUG

        if r_next is not None:
            temp_penalty = self.rho * ((r_next - self.deltas * z + self.thetas * a)**2 - 
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

    def activation_z_unrolled(self, forward, r_ltnext, a_t):
        """Unrolled version of the update, handling specific time-step logic.

        Formula: 
        z = (forward + deltas * (r_ltnext + thetas * a_t)) / (1.0 + deltas^2) if t<T
        z = forward if t=T
        
        Args:
            q (torch.Tensor): The residual target tensor.
            r_ltnext (torch.Tensor): The temporal penalty tensor from the next timestep.
            a_t (torch.Tensor): The pre-activation tensor at the current timestep.

        Returns:
            torch.Tensor: The updated z tensor for the current timestep.
        """
        if r_ltnext is not None:
            z_res = (forward + self.deltas * (r_ltnext + self.thetas * a_t)) / (1.0 + self.deltas ** 2)
        else:
            z_res = forward.clone() 
        return self.check_entries(z_res, forward, a_t, r_ltnext, is_sequence=False)
    
    def activation_z_update(self, res, z, a, **kwargs):
        """Executes a Pure Vectorized (Jacobi) block of the z-update.

        Args:
            res (torch.Tensor): The residual tensor.
            z (torch.Tensor): The current $z$ tensor.
            a (torch.Tensor): The pre-activation tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The updated $z$ tensor.
        """

        q = res.clone()
        q[1:] += self.deltas * z[:-1] 
        q[1:] -= self.thetas * a[:-1] 

        r = z - res
            
        numerator = self.rho * q
        denominator = self.rho

        temporal_penalty_num = torch.zeros_like(numerator)
        temporal_penalty_den = torch.zeros_like(numerator)
        temporal_penalty_num[:-1] = self.deltas * self.rho * (r[1:] + self.thetas * a[:-1]) 
        temporal_penalty_den[:-1] = (self.deltas**2) * self.rho  

        numerator = numerator + temporal_penalty_num
        denominator = denominator + temporal_penalty_den

        return self.check_entries(numerator / denominator,  q, a, r, is_sequence=True)
