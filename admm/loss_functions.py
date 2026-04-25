"""
ADMM Loss Functions Module

This module defines the objective functions for the ADMM network.
Each loss function provides both a scalar calculation for metrics, 
and the analytical 'z_update' formula to solve the final layer's ADMM subproblem.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

class ADMM_Loss(ABC):
    """Abstract Base Class for ADMM Loss Functions."""
    
    @abstractmethod
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the scalar loss for logging and metrics.
        """
        pass
        
    @abstractmethod
    def  update_z_last_core(self, v: torch.Tensor, targets: torch.Tensor, rho: float) -> torch.Tensor:
        pass

class ADMM_SSE(ADMM_Loss):
    """Mean Squared Error (L2) Loss for ADMM."""
    def __str__(self):
        return "SSE_Loss"
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets, reduction='sum')

    def update_z_last_core(self, forward: torch.Tensor, rho: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Solves the proximal update for the 'z' variable for the last layer.

        Formula:
        z_last= numerator / denominator, where
        numerator = rho * forward(a_prev) + (2*labels - lambda)
        denominator = 2 + rho

        For spiking networks, this also incorporates temporal penalties into the 
        numerator and denominator.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth labels.
            lambda_lagrange (torch.Tensor): The Lagrange multiplier.
            time_steps (list, optional): Time steps for spiking networks. Defaults to None.
        """
        forward.mul_(rho)
        forward.add_(labels, alpha=2.0)
        forward.sub_(lambda_lagrange)
        forward.div_(2.0 + rho)
        return forward
    
    def update_z_last_spiking(self, forward: torch.Tensor, temporal_forward: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, z:torch.Tensor, rho:torch.Tensor, deltas: torch.Tensor):
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
        numerator = temporal_forward.clone().mul_(rho)
        z_minus_fwd = z.clone().sub_(forward)
        
        numerator[:-1].add_(z_minus_fwd[1:], alpha=rho * deltas)
        numerator[-2].add_(lambda_lagrange, alpha=deltas)
        numerator[-1].add_(labels, alpha=2.0).sub_(lambda_lagrange)
        
        denominator_main = rho * (deltas ** 2) + rho
        denominator_last = 2.0 + rho
        
        numerator[:-1].div_(denominator_main)
        numerator[-1].div_(denominator_last)
        return numerator 
         
    def update_z_last_unrolled_spiking(self, forward: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor,z:torch.Tensor,rho:torch.Tensor,  deltas:torch.Tensor, time_steps: list, jacobi:bool=False):
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
        T = z.size(0)
        denominator_main =  (rho * deltas ** 2) + rho
        z_to_use = z.clone() if jacobi else z
        buffer = torch.empty_like(z[0])
        for t in time_steps:
            if t == T - 1:
                continue
            buffer.copy_(forward[t])
            if t >= 1:
                buffer.add_(z_to_use[t-1], alpha=deltas)

            buffer.add_(z_to_use[t+1], alpha=deltas)
            buffer.add_(forward[t+1], alpha=-deltas)
            buffer.mul_(rho)
            if t == T - 2:
                buffer.add_(lambda_lagrange, alpha=deltas)
                
            buffer.div_(denominator_main)
            z[t].copy_(buffer)
        t = T - 1
        buffer.copy_(forward[t])
        del forward
        if t >= 1:
            buffer.add_(z_to_use[t-1], alpha=deltas)
        buffer.mul_(rho)
        buffer.add_(labels, alpha=2.0)
        buffer.sub_(lambda_lagrange)
        buffer.div_(2.0 + rho)
        z[t].copy_(buffer)
        if jacobi:
            del z_to_use
        return z      
    
class ADMM_Hinge(ADMM_Loss):
    """
    Hinge Loss for ADMM (Max-Margin Classification).
    Automatically handles both {0, 1} and {-1, 1} label formats.
    """
    def __str__(self):
        return "Hinge_Loss"
    
    def _format_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """Converts {0, 1} labels to {-1, 1}. Leaves {-1, 1} alone."""
        # If the minimum value is 0, we assume it's a {0, 1} dataset
        if targets.min() == 0.0:
            return 2.0 * targets - 1.0
        return targets

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = self._format_labels(targets)
        margin = 1.0 - (y * predictions)
        return torch.clamp(margin, min=0.0).sum()

    def update_z_last_core(self, forward: torch.Tensor, rho: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor):
        y = self._format_labels(labels)
        forward_penalty = forward.sub(lambda_lagrange / rho)
        cond = y * forward_penalty
        
        cond1 = cond >= 1.0
        cond2 = cond <= (1.0 - 1.0 / rho)
        
        z_tilde = torch.where(
            cond1, cond, 
            torch.where(cond2, cond + (1.0 / rho), torch.ones_like(cond))
        )
        return y * z_tilde

    def update_z_last_spiking(self, forward: torch.Tensor, temporal_forward: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, z: torch.Tensor, rho: torch.Tensor, deltas: torch.Tensor):
        """
        Vectorized Hinge update for Spiking Networks.
        Evaluates the temporal leakage for t < T, and the Hinge proximal operator at t = T.
        """
        y = self._format_labels(labels)
        
        numerator = temporal_forward.clone().mul_(rho)
        z_minus_fwd = z.clone().sub_(forward)
        
        # 1. Update for t < T (Identical backward pass logic to MSE)
        numerator[:-1].add_(z_minus_fwd[1:], alpha=rho * deltas)
        numerator[-2].add_(lambda_lagrange, alpha=deltas)
        
        denominator_main = rho * (deltas ** 2) + rho
        numerator[:-1].div_(denominator_main)
        
        # 2. Update for t = T (Hinge Proximal Operator)
        v_T = temporal_forward[-1].sub(lambda_lagrange / rho)
        cond = y * v_T
        
        cond1 = cond >= 1.0
        cond2 = cond <= (1.0 - 1.0 / rho)
        
        z_tilde = torch.where(
            cond1, cond, 
            torch.where(cond2, cond + (1.0 / rho), torch.ones_like(cond))
        )
        
        # Overwrite the final timestep with the mapped Hinge result
        numerator[-1] = y * z_tilde
        return numerator 
         
    def update_z_last_unrolled_spiking(self, forward: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, z: torch.Tensor, rho: torch.Tensor,  deltas: torch.Tensor, time_steps: list, jacobi: bool = False):
        """
        Unrolled Hinge update for Spiking Networks.
        """
        y = self._format_labels(labels)
        
        T = z.size(0)
        denominator_main =  (rho * deltas ** 2) + rho
        z_to_use = z.clone() if jacobi else z
        buffer = torch.empty_like(z[0])
        
        # 1. Loop for t < T (Identical to MSE)
        for t in time_steps:
            if t == T - 1:
                continue
            buffer.copy_(forward[t])
            if t >= 1:
                buffer.add_(z_to_use[t-1], alpha=deltas)

            buffer.add_(z_to_use[t+1], alpha=deltas)
            buffer.add_(forward[t+1], alpha=-deltas)
            buffer.mul_(rho)
            
            if t == T - 2:
                buffer.add_(lambda_lagrange, alpha=deltas)
                
            buffer.div_(denominator_main)
            z[t].copy_(buffer)
            
        # 2. Final Timestep t = T (Hinge Proximal Update)
        t = T - 1
        buffer.copy_(forward[t])
        del forward
        if t >= 1:
            buffer.add_(z_to_use[t-1], alpha=deltas)
            
        # buffer is now strictly 'temporal_forward'
        v_T = buffer.sub(lambda_lagrange / rho)
        cond = y * v_T
        
        cond1 = cond >= 1.0
        cond2 = cond <= (1.0 - 1.0 / rho)
        
        z_tilde = torch.where(
            cond1, cond, 
            torch.where(cond2, cond + (1.0 / rho), torch.ones_like(cond))
        )
        
        z[t].copy_(y * z_tilde)
        
        if jacobi:
            del z_to_use
        return z
    
class ADMM_CrossEntropy(ADMM_Loss):
    """
    Cross-Entropy Loss for ADMM.
    Uses a Proximal Gradient approximation to solve the non-linear z-update analytically.
    """
    def __str__(self):
        return "CrossEntropy_Loss"

    def _ensure_one_hot(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Helper to ensure labels are one-hot encoded for gradient calculations."""
        if labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
            return F.one_hot(labels.view(-1).long(), num_classes=num_classes).to(torch.float32)
        return labels.to(torch.float32)

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # PyTorch F.cross_entropy expects class indices (1D) or probabilities (2D)
        # If targets are one-hot, we convert them to indices for the standard metric calculation
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        else:
            targets = targets.long().view(-1)
            
        return F.cross_entropy(predictions, targets, reduction='sum')

    def update_z_last_core(self, forward: torch.Tensor, rho: float, labels: torch.Tensor, lambda_lagrange: torch.Tensor):
        """
        Static CE z-update using Proximal Gradient Descent.
        Formula: z = forward - (lambda / rho) - ((p - y) / rho)
        """
        y_one_hot = self._ensure_one_hot(labels, num_classes=forward.size(-1))
        
        # Calculate probabilities at the forward pass
        p = F.softmax(forward, dim=-1)
        grad_ce = p - y_one_hot
        
        z_new = forward.clone()
        z_new.sub_(lambda_lagrange, alpha=1.0 / rho)
        z_new.sub_(grad_ce, alpha=1.0 / rho)
        
        return z_new

    def update_z_last_spiking(self, forward: torch.Tensor, temporal_forward: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, z: torch.Tensor, rho: torch.Tensor, deltas: torch.Tensor):
        """
        Vectorized CE update for Spiking Networks.
        Evaluates the temporal leakage for t < T, and the CE proximal operator at t = T.
        """
        y_one_hot = self._ensure_one_hot(labels, num_classes=temporal_forward.size(-1))
        
        numerator = temporal_forward.clone().mul_(rho)
        z_minus_fwd = z.clone().sub_(forward)
        
        # 1. Update for t < T (Identical backward pass logic to MSE/Hinge)
        numerator[:-1].add_(z_minus_fwd[1:], alpha=rho * deltas)
        numerator[-2].add_(lambda_lagrange, alpha=deltas)
        
        denominator_main = rho * (deltas ** 2) + rho
        numerator[:-1].div_(denominator_main)
        
        # 2. Update for t = T (CE Proximal Gradient Operator)
        p = F.softmax(temporal_forward[-1], dim=-1)
        grad_ce = p - y_one_hot
        
        z_T = temporal_forward[-1].clone()
        z_T.sub_(lambda_lagrange, alpha=1.0 / rho)
        z_T.sub_(grad_ce, alpha=1.0 / rho)
        
        # Overwrite the final timestep
        numerator[-1] = z_T
        return numerator 

    def update_z_last_unrolled_spiking(self, forward: torch.Tensor, labels: torch.Tensor, lambda_lagrange: torch.Tensor, z: torch.Tensor, rho: torch.Tensor, deltas: torch.Tensor, time_steps: list, jacobi: bool = False):
        """
        Unrolled CE update for Spiking Networks.
        """
        y_one_hot = self._ensure_one_hot(labels, num_classes=forward.size(-1))
        
        T = z.size(0)
        denominator_main = (rho * deltas ** 2) + rho
        z_to_use = z.clone() if jacobi else z
        buffer = torch.empty_like(z[0])
        
        # 1. Loop for t < T (Identical to MSE/Hinge)
        for t in time_steps:
            if t == T - 1:
                continue
            buffer.copy_(forward[t])
            if t >= 1:
                buffer.add_(z_to_use[t-1], alpha=deltas)

            buffer.add_(z_to_use[t+1], alpha=deltas)
            buffer.add_(forward[t+1], alpha=-deltas)
            buffer.mul_(rho)
            
            if t == T - 2:
                buffer.add_(lambda_lagrange, alpha=deltas)
                
            buffer.div_(denominator_main)
            z[t].copy_(buffer)
            
        # 2. Final Timestep t = T (CE Proximal Gradient Update)
        t = T - 1
        buffer.copy_(forward[t])
        del forward
        if t >= 1:
            buffer.add_(z_to_use[t-1], alpha=deltas)
            
        # buffer is now strictly 'temporal_forward' for the final timestep
        p = F.softmax(buffer, dim=-1)
        grad_ce = p - y_one_hot
        
        buffer.sub_(lambda_lagrange, alpha=1.0 / rho)
        buffer.sub_(grad_ce, alpha=1.0 / rho)
        
        z[t].copy_(buffer)
        
        if jacobi:
            del z_to_use
        return z