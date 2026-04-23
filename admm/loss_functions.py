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
    def z_update(self, v: torch.Tensor, targets: torch.Tensor, rho: float) -> torch.Tensor:
        pass

class ADMM_MSE(ADMM_Loss):
    """Mean Squared Error (L2) Loss for ADMM."""
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)

    def z_update(self, v: torch.Tensor, targets: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Closed-form ADMM update for MSE.
        Formula: z = (y + rho * v) / (1 + rho)
        """
        return (targets + rho * v) / (1.0 + rho)

class ADMM_Hinge(ADMM_Loss):
    """
    Hinge Loss for ADMM (Max-Margin Classification).
    NOTE: Targets MUST be formatted as {-1, 1}, not {0, 1}.
    """
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard Hinge Loss: max(0, 1 - y * z)
        margin = 1.0 - (targets * predictions)
        return torch.clamp(margin, min=0.0).mean()

    def z_update(self, v: torch.Tensor, targets: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Proximal operator for Hinge Loss.
        Evaluates the piecewise conditions of the max-margin boundary.
        """
        # Calculate the margin using the 'v' state
        y_v = targets * v
        
        # Piecewise analytical solution for Hinge proximal operator
        # Condition 1: Well outside the margin (No penalty)
        cond1 = y_v >= 1.0
        
        # Condition 2: Inside the margin penalty zone
        cond2 = y_v <= (1.0 - 1.0 / rho)
        
        # Condition 3: On the margin boundary
        # (This is the default 'else' state, where z_tilde = 1)
        
        # Build the intermediate z_tilde tensor
        z_tilde = torch.where(
            cond1, y_v, 
            torch.where(cond2, y_v + 1.0 / rho, torch.ones_like(y_v))
        )
        
        # Multiply back by the target signs to get the true z update
        return targets * z_tilde
