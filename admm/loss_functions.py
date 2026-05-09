"""
This module defines the objective functions for the ADMM network.
Each loss function provides both a scalar calculation for metrics,
and the analytical ['update_z'][admm.affine.ADMM_AffineLayer.update_z] formula to solve the final layer's ADMM subproblem.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class ADMM_Loss(ABC):
    """Abstract Base Class for ADMM Loss Functions."""

    @abstractmethod
    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the scalar loss for logging and metrics.
        """
        pass

    def update_z_last_core(self, forward, rho, labels, lambda_lagrange):
        """Non-spiking (static) proximal update."""
        return self._last_timestep_update(forward, labels, lambda_lagrange, rho)

    @abstractmethod
    def _last_timestep_update(
        self,
        temporal_forward_T: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        """The loss-specific calculation for the final time step T."""
        pass

    def update_z_last_spiking(
        self,
        forward: torch.Tensor,
        temporal_forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        z: torch.Tensor,
        rho: float,
        deltas: float,
    ):
        """
        Vectorized update shared by all loss functions.
        Handles temporal logic for t < T and calls the subclass for t = T.
        """
        numerator = temporal_forward.clone().mul_(rho)
        z_minus_fwd = z.clone().sub_(forward)

        numerator[:-1].add_(z_minus_fwd[1:], alpha=rho * deltas)
        numerator[-2].add_(lambda_lagrange, alpha=deltas)

        denominator_main = rho * (deltas**2) + rho
        numerator[:-1].div_(denominator_main)

        numerator[-1] = self._last_timestep_update(
            temporal_forward[-1], labels, lambda_lagrange, rho
        )
        return numerator

    def update_z_last_unrolled_spiking(
        self,
        forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        z: torch.Tensor,
        rho: float,
        deltas: float,
        time_steps: list,
        jacobi: bool = False,
    ):
        """
        Unrolled loop shared by all loss functions.
        """
        T = z.size(0)
        denominator_main = (rho * deltas**2) + rho
        z_to_use = z.clone() if jacobi else z
        buffer = torch.empty_like(z[0])

        for t in time_steps:
            if t < T - 1:
                buffer.copy_(forward[t])
                if t >= 1:
                    buffer.add_(z_to_use[t - 1], alpha=deltas)
                buffer.add_(z_to_use[t + 1], alpha=deltas)
                buffer.add_(forward[t + 1], alpha=-deltas)
                buffer.mul_(rho)
                if t == T - 2:
                    buffer.add_(lambda_lagrange, alpha=deltas)
                buffer.div_(denominator_main)
                z[t].copy_(buffer)

            else:
                buffer.copy_(forward[t])
                if t >= 1:
                    buffer.add_(z_to_use[t - 1], alpha=deltas)
                z[t].copy_(
                    self._last_timestep_update(buffer, labels, lambda_lagrange, rho)
                )

        if jacobi:
            del z_to_use
        return z


class ADMM_SSE(ADMM_Loss):
    """
    Sum of Squared Errors Loss for ADMM.
    """

    def __str__(self):
        return "SSE_Loss"

    def __call__(self, predictions, targets):
        return F.mse_loss(predictions, targets, reduction="sum")

    def _last_timestep_update(self, temporal_forward_T, labels, lambda_lagrange, rho):
        # (rho * v + 2y - lambda) / (2 + rho)
        num = temporal_forward_T.mul(rho).add_(labels, alpha=2.0).sub_(lambda_lagrange)
        return num.div_(2.0 + rho)


class ADMM_Hinge(ADMM_Loss):
    """
    Hinge Loss for ADMM.
    """

    def __str__(self):
        return "Hinge_Loss"

    def _format_labels(self, targets):
        return 2.0 * targets - 1.0 if targets.min() == 0.0 else targets

    def __call__(self, predictions, targets):
        y = self._format_labels(targets)
        return torch.clamp(1.0 - (y * predictions), min=0.0).sum()

    def _last_timestep_update(self, temporal_forward_T, labels, lambda_lagrange, rho):
        y = self._format_labels(labels)
        v_T = temporal_forward_T.sub(lambda_lagrange / rho)
        cond = y * v_T
        z_tilde = torch.where(
            cond >= 1.0,
            cond,
            torch.where(
                cond <= (1.0 - 1.0 / rho), cond + (1.0 / rho), torch.ones_like(cond)
            ),
        )
        return y * z_tilde


class ADMM_CrossEntropy_Taylor(ADMM_Loss):
    """
    Cross-Entropy Loss for ADMM.
    Uses Taylor expansion to derive a CLOSE-TO-OPTIMAL analytical minimizer for the non-linear z-update.
    """

    def __str__(self):
        return "CrossEntropy_Taylor_Loss"

    def _ensure_one_hot(self, labels, num_classes):
        if labels.dim() == 1 or labels.size(1) == 1:
            return F.one_hot(labels.view(-1).long(), num_classes=num_classes).to(
                torch.float32
            )
        return labels.to(torch.float32)

    def __call__(self, predictions, targets):
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(predictions, targets.long().view(-1), reduction="sum")

    def _last_timestep_update(self, temporal_forward_T, labels, lambda_lagrange, rho):
        y_one_hot = self._ensure_one_hot(labels, temporal_forward_T.size(-1))
        p = F.softmax(temporal_forward_T, dim=-1)
        grad_ce = p - y_one_hot
        return temporal_forward_T.sub(lambda_lagrange + grad_ce, alpha=1.0 / rho)


class ADMM_CrossEntropy(ADMM_Loss):
    """
    Cross-Entropy Loss for ADMM.
    Uses the Newton-Raphson method to find the EXACT analytical minimizer
    for the non-linear z-update.
    """

    def __str__(self):
        return "CrossEntropy_Loss"

    def _ensure_one_hot(self, labels, num_classes):
        if labels.dim() == 1 or labels.size(1) == 1:
            return F.one_hot(labels.view(-1).long(), num_classes=num_classes).to(
                torch.float32
            )
        return labels.to(torch.float32)

    def __call__(self, predictions, targets):
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(predictions, targets.long().view(-1), reduction="sum")

    def _last_timestep_update(
        self, temporal_forward_T, labels, lambda_lagrange, rho, max_iter=15, tol=1e-5
    ):
        y_one_hot = self._ensure_one_hot(labels, temporal_forward_T.size(-1))

        # The target 'v' that the ADMM consensus wants us to reach
        v = temporal_forward_T.sub(lambda_lagrange / rho)

        # Start our guess using the forward pass (usually very close to the answer)
        z = temporal_forward_T.clone()

        # Identity matrix for the Hessian (shape: [1, Classes, Classes])
        I = torch.eye(z.size(-1), device=z.device, dtype=z.dtype).unsqueeze(0)

        for i in range(max_iter):
            p = F.softmax(z, dim=-1)

            # 1. First Derivative (Gradient): g(z) = p - y + rho * (z - v)
            g = p - y_one_hot + rho * (z - v)

            # Check if we have converged to the exact answer
            if torch.max(torch.abs(g)) < tol:
                break

            # 2. Second Derivative (Hessian matrix): H(z) = rho*I + diag(p) - p*p^T
            p_diag = torch.diag_embed(p)  # Shape: [Batch, Classes, Classes]
            p_outer = torch.bmm(
                p.unsqueeze(2), p.unsqueeze(1)
            )  # Shape: [Batch, Classes, Classes]
            H = rho * I + p_diag - p_outer

            # 3. Newton-Raphson Step: z_new = z_old - H^{-1} * g
            # We use torch.linalg.solve to compute H^{-1} * g safely and fully vectorized
            delta = torch.linalg.solve(H, g.unsqueeze(2)).squeeze(2)
            z = z - delta

        return z
