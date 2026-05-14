r"""
This module defines the objective functions for the ADMM network.

Each loss function provides the loss calculation for logging and metrics,
a the analytical $z$-update formula required
to solve the final layer's ADMM subproblem.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataclasses import ADMM_LayerConfig, ADMM_LayerState
from .functional.utils import broadcast_to_match, compute_temporal_dependencies


class ADMM_Loss(nn.Module, ABC):
    """Abstract Base Class for ADMM Loss Functions."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the loss for logging and metrics.

        Args:
            predictions (torch.Tensor): The network's final output.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        pass

    @abstractmethod
    def _loss_update(
        self,
        temporal_forward_T: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> torch.Tensor:
        """The loss-specific update.

        Args:
            state (ADMM_LayerState): The state object for the final layer.
            forward (torch.Tensor): The precomputed spatial transformation.

            temporal_forward_T: torch.Tensor,
            labels (torch.Tensor): The ground truth target labels.
            lambda_lagrange(torch.Tensor),
            config (ADMM_LayerConfig): The configuration parameters for the final layer.

        Returns:
            torch.Tensor: The new $z_last$ updated tensor.
        """
        pass

    def update_z_last(
        self,
        state: ADMM_LayerState,
        forward: torch.Tensor,
        labels: torch.Tensor,
        config: ADMM_LayerConfig,
        spiking: bool,
    ) -> None:
        """Routes the final layer's pre-activation ($z$) update based on network type."""
        if spiking:
            self.update_z_last_spiking(
                state=state,
                forward=forward,
                labels=labels,
                config=config,
            )
        else:
            self.update_z_last_static(
                state=state, forward=forward, labels=labels, config=config
            )

    def update_z_last_static(
        self,
        state: ADMM_LayerState,
        forward: torch.Tensor,
        labels: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> None:
        """Applies the vectorized $z$ update for static networks.

        Args:
            state (ADMM_LayerState): The state object for the final layer.
            forward (torch.Tensor): The precomputed spatial transformation.
            labels (torch.Tensor): The ground truth target labels.
            config (ADMM_LayerConfig): The configuration parameters for the final layer.
        """
        labels = broadcast_to_match(labels, forward)
        if config.use_lagrange and state.lambda_lagrange is not None:
            lambda_lagrange = broadcast_to_match(state.lambda_lagrange, forward)
        else:
            lambda_lagrange = torch.zeros_like(forward)
        state.z.copy_(self._loss_update(forward, labels, lambda_lagrange, config))

    def update_z_last_spiking(
        self,
        state: ADMM_LayerState,
        forward: torch.Tensor,
        labels: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> None:
        """Applies the $z$ update to spiking neurons using the [loss function operators][admm.loss_functions].

        You can find the corresponding unrolled and non-spiking versions [update_z_last_unrolled][admm.spiking_mixin.ADMM_Spiking.update_z_last_unrolled] and [update_z_last][admm.core.ADMM_Layer.update_z_last] in the [Core][admm.core] and [Spiking Mixin][admm.spiking_mixin] modules.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth target labels.
            loss_f (ADMM_Loss, optional): The objective function managing the update. Defaults to None.
        """
        temporal_forward = forward + (compute_temporal_dependencies(state, config))

        shape = state.z[-1]
        labels = broadcast_to_match(labels, shape)
        if config.use_lagrange and state.lambda_lagrange is not None:
            lambda_lagrange = broadcast_to_match(state.lambda_lagrange, shape)
        else:
            lambda_lagrange = torch.zeros_like(shape)

        numerator = temporal_forward.clone().mul_(config.rho)
        z_minus_fwd = state.z.clone().sub_(forward)

        numerator[:-1].add_(z_minus_fwd[1:], alpha=config.rho * config.deltas)
        numerator[-2].add_(lambda_lagrange, alpha=config.deltas)

        denominator_main = config.rho * (config.deltas**2) + config.rho
        numerator[:-1].div_(denominator_main)

        numerator[-1] = self._loss_update(
            temporal_forward[-1], labels, lambda_lagrange, config
        )

        state.z.copy_(numerator)

    def update_z_last_unrolled(
        self,
        state: ADMM_LayerState,
        forward: torch.Tensor,
        labels: torch.Tensor,
        config: ADMM_LayerConfig,
        time_steps: list,
    ) -> None:
        """Applies the $z$ update to spiking neurons in an unrolled manner using the [loss function operators][admm.loss_functions].

        You can find the corresponding spikingand non-spiking versions [update_z_last][admm.spiking_mixin.ADMM_Spiking.update_z_last] and [update_z_last][admm.core.ADMM_Layer.update_z_last] in the [Core][admm.core] and [Spiking Mixin][admm.spiking_mixin] modules.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.
            labels (torch.Tensor): The ground truth target labels.
            time_steps (list): The list of sequence time steps to update.
            jacobi (bool): Flag to determine if the unrolled update should use Jacobi-style updates. Defaults to False.
            loss_f (ADMM_Loss, optional): The objective function managing the update. Defaults to None.
        """
        labels = broadcast_to_match(labels, state.z[-1])
        if config.use_lagrange and state.lambda_lagrange is not None:
            lambda_lagrange = broadcast_to_match(state.lambda_lagrange, state.z[-1])
        else:
            lambda_lagrange = torch.zeros_like(state.z[-1])
        T = state.z.size(0)
        denominator_main = (config.rho * config.deltas**2) + config.rho
        buffer = torch.empty_like(state.z[0])

        for t in time_steps:
            if t < T - 1:
                buffer.copy_(forward[t])
                if t >= 1:
                    buffer.add_(state.z[t - 1], alpha=config.deltas)
                buffer.add_(state.z[t + 1], alpha=config.deltas)
                buffer.add_(forward[t + 1], alpha=-config.deltas)
                buffer.mul_(config.rho)
                if t == T - 2:
                    buffer.add_(lambda_lagrange, alpha=config.deltas)
                buffer.div_(denominator_main)
                state.z[t].copy_(buffer)

            else:
                buffer.copy_(forward[t])
                if t >= 1:
                    buffer.add_(state.z[t - 1], alpha=config.deltas)
                state.z[t].copy_(
                    self._loss_update(buffer, labels, lambda_lagrange, config)
                )


class ADMM_SSE(ADMM_Loss):
    """
    Sum of Squared Errors Loss for ADMM.
    """

    def __str__(self):
        return "SSE_Loss"

    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets, reduction="sum")

    def _loss_update(self, temporal_forward_T, labels, lambda_lagrange, config):
        # (rho * v + 2y - lambda) / (2 + rho)
        num = (
            temporal_forward_T.mul(config.rho)
            .add_(labels, alpha=2.0)
            .sub_(lambda_lagrange)
        )
        return num.div_(2.0 + config.rho)


class ADMM_Hinge(ADMM_Loss):
    """
    Hinge Loss for ADMM.
    """

    def __str__(self):
        return "Hinge_Loss"

    def _format_labels(self, targets):
        return 2.0 * targets - 1.0 if targets.min() == 0.0 else targets

    def forward(self, predictions, targets):
        y = self._format_labels(targets)
        return torch.clamp(1.0 - (y * predictions), min=0.0).sum()

    def _loss_update(self, temporal_forward_T, labels, lambda_lagrange, config):
        y = self._format_labels(labels)
        v_T = temporal_forward_T.sub(lambda_lagrange / config.rho)
        cond = y * v_T
        z_tilde = torch.where(
            cond >= 1.0,
            cond,
            torch.where(
                cond <= (1.0 - 1.0 / config.rho),
                cond + (1.0 / config.rho),
                torch.ones_like(cond),
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

    def forward(self, predictions, targets):
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(predictions, targets.long().view(-1), reduction="sum")

    def _loss_update(self, temporal_forward_T, labels, lambda_lagrange, config):
        y_one_hot = self._ensure_one_hot(labels, temporal_forward_T.size(-1))
        p = F.softmax(temporal_forward_T, dim=-1)
        grad_ce = p - y_one_hot
        return temporal_forward_T.sub(lambda_lagrange + grad_ce, alpha=1.0 / config.rho)


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

    def forward(self, predictions, targets):
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(predictions, targets.long().view(-1), reduction="sum")

    def _loss_update(
        self, temporal_forward_T, labels, lambda_lagrange, config, max_iter=15, tol=1e-5
    ):
        y_one_hot = self._ensure_one_hot(labels, temporal_forward_T.size(-1))

        # The target 'v' that the ADMM consensus wants us to reach
        v = temporal_forward_T.sub(lambda_lagrange / config.rho)

        # Start our guess using the forward pass (usually very close to the answer)
        z = temporal_forward_T.clone()

        # Identity matrix for the Hessian (shape: [1, Classes, Classes])
        Identity = torch.eye(z.size(-1), device=z.device, dtype=z.dtype).unsqueeze(0)

        for i in range(max_iter):
            p = F.softmax(z, dim=-1)

            # 1. First Derivative (Gradient): g(z) = p - y + rho * (z - v)
            g = p - y_one_hot + config.rho * (z - v)

            # Check if we have converged to the exact answer
            if torch.max(torch.abs(g)) < tol:
                break

            # 2. Second Derivative (Hessian matrix): H(z) = rho*I + diag(p) - p*p^T
            p_diag = torch.diag_embed(p)  # Shape: [Batch, Classes, Classes]
            p_outer = torch.bmm(
                p.unsqueeze(2), p.unsqueeze(1)
            )  # Shape: [Batch, Classes, Classes]
            H = config.rho * Identity + p_diag - p_outer

            # 3. Newton-Raphson Step: z_new = z_old - H^{-1} * g
            # We use torch.linalg.solve to compute H^{-1} * g safely and fully vectorized
            delta = torch.linalg.solve(H, g.unsqueeze(2)).squeeze(2)
            z = z - delta

        return z
