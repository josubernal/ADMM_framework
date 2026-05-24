"""This module defines the objective functions for the ADMM network.

Each loss function provides the loss calculation for logging and metrics,
and the analytical z-update formula required to solve the final layer's
ADMM subproblem.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataclasses import ADMM_LayerConfig, ADMM_LayerState
from .functional.utils import broadcast_to_match, compute_temporal_dependencies


class ADMM_Loss(nn.Module, ABC):
    """Abstract Base Class for ADMM Loss Functions.

    Provides the structural template for routing final-layer pre-activation (z)
    updates for both static and spiking networks, while requiring subclasses to
    define the specific analytical minimizers.
    """

    def __init__(self) -> None:
        super().__init__()

    def _format_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Centrally formats integer labels into one-hot tensors to prevent broadcast corruption."""
        if labels.dim() == 1 or labels.shape[-1] != num_classes:
            return F.one_hot(labels.view(-1).long(), num_classes=num_classes).float()
        return labels.float()

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
        temporal_forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> torch.Tensor:
        """The loss-specific update.

        Args:
            temporal_forward: torch.Tensor,
            labels (torch.Tensor): The ground truth target labels.
            lambda_lagrange (torch.Tensor): Current dual variables.
            config (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] for the final layer.

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
        """Routes the final layer's pre-activation (z) update based on network type.

        Args:
            state (ADMM_LayerState): The [state object][src.admm.dataclasses.ADMM_LayerState] for the final layer.
            forward (torch.Tensor): The precomputed spatial transformation.
            labels (torch.Tensor): The ground truth target labels.
            config (ADMM_LayerConfig): The c[configuration object][src.admm.dataclasses.ADMM_LayerConfig] for the final layer.
            spiking (bool): Flag indicating if the network utilizes spiking dynamics.
        """
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
            state (ADMM_LayerState): The [state object][src.admm.dataclasses.ADMM_LayerState] for the final layer.
            forward (torch.Tensor): The precomputed spatial transformation.
            labels (torch.Tensor): The ground truth target labels.
            config (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] for the final layer.
        """
        labels_formatted = self._format_labels(labels, forward.size(-1))
        if config.use_lagrange and state.lambda_lagrange is not None:
            lambda_lagrange = broadcast_to_match(state.lambda_lagrange, forward)
        else:
            lambda_lagrange = torch.zeros_like(forward)
        state.z.copy_(
            self._loss_update(forward, labels_formatted, lambda_lagrange, config)
        )

    def update_z_last_spiking(
        self,
        state: ADMM_LayerState,
        forward: torch.Tensor,
        labels: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> None:
        """Applies the $z$ update to spiking neurons in a fully vectorized pass.

        Args:
            state (ADMM_LayerState): The [state object][src.admm.dataclasses.ADMM_LayerState] for the final layer.
            forward (torch.Tensor): The precomputed spatial transformation.
            labels (torch.Tensor): The ground truth target labels.
            config (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] for the final layer.
        """
        temporal_forward = forward + (compute_temporal_dependencies(state, config))

        shape = state.z[-1]
        labels_formatted = self._format_labels(labels, shape.size(-1))
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
            temporal_forward[-1], labels_formatted, lambda_lagrange, config
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
        """Applies the $z$ update to spiking neurons sequentially across time.

        Args:
            state (ADMM_LayerState): The [state object][src.admm.dataclasses.ADMM_LayerState] for the final layer.
            forward (torch.Tensor): The precomputed spatial transformation.
            labels (torch.Tensor): The ground truth target labels.
            config (ADMM_LayerConfig): The [configuration object][src.admm.dataclasses.ADMM_LayerConfig] for the final layer.
            time_steps (List[int]): The ordered sequence of time steps to update.
        """
        labels_formatted = self._format_labels(labels, state.z.size(-1))
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
                    self._loss_update(buffer, labels_formatted, lambda_lagrange, config)
                )


class ADMM_SSE(ADMM_Loss):
    """Sum of Squared Errors (SSE) Loss for ADMM.

    Provides the exact, closed-form analytical minimizer for regression and
    continuous target mapping tasks.
    """

    def __str__(self) -> str:
        return "SSE_Loss"

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the standard Mean Squared Error (summed over the batch)."""
        if targets.dim() == 1 or targets.shape != predictions.shape:
            targets = F.one_hot(
                targets.view(-1).long(), num_classes=predictions.shape[-1]
            ).float()
        return F.mse_loss(predictions, targets, reduction="sum")

    def _loss_update(
        self,
        temporal_forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> torch.Tensor:
        r"""Computes the SSE specific update.

        Formula:

        $$ (\rho_l  v + 2y - \lambda_l) / (2 + \rho_l) $$
        """
        num = (
            temporal_forward.mul(config.rho)
            .add_(labels, alpha=2.0)
            .sub_(lambda_lagrange)
        )
        return num.div_(2.0 + config.rho)


class ADMM_Hinge(ADMM_Loss):
    """Hinge Loss for ADMM.

    Evaluates maximum-margin classifications. Uses a piecewise conditional
    formula to strictly enforce margin bounds during the update.
    """

    def __str__(self) -> str:
        return "Hinge_Loss"

    def _apply_hinge_bounds(self, targets: torch.Tensor) -> torch.Tensor:
        """Converts [0, 1] labels to [-1, 1] format required for Hinge bounds."""
        return 2.0 * targets - 1.0 if targets.min() == 0.0 else targets

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = self._format_labels(targets, predictions.shape[-1])
        y = self._apply_hinge_bounds(y)
        return torch.clamp(1.0 - (y * predictions), min=0.0).sum()

    def _loss_update(
        self,
        temporal_forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> torch.Tensor:
        """Computes the piecewise Hinge specific update."""
        y = self._apply_hinge_bounds(labels)
        v_T = temporal_forward.sub(lambda_lagrange / config.rho)
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
    """Taylor-expanded Cross-Entropy Loss for ADMM.

    Uses a first-order Taylor expansion to derive a computationally lightweight,
    close-to-optimal analytical minimizer for the non-linear z-update. Highly
    recommended for large, multi-class architectures.
    """

    def __str__(self) -> str:
        return "CrossEntropy_Taylor_Loss"

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(predictions, targets.long().view(-1), reduction="sum")

    def _loss_update(
        self,
        temporal_forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        config: ADMM_LayerConfig,
    ) -> torch.Tensor:
        """Computes the linear Taylor-approximated CE update."""
        p = F.softmax(temporal_forward, dim=-1)
        grad_ce = p - labels
        return temporal_forward.sub(lambda_lagrange + grad_ce, alpha=1.0 / config.rho)


class ADMM_CrossEntropy(ADMM_Loss):
    """Exact Cross-Entropy Loss for ADMM.

    Uses an iterative Newton-Raphson method to find the exact, non-linear
    analytical minimizer. More computationally expensive than the Taylor
    expansion, but guarantees maximum mathematical precision.
    """

    def __str__(self) -> str:
        return "CrossEntropy_Loss"

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the standard Cross-Entropy metric."""
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = torch.argmax(targets, dim=1)
        return F.cross_entropy(predictions, targets.long().view(-1), reduction="sum")

    def _loss_update(
        self,
        temporal_forward: torch.Tensor,
        labels: torch.Tensor,
        lambda_lagrange: torch.Tensor,
        config: ADMM_LayerConfig,
        max_iter: int = 15,
        tol: float = 1e-5,
    ) -> torch.Tensor:
        """Computes the exact CE proximal update using Newton-Raphson.

        Args:
            temporal_forward (torch.Tensor): Precomputed transformation.
            labels (torch.Tensor): Ground truth labels.
            lambda_lagrange (torch.Tensor): Current dual variables.
            config (ADMM_LayerConfig): [Configuration object][src.admm.dataclasses.ADMM_LayerConfig] of the layer.
            max_iter (int, optional): Maximum Newton steps. Defaults to 15.
            tol (float, optional): Tolerance for gradient convergence. Defaults to 1e-5.

        Returns:
            torch.Tensor: The iteratively solved z_last tensor.
        """
        # The target 'v' that the ADMM consensus wants us to reach
        v = temporal_forward.sub(lambda_lagrange / config.rho)

        # Start our guess using the forward pass (usually very close to the answer)
        z = temporal_forward.clone()

        # Identity matrix for the Hessian (shape: [1, Classes, Classes])
        Identity = torch.eye(z.size(-1), device=z.device, dtype=z.dtype).unsqueeze(0)

        for i in range(max_iter):
            p = F.softmax(z, dim=-1)

            # 1. First Derivative (Gradient): g(z) = p - y + rho * (z - v)
            g = p - labels + config.rho * (z - v)

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
