r"""This module dynamically manages the optimization hyper-parameters ($\rho$ and $\beta$)
during training to maintain a healthy balance between Primal and Dual residuals,
preventing algorithm stagnation.
"""

import torch


class ADMM_Scheduler:
    r"""Dynamically balances Primal and Dual residuals by scaling $\rho$ and $\beta$.

    Upgraded for Deep Learning: Uses frequency control and a stop-epoch to
    prevent 'Ping-Pong' oscillation in non-convex landscapes.

    Attributes:
        model (nn.Module): The ADMM manager/network being optimized.
        mu (float): The tolerance threshold ratio between primal and dual residuals.
        tau (float): The scaling multiplier applied to $\rho$ and $\beta$ when unbalanced.
        balance_freq (int): Number of epochs to wait between balancing operations.
        stop_epoch (int): The epoch at which to permanently cease balancing to allow fine-tuning.
    """

    def __init__(
        self,
        model,
        mu: float = 10.0,
        tau: float = 2.0,
        balance_freq: int = 5,
        stop_epoch: int = 150,
    ):
        """Initializes the scheduler controls.

        Args:
            model (nn.Module): The ADMM manager instance.
            mu (float, optional): Balance ratio threshold. Defaults to 10.0.
            tau (float, optional): Scaling multiplier. Defaults to 2.0.
            balance_freq (int, optional): Frequency of balancing. Defaults to 5.
            stop_epoch (int, optional): Epoch to halt scheduling. Defaults to 150.
        """
        self.model = model
        self.mu = mu
        self.tau = tau

        # Deep Learning Controls
        self.balance_freq = balance_freq
        self.stop_epoch = stop_epoch
        self.current_epoch = 0

        # State placeholders
        self.z_old = None
        self.a_old = None

    def capture_state(self):
        """Snapshots the network auxiliary state BEFORE the epoch starts.

        This is required to compute the dual residuals (change in states over time).
        """
        self.z_old = [
            layer.z.detach().clone()
            for layer in self.model.layers
            if hasattr(layer, "z")
        ]
        self.a_old = [
            layer.a.detach().clone()
            for layer in self.model.layers[:-1]
            if hasattr(layer, "a")
        ]

    def step(self, primal_rho_list: list, primal_beta_list: list):
        r"""Called AFTER the epoch ends to compute duals and balance $\rho$ and $\beta$ penalties.

        Args:
            primal_rho_list (list of float): The pre-activation constraint norms.
            primal_beta_list (list of float): The activation constraint norms.
        """
        self.current_epoch += 1

        if self.z_old is None or self.a_old is None:
            print(
                "Warning: capture_state() must be called before step(). Skipping balance."
            )
            return

        # =========================================================
        # Deep Learning Control: Only balance every N epochs, and
        # stop balancing near the end of training to allow fine-tuning!
        # =========================================================
        if (self.current_epoch % self.balance_freq != 0) or (
            self.current_epoch > self.stop_epoch
        ):
            self.z_old = None
            self.a_old = None
            return

        # 1. Compute NORMALIZED Dual Residuals on the fly
        z_new = [layer.z.detach() for layer in self.model.layers if hasattr(layer, "z")]
        a_new = [
            layer.a.detach() for layer in self.model.layers[:-1] if hasattr(layer, "a")
        ]

        # ---------------------------------------------------------
        # 2. Balance RHO per layer (Applied to ALL layers)
        # ---------------------------------------------------------
        for layer, zn, zo, p_rho in zip(
            self.model.layers, z_new, self.z_old, primal_rho_list
        ):
            norm_factor = zn.numel() ** 0.5
            d_rho = layer.config.rho * (torch.norm(zn - zo).item() / norm_factor)

            if p_rho > self.mu * d_rho:
                layer.config.rho *= self.tau

                # --- NEW: Safe Layer-wise Lagrange Scaling ---
                if (
                    getattr(layer.config, "use_lagrange", False)
                    and getattr(layer, "lambda_lagrange", None) is not None
                ):
                    layer.lambda_lagrange /= self.tau

            elif d_rho > self.mu * p_rho:
                layer.config.rho /= self.tau

                # --- NEW: Safe Layer-wise Lagrange Scaling ---
                if (
                    getattr(layer.config, "use_lagrange", False)
                    and getattr(layer, "lambda_lagrange", None) is not None
                ):
                    layer.lambda_lagrange *= self.tau

        # ---------------------------------------------------------
        # 3. Balance BETA per layer (Applied to HIDDEN layers only)
        # ---------------------------------------------------------
        for layer, an, ao, p_beta in zip(
            self.model.layers[:-1], a_new, self.a_old, primal_beta_list
        ):
            norm_factor = an.numel() ** 0.5
            d_beta = layer.config.beta * (torch.norm(an - ao).item() / norm_factor)

            if p_beta > self.mu * d_beta:
                layer.config.beta *= self.tau
            elif d_beta > self.mu * p_beta:
                layer.config.beta /= self.tau

        # Free memory immediately
        self.z_old = None
        self.a_old = None
