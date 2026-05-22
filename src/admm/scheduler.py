r"""This module dynamically manages the optimization hyper-parameters ($\rho$ and $\beta$)
during training to maintain a healthy balance between Primal and Dual residuals,
preventing algorithm stagnation.

IMPORTANT:
    - The implementation of this module is based on the book,
    [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf),
    however due to not filling the convex nature required for optimality implementing
    alternative versions of shedulers (dynamic hyper-parameters) is on our roadmap.
    - This module only works for single-batch for now.
"""

import warnings

import torch


class ADMM_Scheduler:
    r"""Dynamically balances Primal and Dual residuals by scaling $\rho$ and $\beta$.

    Mimics the version described in [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf),
    but has been upgraded for Deep Learning: Uses frequency control and a stop-epoch to
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

        # State placeholder
        self.state_old = None

    def capture_state(self):
        """Snapshots the network auxiliary state BEFORE the epoch starts.

        This is required to compute the dual residuals (change in states over time).
        """
        if not self.model.state_handler.in_memory:
            raise RuntimeError(
                "ADMM_Scheduler currently only supports single-batch training. "
                "The state_handler detected multiple batches (in_memory=False). "
                "Please reduce your dataset or increase your batch size to encompass the entire dataset."
            )

        _, _, batch_state = self.model.state_handler.load_batch(0)

        self.state_old = []
        for l_state in batch_state.layer_states:
            z_old = l_state.z.detach().clone()
            a_old = l_state.a.detach().clone() if l_state.a is not None else None
            self.state_old.append({"z": z_old, "a": a_old})

    def step(
        self,
        primal_rho_list: list,
        primal_beta_list: list,
    ):
        r"""Called AFTER the epoch ends to compute duals and balance $\rho$ and $\beta$ penalties.

        Evaluates the normalized dual residuals ($d_\rho$ and $d_\beta$) and scales the 
        penalty parameters if the Primal and Dual residuals fall out of the tolerance ratio ($\mu$).

        **$\rho$ Update:**
        $$
        \rho_l^{k+1} = 
        \begin{cases} 
        \tau \rho_l^k & \text{if } p_{\rho_l}^k > \mu d_{\rho_l}^k \\\\
        \rho_l^k / \tau & \text{if } d_{\rho_l}^k > \mu p_{\rho_l}^k \\\\
        \rho_l^k & \text{otherwise}
        \end{cases}
        $$
        *Where:* $d_{\rho_l}^k = \rho_l^k \frac{\|z_l^{k} - z_l^{k-1}\|_2}{\sqrt{N_z}}$

        **Lagrange Multiplier ($\lambda$) Scaling:**
        *(If enabled via `use_lagrange`)*
        $$
        \lambda_l^{k+1} = 
        \begin{cases} 
        \lambda_l^k / \tau & \text{if } p_{\rho_l}^k > \mu d_{\rho_l}^k \\\\
        \tau \lambda_l^k & \text{if } d_{\rho_l}^k > \mu p_{\rho_l}^k \\\\
        \lambda_l^k & \text{otherwise}
        \end{cases}
        $$
        
        **$\beta$ Update:**
        $$
        \beta_l^{k+1} = 
        \begin{cases} 
        \tau \beta_l^k & \text{if } p_{\beta_l}^k > \mu d_{\beta_l}^k \\\\
        \beta_l^k / \tau & \text{if } d_{\beta_l}^k > \mu p_{\beta_l}^k \\\\
        \beta_l^k & \text{otherwise}
        \end{cases}
        $$
        *Where:* $d_{\beta_l}^k = \beta_l^k \frac{\|a_l^{k} - a_l^{k-1}\|_2}{\sqrt{N_a}}$

        Args:
            primal_rho_list (list of float): The pre-activation constraint norms.
            primal_beta_list (list of float): The activation constraint norms.
        """
        self.current_epoch += 1

        if self.state_old is None:
            warnings.warn(
                "capture_state() must be called before step(). Skipping balance.",
                UserWarning,
            )
            return

        # Epoch control
        if (self.current_epoch % self.balance_freq != 0) or (
            self.current_epoch > self.stop_epoch
        ):
            self.state_old = None
            return

        _, _, batch_state = self.model.state_handler.load_batch(0)

        # Balance rho and lambda
        for i, (layer, l_state, p_rho) in enumerate(
            zip(self.model.layers, batch_state.layer_states, primal_rho_list)
        ):
            z_new = l_state.z.detach()
            z_old = self.state_old[i]["z"]

            norm_factor = z_new.numel() ** 0.5  # sqrt(N_z)
            d_rho = layer.config.rho * (torch.norm(z_new - z_old).item() / norm_factor)

            if p_rho > self.mu * d_rho:
                layer.config.rho *= self.tau
                if (
                    getattr(layer.config, "use_lagrange", False)
                    and l_state.lambda_lagrange is not None
                ):
                    l_state.lambda_lagrange /= self.tau

            elif d_rho > self.mu * p_rho:
                layer.config.rho /= self.tau
                if (
                    getattr(layer.config, "use_lagrange", False)
                    and l_state.lambda_lagrange is not None
                ):
                    l_state.lambda_lagrange *= self.tau

        # 2. Balance beta
        for i, (layer, p_beta) in enumerate(
            zip(self.model.layers[:-1], primal_beta_list)
        ):
            a_new = batch_state.layer_states[i].a
            if a_new is None:
                continue

            a_new = a_new.detach()
            a_old = self.state_old[i]["a"]

            norm_factor = a_new.numel() ** 0.5  # sqrt(N_a)
            d_beta = layer.config.beta * (
                torch.norm(a_new - a_old).item() / norm_factor
            )

            if p_beta > self.mu * d_beta:
                layer.config.beta *= self.tau
            elif d_beta > self.mu * p_beta:
                layer.config.beta /= self.tau

        # Free memory immediately
        self.state_old = None
