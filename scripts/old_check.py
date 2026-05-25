import random
from typing import List

import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

from src.admm import (
    ADMM,
    ADMM_SSE,
    ADMM_Config,
    ADMM_Heaviside,
    ADMM_LayerConfig,
    ADMM_Metrics,
    ADMM_SpikingFeedForward,
)


class ADMM_SNN:
    """Class for ADMM Neural Network."""

    def __init__(
        self,
        n_samples: int,
        n_timesteps: int,
        input_dim: int,
        hidden_dims: List[int],
        n_outputs: int,
        rho: float,
        deltas: torch.Tensor,
        thetas: torch.Tensor,
        beta: float,
    ):
        """
        Initializes the ADMM Spiking Neural Network model.

        Args:
            n_samples (int): Number of samples in a batch.
            n_timesteps (int): Number of simulation timesteps.
            input_dim (int): Dimensionality of the input layer.
            hidden_dims (List[int]): List containing the dimensions of the hidden layers.
            n_outputs (int): Dimensionality of the output layer.
            rho (float): Penalty parameter for the ADMM algorithm.
            deltas (torch.Tensor): Decay factors for the LIF neurons.
            thetas (torch.Tensor): Firing thresholds for the LIF neurons.
            beta (float): Regularization parameter for activation updates.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rho = rho
        self.deltas = deltas
        self.thetas = thetas
        self.beta = beta
        self.pinv_l_0 = None  # Attribute to store the pseudoinverse at layer 0

        self.L = len(hidden_dims) + 1  # Total number of layers (including output)
        self.T = n_timesteps  # Number of timesteps

        # === Initialize W_l (Weights) ===
        self.W = []
        for i, hidden_dim in enumerate(hidden_dims + [n_outputs]):
            if i == 0:
                self.W.append(torch.zeros((hidden_dim, input_dim)).to(self.device))
            else:
                self.W.append(
                    torch.zeros((hidden_dim, hidden_dims[i - 1])).to(self.device)
                )

        # === Initialize z_l (Pre-activation potentials) ===
        self.z = []
        for i, hidden_dim in enumerate(hidden_dims + [n_outputs]):
            self.z.append(
                # torch.rand((n_timesteps, n_samples, hidden_dim)).to(self.device)
                # CHANGE
                torch.zeros((n_timesteps, n_samples, hidden_dim)).to(self.device)
            )

        # === Initialize a_l (Activations/Spikes) ===
        self.a = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.a.append(
                # torch.rand((n_timesteps, n_samples, hidden_dim)).to(self.device)
                # CHANGE
                torch.zeros((n_timesteps, n_samples, hidden_dim)).to(self.device)
            )

        # === Initialize lagrange multipliers (only for the output layer constraint) ===
        self.lambda_lagrange = torch.zeros((n_samples, n_outputs)).to(self.device)

    def _heaviside(self, x):
        """
        Applies the Heaviside step function element-wise.

        Returns 1 if x > threshold, 0 otherwise.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with Heaviside function applied.
        """
        return (x > self.thetas).float()

    def _weight_update(self, x_l, a_lminus1, l):
        """
        Implements the weight update for layers 1 to L-1 (Equation 4).

        Args:
            x_l (torch.Tensor): Precomputed term related to z_l.
            a_lminus1 (torch.Tensor): Activations of the previous layer.

        Returns:
            torch.Tensor: Updated weight matrix W_l.
        """
        numerator = sum(x_l[t].T @ a_lminus1[t] for t in range(self.T))
        denominator = sum((a_lminus1[t].T @ a_lminus1[t]) for t in range(self.T))

        if l == 0:
            if self.pinv_l_0 is None:
                self.pinv_l_0 = torch.linalg.pinv(denominator)

            return numerator @ self.pinv_l_0

        return numerator @ torch.linalg.pinv(denominator)

    def _weight_update_L(self, x_L, a_Lminus1):
        """
        Implements the weight update for the output layer L (Equation 6).

        Includes the Lagrange multiplier term.

        Args:
            x_L (torch.Tensor): Precomputed term related to z_L.
            a_Lminus1 (torch.Tensor): Activations of the previous layer (L-1).

        Returns:
            torch.Tensor: Updated weight matrix W_L.
        """

        lambda_term = self.lambda_lagrange.T / self.rho
        numerator = lambda_term @ a_Lminus1[-1] + sum(
            x_L[t].T @ a_Lminus1[t] for t in range(self.T)
        )
        denominator = sum((a_Lminus1[t].T @ a_Lminus1[t]) for t in range(self.T))

        return numerator @ torch.linalg.pinv(denominator)

    def _z_update(self, q_l, r_l, a_l):
        """
        Implements the z_{l,t} update for hidden layers l = 1..L-1 and t < T (Equation 14).

        Args:
            q_l (torch.Tensor): Precomputed term q_{l,t}.
            r_l (torch.Tensor): Precomputed term r_{l,t+1}.
            a_l (torch.Tensor): Activation a_{l,t}.

        Returns:
            torch.Tensor: Updated pre-activation z_{l,t}.
        """
        return (q_l.T + self.deltas * (r_l.T + self.thetas * a_l)) / (
            1 + self.deltas**2
        )

    def _z_update_L(self, s_Lt, r_Ltnext, lam):
        """
        Implements the z_{L,t} update for the output layer l = L and t < T (Equation 16, t < T).

        Args:
            s_Lt (torch.Tensor): Precomputed term s_{L,t}.
            r_Ltnext (torch.Tensor): Precomputed term r_{L,t+1}.
            lam (torch.Tensor): Lagrange multiplier term (lambda/rho or 0).

        Returns:
            torch.Tensor: Updated pre-activation z_{L,t}.
        """
        return (s_Lt + self.deltas * r_Ltnext + self.deltas * lam / self.rho) / (
            1 + self.deltas**2
        )

    def _z_update_L_T(self, s_LT, y):
        """
        Implements the z_{L,T} update for the output layer l = L and t = T (Equation 16, t = T).

        Args:
            s_LT (torch.Tensor): Precomputed term s_{L,T}.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Updated pre-activation z_{L,T}.
        """
        return (self.rho * s_LT + 2 * y - self.lambda_lagrange.T) / (2 + self.rho)

    def check_entries(self, z, a_lt, q_lt, r_ltnext):
        """
        Implements Algorithm 1 for t < T.

        Args:
            z (torch.Tensor): Current estimate of z_{l,t}.
            a_lt (torch.Tensor): Activation a_{l,t}.
            q_lt (torch.Tensor): Precomputed term q_{l,t}.
            r_ltnext (torch.Tensor): Precomputed term r_{l,t+1}.

        Returns:
            torch.Tensor: Adjusted z_{l,t}.
        """
        delta1 = self.beta * (1 - 2 * a_lt)
        delta2 = (
            self.rho * (z - q_lt.T) ** 2
            + self.rho * (r_ltnext.T - self.deltas * z + self.thetas * a_lt) ** 2
            - self.rho * (self.thetas - q_lt.T) ** 2
            - self.rho
            * (r_ltnext.T - self.deltas * self.thetas + self.thetas * a_lt) ** 2
        )

        mask_z_greater = (z > self.thetas).float()
        mask_deltas1 = (delta1 + delta2 > 0).float()
        mask_deltas2 = (-delta1 + delta2 > 0).float()

        mask_intersection1 = mask_z_greater * mask_deltas1
        z[mask_intersection1 == 1] = self.thetas

        mask_intersection2 = (1 - mask_z_greater) * mask_deltas2
        z[mask_intersection2 == 1] = self.thetas + 1e-5

        return z

    def check_entries_T(self, z, a_lt, q_lt):
        """
        Implements Algorithm 1 variant for the final timestep T.

        Args:
            z (torch.Tensor): Current estimate of z_{l,T}.
            a_lt (torch.Tensor): Activation a_{l,T}.
            q_lt (torch.Tensor): Precomputed term q_{l,T}.

        Returns:
            torch.Tensor: Adjusted z_{l,T}.
        """
        delta1 = self.beta * (1 - 2 * a_lt).T
        delta2 = self.rho * ((z - q_lt) ** 2 - self.rho * (self.thetas - q_lt) ** 2)

        mask_z_greater = (z > self.thetas).float()
        mask_deltas1 = (delta1 + delta2 > 0).float()
        mask_deltas2 = (-delta1 + delta2 > 0).float()

        mask_intersection1 = mask_z_greater * mask_deltas1
        z[mask_intersection1 == 1] = self.thetas

        mask_intersection2 = (1 - mask_z_greater) * mask_deltas2
        z[mask_intersection2 == 1] = self.thetas + 1e-5

        return z

    def _activation_update(self, Wl_next, wl, zl, vl_next):
        """
        Implements the Activation update for l=1..L-2, t=1..T-1.

        Args:
            Wl_next (torch.Tensor): Weights of the next layer (l+1).
            wl (torch.Tensor): Precomputed term w_{l,t+1}.
            zl (torch.Tensor): Pre-activation z_{l,t}.
            vl_next (torch.Tensor): Precomputed term v_{l+1,t}.

        Returns:
            torch.Tensor: Updated activation a_{l,t}. Clipped between 0 and 1.
        """
        WtW = Wl_next.T @ Wl_next
        term1 = WtW + ((self.thetas**2) + (self.beta / self.rho)) * torch.eye(
            Wl_next.size(1)
        ).to(self.device)
        term2 = -self.thetas * wl.T
        term3 = (
            torch.mm(Wl_next.T, vl_next.T)
            + (self.beta / self.rho) * self._heaviside(zl).T
        )

        activation_update = torch.mm(torch.inverse(term1), (term2 + term3))

        return torch.clip(activation_update, 0, 1)

    def _activation_update_T(self, W_lnext, v_lnext, z_l):
        """
        Implements the Activation update for l=1..L-2, t=T.

        Args:
            W_lnext (torch.Tensor): Weights of the next layer (l+1).
            v_lnext (torch.Tensor): Precomputed term v_{l+1,T}.
            z_l (torch.Tensor): Pre-activation z_{l,T}.

        Returns:
            torch.Tensor: Updated activation a_{l,T}. Clipped between 0 and 1.
        """
        term1 = W_lnext.T @ W_lnext + (self.beta / self.rho) * torch.eye(
            W_lnext.size(1)
        ).to(self.device)
        a = torch.inverse(term1) @ (
            W_lnext.T @ v_lnext.T + (self.beta / self.rho) * self._heaviside(z_l).T
        )

        return torch.clip(a, 0, 1)

    def _activation_update_Lminus1(self, Wl_next, u_L, w_Lminus1, zl):
        """
        Implements the Activation update for l=L-1, t=1..T-1.

        Args:
            Wl_next (torch.Tensor): Weights of the next layer (L).
            u_L (torch.Tensor): Precomputed term u_{L,t}.
            w_Lminus1 (torch.Tensor): Precomputed term w_{L-1,t+1}.
            zl (torch.Tensor): Pre-activation z_{L-1,t}.

        Returns:
            torch.Tensor: Updated activation a_{L-1,t}. Clipped between 0 and 1.
        """
        WtW = Wl_next.T @ Wl_next
        term1 = WtW + ((self.thetas**2) + (self.beta / self.rho)) * torch.eye(
            Wl_next.size(1)
        ).to(self.device)
        term2 = (
            Wl_next.T @ u_L.T
            + (self.beta / self.rho) * self._heaviside(zl).T
            - self.thetas * w_Lminus1.T
        )

        activation_update_Lminus1 = torch.inverse(term1) @ term2

        return torch.clip(activation_update_Lminus1, 0, 1)

    def _activation_update_Lminus1_T(self, W_L, u_L, zl):
        """
        Implements the Activation update for l=L-1, t=T.

        Args:
            W_L (torch.Tensor): Weights of the output layer (L).
            u_L (torch.Tensor): Precomputed term u_{L,T}.
            zl (torch.Tensor): Pre-activation z_{L-1,T}.

        Returns:
            torch.Tensor: Updated activation a_{L-1,T}. Clipped between 0 and 1.
        """
        WtW = W_L.T @ W_L
        term1 = WtW + (self.beta / self.rho) * torch.eye(W_L.size(1)).to(self.device)
        term2 = (
            W_L.T @ (u_L.T + self.lambda_lagrange.T / self.rho)
            + (self.beta / self.rho) * self._heaviside(zl).T
        )

        activation_update_Lminus1_T = torch.inverse(term1) @ term2

        return torch.clip(activation_update_Lminus1_T, 0, 1)

    def _lambda_update(self, z_LT, z_LTprev, W_L, aL_minus_1_T):
        """
        Implements the update of the Lagrange multiplier lambda.

        Args:
            z_LT (torch.Tensor): Pre-activation z_{L,T}.
            z_LTprev (torch.Tensor): Pre-activation z_{L,T-1}.
            W_L (torch.Tensor): Weights of the output layer.
            aL_minus_1_T (torch.Tensor): Activation a_{L-1,T}.

        Returns:
            torch.Tensor: Updated Lagrange multiplier lambda.
        """
        return self.lambda_lagrange.T + self.rho * (
            z_LT.T - self.deltas * z_LTprev.T - W_L @ aL_minus_1_T.T
        )

    def fit(self, inputs, labels, warming=False):
        """
        Performs one iteration of the ADMM optimization algorithm.

        Updates weights (W), pre-activations (z), activations (a), and the
        Lagrange multiplier (lambda) based on the input batch and labels.

        Args:
            inputs (torch.Tensor): Input data batch.
            labels (torch.Tensor): Target labels (one-hot encoded).
            warming (bool, optional): If True, skips the Lagrange multiplier update
                                      (useful during initial iterations). Defaults to False.
        """

        # random_time_steps = random.sample(range(self.T - 1), self.T - 1)
        # random_layers = random.sample(range(self.L - 1), self.L - 1)
        # CHANGE
        random_time_steps = list(range(self.T - 1))
        random_layers = list(range(self.L - 1))

        for l in random_layers:
            # Update self.W[l] using the function _weight_update
            output_spikes = inputs if l == 0 else self.a[l - 1]

            u_lnext = self.z[l + 1].clone()
            u_lnext[1:] -= self.deltas * u_lnext[:-1]

            if l < self.L - 2:
                v_lnext = u_lnext.clone()
                v_lnext[1:] += self.thetas * self.a[l + 1][:-1]

            x_l = self.z[l].clone()
            x_l[1:] -= self.deltas * x_l[:-1]
            x_l[1:] += self.thetas * self.a[l][:-1]

            self.W[l] = self._weight_update(x_l.to(self.device), output_spikes, l)

            u_l = self.z[l].clone()
            u_l[1:] -= self.deltas * u_l[:-1]

            w_l = u_l - torch.matmul(
                self.W[l].unsqueeze(0).unsqueeze(0), output_spikes.unsqueeze(-1)
            ).squeeze(-1)

            for t in random_time_steps:
                if l < self.L - 2:
                    self.a[l][t] = self._activation_update(
                        self.W[l + 1], w_l[t + 1], self.z[l][t], v_lnext[t]
                    ).T
                else:
                    self.a[l][t] = self._activation_update_Lminus1(
                        self.W[l + 1], u_lnext[t], w_l[t + 1], self.z[l][t]
                    ).T

                q_lt = self.W[l] @ output_spikes[t].T
                if t >= 1:
                    q_lt += (
                        self.deltas * self.z[l][t - 1] - self.thetas * self.a[l][t - 1]
                    ).T
                r_ltnext = -self.W[l] @ output_spikes[t + 1].T + self.z[l][t + 1].T

                # update self.z[l][t] using the function _z_update and check_entries
                self.z[l][t] = self.check_entries(
                    self._z_update(q_lt, r_ltnext, self.a[l][t]),
                    self.a[l][t],
                    q_lt,
                    r_ltnext,
                )

            if l < self.L - 2:
                # update self.a[l][T] using the function _activation_update_T
                self.a[l][self.T - 1] = self._activation_update_T(
                    self.W[l + 1], v_lnext[self.T - 1], self.z[l][self.T - 1]
                ).T
            else:
                # update self.a[l][T] using the function _activation_update_Lminus1_T
                self.a[l][self.T - 1] = self._activation_update_Lminus1_T(
                    self.W[self.L - 1], u_lnext[self.T - 1], self.z[l][self.T - 1]
                ).T

            q_lt = (
                self.W[l] @ output_spikes[self.T - 1].T
                + (
                    self.deltas * self.z[l][self.T - 2]
                    - self.thetas * self.a[l][self.T - 2]
                ).T
            )
            # update self.z[l][T] using the function _z_update_T and check_entries
            self.z[l][self.T - 1] = self.check_entries_T(
                q_lt, self.a[l][self.T - 1], q_lt
            ).T

        # ----- Update the last layer -----
        # Update self.W[L] using the function _weight_update_L
        x_L = self.z[self.L - 1].clone()
        x_L[1:] -= self.deltas * x_L[:-1]

        self.W[self.L - 1] = self._weight_update_L(x_L, self.a[self.L - 2])
        for t in random_time_steps:
            # update self.z[L][t] using the function _z_update_L
            s_Lt = self.W[self.L - 1] @ self.a[self.L - 2][t].T
            if t >= 1:
                s_Lt += self.deltas * self.z[self.L - 1][t - 1].T

            r_Ltnext = (
                -self.W[self.L - 1] @ self.a[self.L - 2][t + 1].T
                + self.z[self.L - 1][t + 1].T
            )
            lam = (
                self.lambda_lagrange.T
                if t == self.T - 2
                else torch.zeros_like(self.lambda_lagrange.T)
            )
            self.z[self.L - 1][t] = self._z_update_L(s_Lt, r_Ltnext, lam).T

        # ----- Update the last layer at time T -----
        # update self.z[L][T] using the function _z_update_L_T
        s_LT = (
            self.W[self.L - 1] @ self.a[self.L - 2][self.T - 1].T
            + self.deltas * self.z[self.L - 1][self.T - 2].T
        )

        self.z[self.L - 1][self.T - 1] = self._z_update_L_T(s_LT, labels).T

        # Update the lagrange multiplier using the function _lambda_update
        if not warming:
            self.lambda_lagrange = self._lambda_update(
                self.z[self.L - 1][self.T - 1],
                self.z[self.L - 1][self.T - 2],
                self.W[self.L - 1],
                self.a[self.L - 2][self.T - 1],
            ).T
        return

    def lagrangian_cost(self, inputs, labels):
        """
        Computes the Augmented Lagrangian cost function (Equation 1).

        Args:
            inputs (torch.Tensor): Input data batch.
            labels (torch.Tensor): Target labels (one-hot encoded).

        Returns:
            torch.Tensor: Scalar value of the Lagrangian cost.
        """
        cost = 0
        for l in range(self.L - 1):
            output_spikes = inputs if l == 0 else self.a[l - 1]
            for t in range(1, self.T):
                term1 = (
                    self.W[l] @ output_spikes[t].T
                    - self.z[l][t].T
                    + self.deltas * self.z[l][t - 1].T
                )
                cost += self.rho / 2 * torch.norm(term1) ** 2
                term2 = self.a[l][t] - self._heaviside(self.z[l][t])
                cost += self.rho / 2 * torch.norm(term2) ** 2

            term1_t0 = self.W[l] @ output_spikes[0].T - self.z[l][0].T
            cost += self.rho / 2 * torch.norm(term1_t0) ** 2
            term2_t0 = self.a[l][0] - self._heaviside(self.z[l][0])
            cost += self.rho / 2 * torch.norm(term2_t0) ** 2

        for t in range(1, self.T):
            term1_L = (
                self.W[self.L - 1] @ self.a[self.L - 2][t].T
                - self.z[self.L - 1][t].T
                + self.deltas * self.z[self.L - 1][t - 1].T
            )
            cost += self.rho / 2 * torch.norm(term1_L) ** 2

        for i in range(self.lambda_lagrange.size(0)):
            cost += self.lambda_lagrange[i] @ (
                self.z[self.L - 1][self.T - 1][i]
                - self.deltas * self.z[self.L - 1][self.T - 2][i]
                - self.W[self.L - 1] @ self.a[self.L - 2][self.T - 1][i]
            )

        cost += torch.norm(self.z[self.L - 1][self.T - 1].T - labels) ** 2

        return cost

    def forward_model(self, inputs):
        """
        Performs a forward pass using the ADMM variables (W, z, a).

        This function simulates the network dynamics based on the learned ADMM parameters.

        Args:
            inputs (torch.Tensor): Input data batch.

        Returns:
            torch.Tensor: Output potentials (z) of the last layer
                          at the final timestep.
        """
        inputs = inputs.to(self.device)
        potential = [torch.zeros_like(z_l).to(self.device) for z_l in self.z]

        for t in range(self.T):
            current_input = (
                inputs[t]
                if t < inputs.size(0)
                else torch.zeros_like(inputs[0]).to(self.device)
            )

            for l in range(self.L):
                # Calculate post-synaptic current for the current layer
                post_syn_current = current_input @ self.W[l].T
                potential[l][t, :, :] = post_syn_current
                if t > 0:
                    potential[l][t, :, :] += self.deltas * potential[l][t - 1, :, :]
                    if l < self.L - 1:
                        potential[l][t, :, :] -= self.thetas * self._heaviside(
                            potential[l][t - 1, :, :]
                        )

                current_input = self._heaviside(potential[l][t, :, :])

        firing_rate = []

        for i in range(self.L - 1):
            print(
                self._heaviside(potential[i]).sum()
                / (self.z[i].shape[0] * self.z[i].shape[1] * self.z[i].shape[2])
            )
            firing_rate.append(
                self._heaviside(potential[i]).sum().item()
                / (self.z[i].shape[0] * self.z[i].shape[1] * self.z[i].shape[2])
            )

        return potential[-1][-1], firing_rate

    def primal_residual_norm(self):
        """
        Calculates the normalized L1 sum of the primal residual.

        The primal residual measures the violation of the constraint involving
        the output layer's final state (z_{L,T}). Used for monitoring convergence.

        Returns:
            torch.Tensor: Scalar value of the normalized primal residual sum.
        """

        r = (
            self.z[self.L - 1][self.T - 1]
            - self.deltas * self.z[self.L - 1][self.T - 2]
            - torch.mm(self.W[self.L - 1], self.a[self.L - 2][self.T - 1].T).T
        )

        return torch.norm(r) / (r.size(0) * r.size(1)) ** (1 / 2)

    def loss(self, labels):
        """
        Calculates the squared L2 loss between the final output potentials
        (z_{L,T}) and the target labels.

        Args:
            labels (torch.Tensor): Target labels (one-hot encoded).

        Returns:
            torch.Tensor: Scalar value of the loss.
        """
        return torch.norm(self.z[self.L - 1][self.T - 1].T - labels) ** 2

    def preactivation_constraint_sum(self, input_data_batch):
        """
        Calculates the sum of normalized L1 sums for the pre-activation constraints.

        Measures the violation of the equation:
        z_{l,t} - delta*z_{l,t-1} - W_l @ a_{l-1,t} + theta*a_{l,t-1} = 0
        (Adjusted for t=0 and output layer where the equation differs).

        Args:
            input_data_batch (torch.Tensor): The input data batch used in the fit step.

        Returns:
            List[torch.Tensor]: A list containing the normalized L1 sum of the
                                residual for each layer's pre-activation constraint.
        """

        constraints_residuals = []
        output_spikes = [input_data_batch] + self.a
        output_spikes_t_minus1 = self.a + [torch.zeros_like(self.z[-1])]
        for preactivation, activation_l_minus1, activation_t_minus1, w in zip(
            self.z, output_spikes, output_spikes_t_minus1, self.W
        ):
            zero_activation_tensor = torch.zeros(
                1, activation_t_minus1.shape[1], activation_t_minus1.shape[2]
            ).to(self.device)
            shifted_activation = torch.cat(
                (zero_activation_tensor, activation_t_minus1[:-1, :, :]), dim=0
            )
            zero_preactivation_tensor = torch.zeros(
                1, preactivation.shape[1], preactivation.shape[2]
            ).to(self.device)
            shifted_preactivation = torch.cat(
                (zero_preactivation_tensor, preactivation[:-1, :, :]), dim=0
            )

            # use signed residual (no absolute value)
            soft_constraint_residual = (
                preactivation
                - self.deltas * shifted_preactivation
                - (w @ activation_l_minus1.unsqueeze(-1)).squeeze(-1)
                + self.thetas * shifted_activation
            )
            constraints_residuals.append(
                torch.norm(soft_constraint_residual).cpu()
                / (
                    soft_constraint_residual.shape[0]
                    * soft_constraint_residual.shape[1]
                    * soft_constraint_residual.shape[2]
                )
                ** (1 / 2)
            )

        return constraints_residuals

    def activation_constraint_sum(self):
        """
        Calculates the sum of normalized L1 sums for the activation constraints.

        Measures the violation of the equation: a_{l,t} - h(z_{l,t}) = 0
        for hidden layers l = 0 to L-2.

        Returns:
            List[torch.Tensor]: A list containing the normalized L1 sum of the
                                residual for each hidden layer's activation constraint.
        """
        constraints_residuals = []
        for preactivation, activation in zip(self.z[:-1], self.a):
            # use signed residual between hevaiside(preactivation) and activation
            soft_constraint_residual = self._heaviside(preactivation) - activation
            constraints_residuals.append(
                torch.norm(soft_constraint_residual).cpu()
                / (
                    soft_constraint_residual.shape[0]
                    * soft_constraint_residual.shape[1]
                    * soft_constraint_residual.shape[2]
                )
                ** (1 / 2)
            )

        return constraints_residuals


if __name__ == "__main__":
    # ==========================================
    # 1. SETUP AND DETERMINISM
    # ==========================================
    seed = 8281003564
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )

    batch_size = 10
    trainset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")

    trainloader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

    n_samples = batch_size
    n_timesteps = 10
    input_dim = sensor_size[0] * sensor_size[1]
    hidden_dims = [100]  # List for old framework compatibility
    n_outputs = 10

    rho = 1.0
    deltas = 0.95
    thetas = 1.0
    beta = 1.0

    num_epochs = 10
    n_warming_iters = 5

    # ==========================================
    # 2. DATA PREPARATION (Shared)
    # ==========================================
    # Get a single batch to overfit
    data, targets_raw = next(iter(trainloader))
    data, targets_raw = data.to(device), targets_raw.to(device)

    data = data.view(data.size(0), data.size(1), -1)
    targets_one_hot = torch.nn.functional.one_hot(
        targets_raw.to(torch.long), num_classes=10
    ).T

    if data.size(1) > n_timesteps:
        data = data[:, :n_timesteps, :]

    data = data.permute(1, 0, 2)
    # Apply deterministic noise for testing equivalence
    torch.manual_seed(seed)
    data += 0.01 * torch.randn_like(data)

    # Save original shapes for the new framework formatting
    new_framework_data = data.clone()
    new_framework_targets = (
        targets_one_hot.T.clone()
    )  # New framework expects [B, num_classes]

    # ==========================================
    # 3. OLD FRAMEWORK TRAINING
    # ==========================================
    print(f"\n{'=' * 40}\nTRAINING OLD FRAMEWORK\n{'=' * 40}")

    # Reset seed right before model init to ensure random weight/state init matches
    torch.manual_seed(seed)
    random.seed(seed)

    old_model = ADMM_SNN(
        n_samples,
        n_timesteps,
        input_dim,
        hidden_dims,
        n_outputs,
        rho,
        deltas,
        thetas,
        beta,
    )

    old_final_loss = 0
    old_final_acc = 0

    for epoch in range(num_epochs + 1):
        old_model.fit(data, targets_one_hot, warming=epoch < n_warming_iters)

        loss = old_model.loss(targets_one_hot).item()
        pot, firing_rate = old_model.forward_model(data)
        _, predicted = pot.max(1)
        labels = torch.argmax(targets_one_hot, dim=0)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        print(f"Epoch [{epoch}/{num_epochs}] | Loss: {loss:.4f} | Acc: {accuracy:.4f}")

        if epoch == num_epochs:
            old_final_loss = loss
            old_final_acc = accuracy

    # ==========================================
    # 4. NEW FRAMEWORK TRAINING
    # ==========================================
    print(f"\n{'=' * 40}\nTRAINING NEW FRAMEWORK\n{'=' * 40}")

    # Reset seed right before new model init
    torch.manual_seed(seed)
    random.seed(seed)

    # Initialize new framework via get_model
    hidden_layer_config = ADMM_LayerConfig(
        rho=rho,
        beta=beta,
        use_bias=False,
        deltas=deltas,
        thetas=thetas,
        use_reset=True,
    )
    out_layer_config = ADMM_LayerConfig(
        rho=rho,
        beta=beta,
        use_bias=False,
        deltas=deltas,
        thetas=thetas,
        use_reset=False,
        use_lagrange=True,
    )
    spff_layers = nn.ModuleList(
        [
            ADMM_SpikingFeedForward(
                in_f=34 * 34 * 2,
                out_f=hidden_dims[0],
                h=ADMM_Heaviside(thetas=thetas),
                config=hidden_layer_config,
            ),
            ADMM_SpikingFeedForward(
                in_f=hidden_dims[0],
                out_f=10,
                h=None,
                config=out_layer_config,
            ),
        ]
    )
    config = ADMM_Config(
        init="zeros",
        train_method="unrolled-sequential",
        layer_order="sequential",
        update_z_first=False,
    )
    new_model = ADMM(
        spff_layers,
        loss_f=ADMM_SSE(),
        T=n_timesteps,
        config=config,
    ).to(device)

    # Create a mock dataloader that just yields our single prepared batch
    mock_dataloader = [(new_framework_data, new_framework_targets)]

    new_final_loss = 0
    new_final_acc = 0

    # You will need to import ADMM_Metrics if you want exactly the same printout,
    # but for simple comparison, we'll manually check the state.
    from src.admm import ADMM_Metrics

    metrics_tracker = ADMM_Metrics(new_model)

    for epoch in range(num_epochs + 1):
        new_model.fit(mock_dataloader, warming=epoch < n_warming_iters)

        # Load the batch state for metric calculation
        _, _, batch_state = new_model.state_handler.load_batch(0)

        # We need to manually calculate the loss metric if the ADMM_CrossEntropy_Taylor
        # differs from the old L2 norm.
        loss = metrics_tracker.loss(new_framework_targets, batch_state)
        acc, _ = metrics_tracker.evaluate_performance(
            new_framework_data, new_framework_targets
        )

        print(f"Epoch [{epoch}/{num_epochs}] | Loss: {loss:.4f} | Acc: {acc:.4f}")

        if epoch == num_epochs:
            new_final_loss = loss
            new_final_acc = acc / 100.0  # Convert percentage back to decimal

    # ==========================================
    # 5. VERIFICATION
    # ==========================================
    print(f"\n{'=' * 40}\nCOMPARISON RESULTS\n{'=' * 40}")
    print(
        f"Old Model - Final Loss: {old_final_loss:.4f}, Final Acc: {old_final_acc:.4f}"
    )
    print(
        f"New Model - Final Loss: {new_final_loss:.4f}, Final Acc: {new_final_acc:.4f}"
    )

    if (
        abs(old_final_loss - new_final_loss) < 1e-4
        and abs(old_final_acc - new_final_acc) < 1e-4
    ):
        print("\n✅ SUCCESS: The frameworks match exactly.")
    else:
        print("\n❌ MISMATCH: The frameworks produced different results.")
