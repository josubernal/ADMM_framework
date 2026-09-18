import configparser
import itertools
import json
import os
import random
from typing import List

import matplotlib.pyplot as plt
import tonic
import tonic.transforms as transforms
import torch
from safetensors.torch import load_file, save_file
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


class ADMM_SNN:
    """Class for ADMM Neural Network."""

    def __init__(
        self,
        n_samples: int,
        n_timesteps: int,
        input_dim: int,
        hidden_dims: List[int],
        num_batches: int,
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

        self.preactivation_constraints = [0 for _ in range(len(hidden_dims) + 1)]
        self.activation_constraints = [0 for _ in range(len(hidden_dims))]

        # self.round_factor = 0.1

        self.num_batches = num_batches

        self.pinv_l_0 = [
            None for _ in range(self.num_batches)
        ]  # Attribute to store the pseudoinverse at layer 0

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
        # for i, hidden_dim in enumerate(hidden_dims + [n_outputs]):
        #     self.z.append(torch.rand((n_timesteps, n_samples, hidden_dim)).to(self.device))

        # === Initialize a_l (Activations/Spikes) ===
        self.a = []
        # for i, hidden_dim in enumerate(hidden_dims):
        #     self.a.append(torch.rand(
        #         (n_timesteps, n_samples, hidden_dim)).to(self.device))

        # # === Initialize lagrange multipliers (only for the output layer constraint) ===
        # self.lambda_lagrange = torch.zeros(
        #     (n_samples, n_outputs)).to(self.device)

        self.numerator = [
            torch.zeros(self.W[l].shape[0], self.W[l].shape[1]).to(self.device)
            for l in range(self.L)
        ]
        self.denominator = [
            torch.zeros(self.W[l].shape[1], self.W[l].shape[1]).to(self.device)
            for l in range(self.L)
        ]

        self.lagr = 0
        self.loss = 0
        self.accuracy = 0
        self.firing_rate = 0
        self.size = 0

        self.epoch = 0

    def reset_metrics(self):
        """
        Resets the metrics for loss, accuracy, size, and lagrangian cost.
        """
        self.lagr = 0
        self.loss = 0
        self.accuracy = 0
        self.firing_rate = 0
        self.size = 0

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

        # if l == 0:

        #     if self.pinv_l_0 is None:

        #         self.pinv_l_0 = torch.linalg.pinv(denominator)

        #     return numerator @ self.pinv_l_0

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

    def compute_numerator_l(self, x_l, a_lminus1, l):
        """
        Computes the numerator for the weight update of layer l.

        Args:
            x_l (torch.Tensor): Precomputed term related to z_l.
            a_lminus1 (torch.Tensor): Activations of the previous layer.
            l (int): Layer index.
        Returns:
            torch.Tensor: Numerator for the weight update of layer l.
        """

        return sum(x_l[t].T @ a_lminus1[t] for t in range(self.T))

    def compute_denominator_l(self, a_lminus1, l):
        """
        Computes the denominator for the weight update of layer l.

        Args:
            a_lminus1 (torch.Tensor): Activations of the previous layer.
            l (int): Layer index.
        Returns:
            torch.Tensor: Denominator for the weight update of layer l.
        """

        return sum((a_lminus1[t].T @ a_lminus1[t]) for t in range(self.T))

    def compute_numerator_L(self, x_L, a_Lminus1):
        """
        Computes the numerator for the weight update of the output layer L.

        Args:
            x_L (torch.Tensor): Precomputed term related to z_L.
            a_Lminus1 (torch.Tensor): Activations of the previous layer (L-1).
        Returns:
            torch.Tensor: Numerator for the weight update of the output layer L.
        """

        lambda_term = self.lambda_lagrange.T / self.rho
        return lambda_term @ a_Lminus1[-1] + sum(
            x_L[t].T @ a_Lminus1[t] for t in range(self.T)
        )

    def compute_denominator_L(self, a_Lminus1):
        """
        Computes the denominator for the weight update of the output layer L.

        Args:
            a_Lminus1 (torch.Tensor): Activations of the previous layer (L-1).
        Returns:
            torch.Tensor: Denominator for the weight update of the output layer L.
        """

        return sum((a_Lminus1[t].T @ a_Lminus1[t]) for t in range(self.T))

    def reset_numerator_denominator(self):
        """
        Resets the numerator and denominator accumulators for all layers.
        """

        self.numerator = [
            torch.zeros(self.W[l].shape[0], self.W[l].shape[1]).to(self.device)
            for l in range(self.L)
        ]
        self.denominator = [
            torch.zeros(self.W[l].shape[1], self.W[l].shape[1]).to(self.device)
            for l in range(self.L)
        ]

    def compute_weights_updates(self):

        for l in range(self.L - 1):
            self.W[l] = self.numerator[l] @ torch.linalg.pinv(self.denominator[l])

        self.W[self.L - 1] = self.numerator[self.L - 1] @ torch.linalg.pinv(
            self.denominator[self.L - 1]
        )

    def load_batch(self, num_batch):

        batch_path = os.path.join("batches", f"batch_{num_batch}")

        if os.path.exists(batch_path):
            a_tensors = load_file(os.path.join(batch_path, "a_tensors.safetensors"))
            self.a = [a_tensors[str(i)].to(self.device) for i in range(len(a_tensors))]

            z_tensors = load_file(os.path.join(batch_path, "z_tensors.safetensors"))
            self.z = [z_tensors[str(i)].to(self.device) for i in range(len(z_tensors))]

            lambda_tensor = load_file(
                os.path.join(batch_path, "lambda_lagrange.safetensors")
            )
            self.lambda_lagrange = lambda_tensor["lambda_tensor"].to(self.device)

    def initialize_variables(self, num_batches):

        batch_path = os.path.join("batches")

        os.makedirs(batch_path, exist_ok=True)

        # === Initialize z_l (Pre-activation potentials) ===
        z_variables = []
        for i, hidden_dim in enumerate(hidden_dims + [n_outputs]):
            z_variables.append(
                torch.rand((n_timesteps, n_samples * num_batches, hidden_dim)).to(
                    self.device
                )
            )

        # === Initialize a_l (Activations/Spikes) ===
        a_variables = []
        for i, hidden_dim in enumerate(hidden_dims):
            a_variables.append(
                torch.rand((n_timesteps, n_samples * num_batches, hidden_dim)).to(
                    self.device
                )
            )

        # === Initialize lagrange multipliers (only for the output layer constraint) ===
        lambda_lagrange = torch.zeros((n_samples * num_batches, n_outputs)).to(
            self.device
        )

        for num_batch in range(num_batches):
            self.z = [
                z_l[:, n_samples * (num_batch) : n_samples * (num_batch + 1), :]
                for z_l in z_variables
            ]
            self.a = [
                a_l[:, n_samples * (num_batch) : n_samples * (num_batch + 1), :]
                for a_l in a_variables
            ]
            self.lambda_lagrange = lambda_lagrange[
                n_samples * (num_batch) : n_samples * (num_batch + 1), :
            ]

            self.save_batch(num_batch)

        del z_variables, a_variables, lambda_lagrange

    def save_batch(self, num_batch):

        batch_path = os.path.join("batches", f"batch_{num_batch}")
        os.makedirs(batch_path, exist_ok=True)

        a_tensors = {str(i): self.a[i].cpu() for i in range(len(self.a))}
        save_file(a_tensors, os.path.join(batch_path, "a_tensors.safetensors"))
        z_tensors = {str(i): self.z[i].cpu() for i in range(len(self.z))}
        save_file(z_tensors, os.path.join(batch_path, "z_tensors.safetensors"))
        lambda_tensor = {"lambda_tensor": self.lambda_lagrange.cpu()}
        save_file(
            lambda_tensor, os.path.join(batch_path, "lambda_lagrange.safetensors")
        )

    def fit(self, inputs, targets, num_batches, warming=False):
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

        random_time_steps = random.sample(range(self.T - 1), self.T - 1)
        random_layers = random.sample(range(self.L - 1), self.L - 1)

        for n in range(num_batches):
            self.load_batch(n)

            for l in random_layers:
                # Update self.W[l] using the function _weight_update
                output_spikes = inputs[n] if l == 0 else self.a[l - 1]

                u_lnext = self.z[l + 1].clone()
                u_lnext[1:] -= self.deltas * u_lnext[:-1]

                if l < self.L - 2:
                    v_lnext = u_lnext.clone()
                    v_lnext[1:] += self.thetas * self.a[l + 1][:-1]

                u_l = self.z[l].clone()
                u_l[1:] -= self.deltas * u_l[:-1]

                x_l = self.z[l].clone()
                x_l[1:] -= self.deltas * x_l[:-1]
                x_l[1:] += self.thetas * self.a[l][:-1]

                self.numerator[l] += self.compute_numerator_l(x_l, output_spikes, l)
                self.denominator[l] += self.compute_denominator_l(output_spikes, l)

        for l in range(self.L - 1):
            self.W[l] = self.numerator[l] @ torch.linalg.pinv(self.denominator[l])

        for n in range(num_batches):
            self.load_batch(n)

            for l in random_layers:
                output_spikes = inputs[n] if l == 0 else self.a[l - 1]

                u_lnext = self.z[l + 1].clone()
                u_lnext[1:] -= self.deltas * u_lnext[:-1]

                if l < self.L - 2:
                    v_lnext = u_lnext.clone()
                    v_lnext[1:] += self.thetas * self.a[l + 1][:-1]

                u_l = self.z[l].clone()
                u_l[1:] -= self.deltas * u_l[:-1]

                x_l = self.z[l].clone()
                x_l[1:] -= self.deltas * x_l[:-1]
                x_l[1:] += self.thetas * self.a[l][:-1]

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
                            self.deltas * self.z[l][t - 1]
                            - self.thetas * self.a[l][t - 1]
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

            self.save_batch(n)

        for n in range(num_batches):
            self.load_batch(n)

            x_L = self.z[self.L - 1].clone()
            x_L[1:] -= self.deltas * x_L[:-1]

            self.numerator[self.L - 1] += self.compute_numerator_L(
                x_L, self.a[self.L - 2]
            )
            self.denominator[self.L - 1] += self.compute_denominator_L(
                self.a[self.L - 2]
            )

            self.save_batch(n)

        self.W[self.L - 1] = self.numerator[self.L - 1] @ torch.linalg.pinv(
            self.denominator[self.L - 1]
        )

        for n in range(num_batches):
            self.load_batch(n)

            x_L = self.z[self.L - 1].clone()
            x_L[1:] -= self.deltas * x_L[:-1]

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

            self.z[self.L - 1][self.T - 1] = self._z_update_L_T(s_LT, targets[n]).T

            if not warming:
                self.lambda_lagrange = self._lambda_update(
                    self.z[self.L - 1][self.T - 1],
                    self.z[self.L - 1][self.T - 2],
                    self.W[self.L - 1],
                    self.a[self.L - 2][self.T - 1],
                ).T

            # if epoch >= 120:

            #     self.z = [torch.round(z_l * 1e3) / 1e3 for z_l in self.z]
            #     self.lambda_lagrange = torch.round(self.lambda_lagrange * 1e4) / 1e4
            #     # self.W = [torch.round(W_l * 1e5) / 1e5 for W_l in self.W]

            # if epoch >= 140:

            #     self.a = [torch.where(a_l < self.round_factor, torch.zeros_like(a_l), a_l) for a_l in self.a]
            #     self.a  = [torch.where(a_l >= 1 - self.round_factor, torch.ones_like(a_l), a_l) for a_l in self.a]

            #     self.round_factor = self.round_factor * 1.1 if self.round_factor < 0.5 else 0.5

            if (self.epoch) % 10 == 0:
                self.size += inputs[n].size(1)

                self.lagr += self.lagrangian_cost(inputs[n], targets[n]).item()

                # Calculate loss
                self.loss += self.compute_loss(targets[n]).item()

                self.preactivation_constraints = [
                    self.preactivation_constraints[l]
                    + self.preactivation_constraint_sum(inputs[n])[l]
                    for l in range(len(self.z))
                ]
                self.activation_constraints = [
                    self.activation_constraints[l] + self.activation_constraint_sum()[l]
                    for l in range(len(self.a))
                ]

                # Run forward model to get predictions
                pot, firing_rate = self.forward_model(inputs[n])
                _, predicted = pot.max(1)
                labels = torch.argmax(targets[n], dim=0)
                self.accuracy += (predicted == labels).sum().item()

            self.save_batch(n)

        if (self.epoch) % 10 == 0:
            self.accuracy /= self.size
            self.preactivation_constraints = [
                preactivation_residuals ** (1 / 2)
                / (self.z[l].size(0) * self.size * self.z[l].size(2)) ** (1 / 2)
                for l, preactivation_residuals in enumerate(
                    self.preactivation_constraints
                )
            ]
            self.activation_constraints = [
                activation_residuals ** (1 / 2)
                / (self.a[l].size(0) * self.size * self.a[l].size(2)) ** (1 / 2)
                for l, activation_residuals in enumerate(self.activation_constraints)
            ]

            self.round_residuals()

            self.lagr /= self.size
            self.loss /= self.size

        self.epoch += 1

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

    def round_residuals(self):
        """
        Rounds the pre-activation potentials (z) and Lagrange multipliers
        to reduce numerical noise in the residual calculations.
        """

        self.preactivation_constraints = [
            round(preactivation_residuals, 6)
            for preactivation_residuals in self.preactivation_constraints
        ]
        self.activation_constraints = [
            round(activation_residuals, 6)
            for activation_residuals in self.activation_constraints
        ]

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

    def compute_loss(self, labels):
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
            soft_constraint_residual = (
                torch.sum(soft_constraint_residual**2).cpu().item()
            )
            constraints_residuals.append(
                round(soft_constraint_residual, 6)
            )  # / (soft_constraint_residual.shape[0] * soft_constraint_residual.shape[1] * soft_constraint_residual.shape[2]) ** (1 / 2)

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
            soft_constraint_residual = (
                torch.sum(soft_constraint_residual**2).cpu().item()
            )
            constraints_residuals.append(
                round(soft_constraint_residual, 6)
            )  # / (soft_constraint_residual.shape[0] * soft_constraint_residual.shape[1] * soft_constraint_residual.shape[2]) ** (1 / 2)

        return constraints_residuals


if __name__ == "__main__":
    seed = 8281003564
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = configparser.ConfigParser()
    config.read("paper/config/config.ini")
    # Define the ADMM SNN model parameters (use command line args if provided, otherwise use config)
    batch_size = config.getint("config", "batch_size_spiking")
    n_samples = batch_size  # Update both variables

    n_timesteps = config.getint("config", "n_timesteps")

    # Handle hidden_dims specially since it's a list
    hidden_dims = eval(config.get("config", "hidden_dims"))

    n_outputs = config.getint("config", "n_outputs")

    rho = config.getfloat("config", "spff_rho")
    deltas = config.getfloat("config", "ffdeltas")

    thetas = config.getfloat("config", "ffthetas")

    beta = config.getfloat("config", "spff_beta")

    num_batches = config.getint("config", "num_batches")

    # Define transformations
    sensor_size = tonic.datasets.NMNIST.sensor_size
    input_dim = sensor_size[0] * sensor_size[1]  # there are 2 channels
    # Combine channels
    input_dim *= 2

    # Define transformations applied to the dataset
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
        ]
    )

    # Load datasets
    batch_size = config.getint("config", "batch_size_spiking")

    trainset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=True
    )
    testset = tonic.datasets.NMNIST(
        save_to="./data", transform=frame_transform, train=False
    )

    # Cache datasets
    cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/nmnist/train")
    cached_testset = DiskCachedDataset(testset, cache_path="./cache/nmnist/test")

    # DataLoaders
    trainloader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    testloader = DataLoader(
        cached_testset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(),
        drop_last=True,
    )

    model = ADMM_SNN(
        n_samples,
        n_timesteps,
        input_dim,
        hidden_dims,
        num_batches,
        n_outputs,
        rho,
        deltas,
        thetas,
        beta,
    )

    sample_data, sample_target = next(iter(trainloader))
    print("Sample data shape:", sample_data.shape)
    print("Sample target shape:", sample_target.shape)

    # Training parameters
    num_epochs = config.getint("config", "epochs")

    n_warming_iters = config.getint("config", "warming_iters")

    data_iterator = iter(trainloader)
    data, targets = next(data_iterator)
    data = data[:, :n_timesteps, ...]

    for batch in itertools.islice(data_iterator, num_batches - 1):
        # ===================== Main ADMM training loop =====================
        # Get a batch of data and preprocess it
        batch_data, batch_targets = batch[0][:, :n_timesteps, ...], batch[1]
        # Concatenate batches in a single tensor
        data, targets = (
            torch.cat((data, batch_data), dim=0),
            torch.cat((targets, batch_targets), dim=0),
        )

    data = data.to(device)
    targets = targets.to(device)

    # Extract and combine channels (optional alternative)
    # data = torch.clip(data, 0, 1)
    # data = data[:, :, 0, :, :] - data[:, :, 1, :, :]
    # Reshape data for processing
    data = data.view(data.size(0), data.size(1), -1)
    # Convert targets to one-hot encoding
    targets = torch.nn.functional.one_hot(targets.to(torch.long), num_classes=10).T
    # Truncate data if needed
    if data.size(1) > n_timesteps:
        data = data[:, :n_timesteps, :]
    # Permute dimensions to have time as first dimension
    data = data.permute(1, 0, 2)
    data += 0.01 * torch.randn_like(data)

    data_list = [
        (
            data[:, batch_size * i : batch_size * (i + 1), :],
            targets[:, batch_size * i : batch_size * (i + 1)],
        )
        for i in range(num_batches)
    ]

    model.initialize_variables(num_batches)

    metrics = {}

    metrics_path = "/paper/results/admm-spiking-perin-et-al/results.json"
    os.makedirs(metrics_path, exist_ok=True)

    # Initialize lists to track metrics
    lagrangians, lambdas = [], []
    soft_constraints = {"a": [], "z": []}
    losses = []
    accuracy_list = []
    firing_rate_list = []

    for epoch in range(num_epochs + 1):
        # Perform ADMM optimization using the `fit` method
        # If in warming iterations, skip lambda updates

        model.reset_metrics()

        model.reset_numerator_denominator()

        model.fit(*zip(*data_list), len(data_list), warming=epoch < n_warming_iters)

        if (epoch) % 10 == 0:
            training_log = (
                f"Lagrangian: {model.lagr:.4f},\n"
                f"lambda sum: {model.lambda_lagrange.norm().item() / (model.lambda_lagrange.size(0) * model.lambda_lagrange.size(1))}, "
                f"loss: {model.loss:.4f}, "
                f"residual norm: {model.primal_residual_norm().item():.4f}\n"
            )

            print(
                ("----------------------------\n")
                + (f"Epoch [{epoch}/{num_epochs}] Train accuracy: {model.accuracy} \n")
                + training_log
                + ("----------------------------")
            )

            # Calculate constraint violations
            soft_constraints["a"].append(
                [constraint_sum for constraint_sum in model.activation_constraints]
            )
            soft_constraints["z"].append(
                [constraint_sum for constraint_sum in model.preactivation_constraints]
            )
            lagrangians.append(model.lagr)
            lambdas.append(model.primal_residual_norm())
            losses.append(model.loss)
            accuracy_list.append(model.accuracy)
            firing_rate_list.append(model.firing_rate)

    metrics["lagrangians"] = lagrangians
    metrics["lambdas"] = lambdas
    metrics["soft_constraints"] = soft_constraints
    metrics["losses"] = losses
    metrics["accuracy_list"] = accuracy_list
    metrics["firing_rate"] = firing_rate_list

    with open(os.path.join(metrics_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Create plots to visualize training metrics
    fig, ax = plt.subplots(3, 3, figsize=(30, 5))
    ax[0, 0].semilogy(lagrangians)
    ax[0, 0].set_title("Lagrangian")
    ax[0, 1].semilogy(lambdas)
    ax[0, 1].set_title("primal residual norm")
    ax[0, 2].semilogy(soft_constraints["a"])
    ax[0, 2].set_title("$\|a - h(z, \\theta)\|_2$")
    ax[1, 0].semilogy(soft_constraints["z"])
    ax[1, 0].set_title(
        "$\|z_1 - \delta*z_{shifted} - W_1a_0 + \theta*a_{1,shifted}\|_2$"
    )
    ax[1, 1].semilogy(losses)
    ax[1, 1].set_title("Loss")
    ax[1, 2].plot(accuracy_list)
    ax[1, 2].set_title("Train accuracy")

    # Run final evaluation on training data
    pot, firing_rate = model.forward_model(data)
    _, predicted = pot.max(1)
    labels = torch.argmax(targets, dim=0)
    print("----------------------------")
    print("Train accuracy:", (predicted == labels).sum().item() / labels.size(0))
    print("----------------------------")
    plt.tight_layout()
    plt.show()
