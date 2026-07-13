import gc
import json
import os
import random
import time
from typing import List, Optional

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
    ADMM_SpikingFeedForward,
)

torch.set_default_dtype(torch.float64)


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
        self.pinv_l_0 = None

        self.L = len(hidden_dims) + 1
        self.T = n_timesteps

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
                torch.zeros((n_timesteps, n_samples, hidden_dim)).to(self.device)
            )

        # === Initialize a_l (Activations/Spikes) ===
        self.a = []
        for i, hidden_dim in enumerate(hidden_dims):
            self.a.append(
                torch.zeros((n_timesteps, n_samples, hidden_dim)).to(self.device)
            )

        # === Initialize lagrange multipliers ===
        self.lambda_lagrange = torch.zeros((n_samples, n_outputs)).to(self.device)

    def _heaviside(self, x):
        return (x > self.thetas).double()

    def _weight_update(self, x_l, a_lminus1, l):
        numerator = sum(x_l[t].T @ a_lminus1[t] for t in range(self.T))
        denominator = sum((a_lminus1[t].T @ a_lminus1[t]) for t in range(self.T))

        if l == 0:
            if self.pinv_l_0 is None:
                self.pinv_l_0 = torch.linalg.pinv(denominator)
            return numerator @ self.pinv_l_0

        return numerator @ torch.linalg.pinv(denominator)

    def _weight_update_L(self, x_L, a_Lminus1):
        lambda_term = self.lambda_lagrange.T / self.rho
        numerator = lambda_term @ a_Lminus1[-1] + sum(
            x_L[t].T @ a_Lminus1[t] for t in range(self.T)
        )
        denominator = sum((a_Lminus1[t].T @ a_Lminus1[t]) for t in range(self.T))

        return numerator @ torch.linalg.pinv(denominator)

    def _z_update(self, q_l, r_l, a_l):
        return (q_l.T + self.deltas * (r_l.T + self.thetas * a_l)) / (
            1 + self.deltas**2
        )

    def _z_update_L(self, s_Lt, r_Ltnext, lam):
        return (s_Lt + self.deltas * r_Ltnext + self.deltas * lam / self.rho) / (
            1 + self.deltas**2
        )

    def _z_update_L_T(self, s_LT, y):
        return (self.rho * s_LT + 2 * y - self.lambda_lagrange.T) / (2 + self.rho)

    def check_entries(self, z, a_lt, q_lt, r_ltnext):
        delta1 = self.beta * (1 - 2 * a_lt)
        delta2 = (
            self.rho * (z - q_lt.T) ** 2
            + self.rho * (r_ltnext.T - self.deltas * z + self.thetas * a_lt) ** 2
            - self.rho * (self.thetas - q_lt.T) ** 2
            - self.rho
            * (r_ltnext.T - self.deltas * self.thetas + self.thetas * a_lt) ** 2
        )

        mask_z_greater = (z > self.thetas).double()
        mask_deltas1 = (delta1 + delta2 > 0).double()
        mask_deltas2 = (-delta1 + delta2 > 0).double()

        mask_intersection1 = mask_z_greater * mask_deltas1
        z[mask_intersection1 == 1] = self.thetas

        mask_intersection2 = (1 - mask_z_greater) * mask_deltas2
        z[mask_intersection2 == 1] = self.thetas + 1e-5

        return z

    def check_entries_T(self, z, a_lt, q_lt):
        delta1 = self.beta * (1 - 2 * a_lt).T
        delta2 = self.rho * (z - q_lt) ** 2 - self.rho * (self.thetas - q_lt) ** 2

        mask_z_greater = (z > self.thetas).double()
        mask_deltas1 = (delta1 + delta2 > 0).double()
        mask_deltas2 = (-delta1 + delta2 > 0).double()

        mask_intersection1 = mask_z_greater * mask_deltas1
        z[mask_intersection1 == 1] = self.thetas

        mask_intersection2 = (1 - mask_z_greater) * mask_deltas2
        z[mask_intersection2 == 1] = self.thetas + 1e-5

        return z

    def _activation_update(self, Wl_next, wl, zl, vl_next):
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
        term1 = W_lnext.T @ W_lnext + (self.beta / self.rho) * torch.eye(
            W_lnext.size(1)
        ).to(self.device)
        a = torch.inverse(term1) @ (
            W_lnext.T @ v_lnext.T + (self.beta / self.rho) * self._heaviside(z_l).T
        )

        return torch.clip(a, 0, 1)

    def _activation_update_Lminus1(self, Wl_next, u_L, w_Lminus1, zl):
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
        WtW = W_L.T @ W_L
        term1 = WtW + (self.beta / self.rho) * torch.eye(W_L.size(1)).to(self.device)
        term2 = (
            W_L.T @ (u_L.T + self.lambda_lagrange.T / self.rho)
            + (self.beta / self.rho) * self._heaviside(zl).T
        )

        activation_update_Lminus1_T = torch.inverse(term1) @ term2

        return torch.clip(activation_update_Lminus1_T, 0, 1)

    def _lambda_update(self, z_LT, z_LTprev, W_L, aL_minus_1_T):
        return self.lambda_lagrange.T + self.rho * (
            z_LT.T - self.deltas * z_LTprev.T - W_L @ aL_minus_1_T.T
        )

    def fit(self, inputs, labels, warming=False):
        metrics = {
            "weight_time": 0.0,
            "weight_mem": 0.0,
            "act_time": 0.0,
            "act_mem": 0.0,
            "z_time": 0.0,
            "z_mem": 0.0,
            "global_peak": 0.0,
        }

        random_time_steps = list(range(self.T - 1))
        random_layers = list(range(self.L - 1))

        for l in random_layers:
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
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
            torch.cuda.synchronize()

            w_peak = torch.cuda.max_memory_allocated()
            metrics["weight_time"] += time.time() - start
            metrics["weight_mem"] = max(metrics["weight_mem"], w_peak)
            metrics["global_peak"] = max(metrics["global_peak"], w_peak)

            for t in random_time_steps:
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                u_l = self.z[l].clone()
                u_l[1:] -= self.deltas * u_l[:-1]

                w_l = u_l - torch.matmul(
                    self.W[l].unsqueeze(0).unsqueeze(0), output_spikes.unsqueeze(-1)
                ).squeeze(-1)
                if l < self.L - 2:
                    self.a[l][t] = self._activation_update(
                        self.W[l + 1], w_l[t + 1], self.z[l][t], v_lnext[t]
                    ).T
                else:
                    self.a[l][t] = self._activation_update_Lminus1(
                        self.W[l + 1], u_lnext[t], w_l[t + 1], self.z[l][t]
                    ).T
                torch.cuda.synchronize()

                a_peak = torch.cuda.max_memory_allocated()
                metrics["act_time"] += time.time() - start
                metrics["act_mem"] = max(metrics["act_mem"], a_peak)
                metrics["global_peak"] = max(metrics["global_peak"], a_peak)

                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                q_lt = self.W[l] @ output_spikes[t].T
                if t >= 1:
                    q_lt += (
                        self.deltas * self.z[l][t - 1] - self.thetas * self.a[l][t - 1]
                    ).T
                r_ltnext = -self.W[l] @ output_spikes[t + 1].T + self.z[l][t + 1].T

                self.z[l][t] = self.check_entries(
                    self._z_update(q_lt, r_ltnext, self.a[l][t]),
                    self.a[l][t],
                    q_lt,
                    r_ltnext,
                )
                torch.cuda.synchronize()

                z_peak = torch.cuda.max_memory_allocated()
                metrics["z_time"] += time.time() - start
                metrics["z_mem"] = max(metrics["z_mem"], z_peak)
                metrics["global_peak"] = max(metrics["global_peak"], z_peak)

            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            if l < self.L - 2:
                self.a[l][self.T - 1] = self._activation_update_T(
                    self.W[l + 1], v_lnext[self.T - 1], self.z[l][self.T - 1]
                ).T
            else:
                self.a[l][self.T - 1] = self._activation_update_Lminus1_T(
                    self.W[self.L - 1], u_lnext[self.T - 1], self.z[l][self.T - 1]
                ).T
            torch.cuda.synchronize()

            a_peak = torch.cuda.max_memory_allocated()
            metrics["act_time"] += time.time() - start
            metrics["act_mem"] = max(metrics["act_mem"], a_peak)
            metrics["global_peak"] = max(metrics["global_peak"], a_peak)

            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            q_lt = (
                self.W[l] @ output_spikes[self.T - 1].T
                + (
                    self.deltas * self.z[l][self.T - 2]
                    - self.thetas * self.a[l][self.T - 2]
                ).T
            )
            self.z[l][self.T - 1] = self.check_entries_T(
                q_lt, self.a[l][self.T - 1], q_lt
            ).T
            torch.cuda.synchronize()

            z_peak = torch.cuda.max_memory_allocated()
            metrics["z_time"] += time.time() - start
            metrics["z_mem"] = max(metrics["z_mem"], z_peak)
            metrics["global_peak"] = max(metrics["global_peak"], z_peak)

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        x_L = self.z[self.L - 1].clone()
        x_L[1:] -= self.deltas * x_L[:-1]
        self.W[self.L - 1] = self._weight_update_L(x_L, self.a[self.L - 2])
        torch.cuda.synchronize()

        w_peak = torch.cuda.max_memory_allocated()
        metrics["weight_time"] += time.time() - start
        metrics["weight_mem"] = max(metrics["weight_mem"], w_peak)
        metrics["global_peak"] = max(metrics["global_peak"], w_peak)

        for t in random_time_steps:
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
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
            torch.cuda.synchronize()

            z_peak = torch.cuda.max_memory_allocated()
            metrics["z_time"] += time.time() - start
            metrics["z_mem"] = max(metrics["z_mem"], z_peak)
            metrics["global_peak"] = max(metrics["global_peak"], z_peak)

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        s_LT = (
            self.W[self.L - 1] @ self.a[self.L - 2][self.T - 1].T
            + self.deltas * self.z[self.L - 1][self.T - 2].T
        )
        self.z[self.L - 1][self.T - 1] = self._z_update_L_T(s_LT, labels).T

        if not warming:
            self.lambda_lagrange = self._lambda_update(
                self.z[self.L - 1][self.T - 1],
                self.z[self.L - 1][self.T - 2],
                self.W[self.L - 1],
                self.a[self.L - 2][self.T - 1],
            ).T
        torch.cuda.synchronize()

        z_peak = torch.cuda.max_memory_allocated()
        metrics["z_time"] += time.time() - start
        metrics["z_mem"] = max(metrics["z_mem"], z_peak)
        metrics["global_peak"] = max(metrics["global_peak"], z_peak)

        return metrics

    def forward_model(self, inputs):
        inputs = inputs.to(self.device)
        potential = [torch.zeros_like(z_l).to(self.device) for z_l in self.z]

        for t in range(self.T):
            current_input = (
                inputs[t]
                if t < inputs.size(0)
                else torch.zeros_like(inputs[0]).to(self.device)
            )

            for l in range(self.L):
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


# ==========================================
# CUSTOM NEW MODEL TRACKER
# ==========================================
class ProfiledADMM(ADMM):
    def _fit_multi_block(
        self, layer_indices: List[int], time_steps: Optional[List[int]], warming: bool
    ) -> None:
        metrics = {
            "phase1_cov_time": 0.0,
            "phase1_cov_mem": 0.0,
            "phase2_weight_time": 0.0,
            "phase2_weight_mem": 0.0,
            "phase3_state_time": 0.0,
            "phase3_state_mem": 0.0,
            "global_peak": 0.0,
        }

        for layer_idx in layer_indices:
            layer = self.layers[layer_idx]

            # ---------------------------------------------
            # PHASE 1: COMPUTE COVARIANCES
            # ---------------------------------------------
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            self.cov_handler.reset_accumulators()
            for batch_id in self.state_handler.get_batch_ids():
                inputs, labels, batch_state = self.state_handler.load_batch(batch_id)
                state = batch_state.layer_states[layer_idx]
                a_prev = self._get_a_prev(
                    layer_idx=layer_idx, inputs=inputs, batch_state=batch_state
                )

                numerator, denominator, bias_sum, bias_count = (
                    layer.compute_batch_covariances(state=state, a_prev=a_prev)
                )
                self.cov_handler.accumulate(
                    layer_idx=layer_idx,
                    numerator=numerator,
                    denominator=denominator,
                    bias_sum=bias_sum,
                    bias_count=bias_count,
                )
            torch.cuda.synchronize()

            p1_peak = torch.cuda.max_memory_allocated()
            metrics["phase1_cov_time"] += time.time() - start_time
            metrics["phase1_cov_mem"] = max(metrics["phase1_cov_mem"], p1_peak)
            metrics["global_peak"] = max(metrics["global_peak"], p1_peak)

            # ---------------------------------------------
            # PHASE 2: WEIGHT & BIAS UPDATE
            # ---------------------------------------------
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            covariances = self.cov_handler.get_covariances(layer_idx)
            new_pinv = self._optimize_weights_and_biases(
                layer=layer, covariances=covariances, cache_pinv=(layer_idx == 0)
            )
            self.cov_handler.set_pinv(layer_idx=layer_idx, pinv=new_pinv)
            torch.cuda.synchronize()

            p2_peak = torch.cuda.max_memory_allocated()
            metrics["phase2_weight_time"] += time.time() - start_time
            metrics["phase2_weight_mem"] = max(metrics["phase2_weight_mem"], p2_peak)
            metrics["global_peak"] = max(metrics["global_peak"], p2_peak)

            # ---------------------------------------------
            # PHASE 3: STATE UPDATE
            # ---------------------------------------------
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            for batch_id in self.state_handler.get_batch_ids():
                inputs, labels, batch_state = self.state_handler.load_batch(batch_id)
                state = batch_state.layer_states[layer_idx]
                a_prev = self._get_a_prev(
                    layer_idx=layer_idx, inputs=inputs, batch_state=batch_state
                )

                self._optimize_states(
                    layer_idx=layer_idx,
                    layer=layer,
                    batch_state=batch_state,
                    state=state,
                    a_prev=a_prev,
                    labels=labels,
                    time_steps=time_steps,
                )
                if not warming:
                    layer.update_lambda(state=state, a_prev=a_prev)

                self.state_handler.save_batch(
                    batch_state=batch_state, inputs=inputs, labels=labels
                )
            torch.cuda.synchronize()

            p3_peak = torch.cuda.max_memory_allocated()
            metrics["phase3_state_time"] += time.time() - start_time
            metrics["phase3_state_mem"] = max(metrics["phase3_state_mem"], p3_peak)
            metrics["global_peak"] = max(metrics["global_peak"], p3_peak)

        self.last_metrics = metrics

    def fit(self, dataloader, warming=False):
        if not self.initialized:
            self.state_handler.initialize_all_batches(dataloader=dataloader)
            self.initialized = True

        time_steps = self._get_time_steps()
        layer_indices = self._get_layer_order()

        if self.config.block_method == "multi-block":
            self._fit_multi_block(layer_indices, time_steps, warming)
        elif self.config.block_method == "two-block":
            super()._fit_two_block(layer_indices, time_steps, warming)
        elif self.config.block_method == "distributed":
            super()._fit_distributed(layer_indices, time_steps, warming)

        return getattr(self, "last_metrics", {})


def format_mb(memory_bytes):
    return memory_bytes / (1024**2)


def get_tensor_memory_log():
    tensor_logs = []
    total_tracked_mb = 0.0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.nelement() / (1024**2)
                if size_mb > 0:
                    tensor_logs.append(
                        {
                            "shape": list(obj.size()),
                            "dtype": str(obj.dtype),
                            "memory_mb": round(size_mb, 4),
                        }
                    )
                    total_tracked_mb += size_mb
        except Exception:
            pass

    tensor_logs = sorted(tensor_logs, key=lambda x: x["memory_mb"], reverse=True)
    return tensor_logs, total_tracked_mb


def aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0].keys() if k != "global_peak"]
    aggregated = {}
    for k in keys:
        if "time" in k:
            aggregated[f"total_{k}"] = sum(m[k] for m in metrics_list)
        elif "mem" in k:
            aggregated[f"peak_{k}_mb"] = format_mb(max(m[k] for m in metrics_list))
    return aggregated


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("FATAL: CUDA is not available. You must run this on the cluster node!")
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensor_size = tonic.datasets.NMNIST.sensor_size

    n_timesteps = 100
    hidden_dims = [100]
    n_outputs = 10

    num_epochs = 10
    n_warming_iters = 5
    rho = 0.1
    beta = 0.1
    deltas = 0.95
    thetas = 1

    batch_sizes = [4, 8, 16, 32, 64, 128, 256]

    for batch_size in batch_sizes:
        seed = 8281003564
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

        frame_transform = transforms.Compose(
            [
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
            ]
        )

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

        data, targets_raw = next(iter(trainloader))
        data, targets_raw = data.to(device), targets_raw.to(device)

        data = data.view(data.size(0), data.size(1), -1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets_raw.to(torch.long), num_classes=10
        ).T

        if data.size(1) > n_timesteps:
            data = data[:, :n_timesteps, :]

        data = data.permute(1, 0, 2)
        torch.manual_seed(seed)
        data += 0.01 * torch.randn_like(data)

        new_framework_data = data.clone()
        new_framework_targets = targets_one_hot.T.clone()
        new_train_loader = [
            (new_framework_data.to(device), new_framework_targets.to(device))
        ]
        n_samples = data.size(1)

        #################################################################
        # OLD MODEL
        #################################################################
        input_dim = data.size(2)
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

        torch.cuda.synchronize()
        mem_init = format_mb(torch.cuda.memory_allocated())

        detailed_old_metrics = []

        # 1. Run the warmup epoch and IGNORE its times
        torch.cuda.reset_peak_memory_stats()
        old_model.fit(data, targets_one_hot, warming=False)
        torch.cuda.synchronize()
        mem_peak_1 = format_mb(torch.cuda.max_memory_allocated())

        # 2. START THE TOTAL TIMER NOW
        start_time = time.time()

        # 3. Track the next 9 epochs
        global_peak_all = 0
        for _ in range(9):
            m = old_model.fit(data, targets_one_hot, warming=False)
            global_peak_all = max(global_peak_all, m["global_peak"])
            detailed_old_metrics.append(m)

        torch.cuda.synchronize()
        mem_peak_10 = format_mb(global_peak_all)
        end_time = time.time()

        old_breakdown = aggregate_metrics(detailed_old_metrics)
        tensor_details, tracked_mb = get_tensor_memory_log()

        metrics = {
            "init_memory_mb": mem_init,
            "peak_1_epoch_mb": mem_peak_1,
            "peak_10_epoch_mb": mem_peak_10,
            "tracked_tensors_mb": tracked_mb,
            "time": end_time - start_time,
            "detailed_parts": old_breakdown,
            "tensor_objects": tensor_details,
        }

        metrics_filename = f"experiments/4_Performance_benchmark/results/detailed/Old/{batch_size}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        del old_model
        gc.collect()
        torch.cuda.empty_cache()

        #################################################################
        # LAYER CONFIGS (Reused for new models)
        #################################################################
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

        def get_spff_layers():
            return nn.ModuleList(
                [
                    ADMM_SpikingFeedForward(
                        in_f=input_dim,
                        out_f=hidden_dims[0],
                        h=ADMM_Heaviside(thetas=thetas),
                        config=hidden_layer_config,
                    ),
                    ADMM_SpikingFeedForward(
                        in_f=hidden_dims[0], out_f=10, h=None, config=out_layer_config
                    ),
                ]
            )

        #################################################################
        # UNROLLED (Default cache) - PROFILED
        #################################################################
        config = ADMM_Config(
            init="zeros",
            train_method="unrolled",
            time_order="sequential",
            layer_order="sequential",
            update_z_first=False,
            block_method="multi-block",
        )
        new_model = ProfiledADMM(
            get_spff_layers(), loss_f=ADMM_SSE(), T=n_timesteps, config=config
        ).to(device)

        torch.cuda.synchronize()
        mem_init = format_mb(torch.cuda.memory_allocated())

        detailed_new_metrics = []

        # 1. Run the warmup epoch and IGNORE its times
        torch.cuda.reset_peak_memory_stats()
        new_model.fit(new_train_loader, warming=False)
        torch.cuda.synchronize()
        mem_peak_1 = format_mb(torch.cuda.max_memory_allocated())

        # 2. START THE TOTAL TIMER NOW
        start_time = time.time()

        # 3. Track the next 9 epochs
        global_peak_all = 0
        for _ in range(9):
            m = new_model.fit(new_train_loader, warming=False)
            global_peak_all = max(global_peak_all, m["global_peak"])
            detailed_new_metrics.append(m)

        torch.cuda.synchronize()
        mem_peak_10 = format_mb(global_peak_all)
        end_time = time.time()

        new_breakdown = aggregate_metrics(detailed_new_metrics)
        tensor_details, tracked_mb = get_tensor_memory_log()

        metrics = {
            "init_memory_mb": mem_init,
            "peak_1_epoch_mb": mem_peak_1,
            "peak_10_epoch_mb": mem_peak_10,
            "tracked_tensors_mb": tracked_mb,
            "time": end_time - start_time,
            "detailed_parts": new_breakdown,
            "tensor_objects": tensor_details,
        }

        metrics_filename = f"experiments/4_Performance_benchmark/results/detailed/New/{batch_size}/results.json"
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

        del new_model
        gc.collect()
        torch.cuda.empty_cache()
