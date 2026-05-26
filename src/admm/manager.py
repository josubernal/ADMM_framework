"""This module contains the core orchestrator for the network and ADMM optimizer.
ADMM breaks the network down into layer-wise sub-problems and updates in a loop.

The manager handles both static and spiking networks, automatically adjusting the optimization loop based on the network type and selected training method.
"""

import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .covariance_handler import ADMM_CovarianceHandler
from .dataclasses import (
    ADMM_BatchState,
    ADMM_Config,
    ADMM_LayerCovariance,
    ADMM_LayerState,
    TemporalCache,
)
from .loss_functions import ADMM_SSE
from .state_handler import ADMM_StateHandler


class ADMM(nn.Module):
    """Universal Singlebatch and Multibatch Manager for ADMM networks.

    Handles both multibatch and singlebatch updates for Static and Spiking ADMM networks automatically, orchestrating
    the layer-wise alternating optimization loops. It manages the global
    hyperparameters and delegates the mathematical proximal updates to the
    individual layers, activation functions or loss functions.

    Args:
        layers (nn.ModuleList): The sequential list of ADMM layers comprising the network.
        loss_f (nn.Module, optional): The [ADMM-compatible objective function][src.admm.loss_functions]. Defaults to [ADMM_SSE()][src.admm.loss_functions.ADMM_SSE].
        T (int, optional): The number of time steps for spiking networks. If None, assumes static network. Defaults to None.
        config (ADMM_Config, optional): [Global configuration][src.admm.dataclasses.ADMM_Config] for ADMM training. Defaults to standard settings.
        device (torch.device, optional): The computing device. If None, auto-detects CUDA. Defaults to None.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        loss_f: Optional[nn.Module] = None,
        T: Optional[int] = None,
        config: Optional[ADMM_Config] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.loss_f = loss_f if loss_f is not None else ADMM_SSE()
        self.layers = layers
        self.L = len(self.layers)

        self.T = T
        self.is_spiking = self.T is not None
        self.initialized = False
        self.config = config if config is not None else ADMM_Config()

        # Separates z and a from layers.
        # Having states isolated in an object allows multibatch handling and loss and activation function modularity.
        self.state_handler = ADMM_StateHandler(
            layers=self.layers,
            init_strategy=self.config.init,
            device=self.device,
        )

        # Separates covariances from layers.
        # This allows multibatch handling.
        self.cov_handler = ADMM_CovarianceHandler(num_layers=self.L)

        if not self.is_spiking and self.config.train_method != "vectorized":
            warnings.warn(
                "Non-spiking networks can only be trained using the 'vectorized' method. "
                "Automatically switching train_method to 'vectorized'.",
                UserWarning,
            )
            self.config.train_method = "vectorized"

        self._configure_layers()

    def _configure_layers(self) -> None:
        """Initializes each layer.

        Provides the [global configuration][src.admm.dataclasses.ADMM_Config] to each layer for consistent initializations across the network.
        """
        for layer in self.layers:
            layer.device = self.device
            layer._setup(global_config=self.config)

    def _get_time_steps(self) -> Optional[List[int]]:
        """Determines the sequence of time-steps depending on the [selected training method][src.admm.dataclasses.ADMM_Config].

        Returns:
            Optional[List[int]]: A list of time steps if the network is spiking, otherwise None.
        """
        time_steps = None
        if self.is_spiking:
            if self.config.train_method.endswith("random"):
                time_steps = random.sample(range(self.T), self.T)
            elif self.config.train_method.endswith("sequential"):
                time_steps = list(range(self.T))
            elif self.config.train_method.endswith("backwards"):
                time_steps = list(range(self.T - 1, -1, -1))
        return time_steps

    def _get_layer_order(self) -> List[int]:
        """Determines the layer processing order based on the [global configuration][src.admm.dataclasses.ADMM_Config].

        Returns:
            List[int]: An ordered list of layer indices.
        """
        if self.config.layer_order == "backwards":
            layer_indices = list(range(self.L - 1, -1, -1))
        elif self.config.layer_order == "random-last":
            layer_indices = random.sample(range(self.L - 1), self.L - 1)
            layer_indices.append(self.L - 1)
        elif self.config.layer_order == "random":
            layer_indices = list(range(self.L))
            random.shuffle(layer_indices)
        elif self.config.layer_order == "sequential":
            layer_indices = list(range(self.L))
        return layer_indices

    def _get_a_prev(
        self, layer_idx: int, inputs: torch.Tensor, batch_state: ADMM_BatchState
    ) -> torch.Tensor:
        """Retrieves the network input or the previous layer's activations.

        Args:
            layer_idx (int): The index of the current layer being processed.
            inputs (torch.Tensor): The raw input data batch fed to the network.
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.

        Returns:
            torch.Tensor: The activation tensor to be used as input for the current layer.
        """
        return inputs if layer_idx == 0 else batch_state.layer_states[layer_idx - 1].a

    def _optimize_weights_and_biases(
        self,
        layer: nn.Module,
        covariances: ADMM_LayerCovariance,
        cache_pinv: bool = False,
    ) -> Optional[torch.Tensor]:
        """Unified interface for updating all trainable parameters (Weights and Biases).

        Args:
            layer (nn.Module): The specific layer to update.
            covariances (ADMM_LayerCovariance): [Covariance object][src.admm.dataclasses.ADMM_LayerCovariance] containing the precomputed numerator and denominator matrices.
            cache_pinv (bool, optional): Whether to cache and return the pseudoinverse for future updates. Defaults to False.

        Returns:
             Optional[torch.Tensor]: The cached pseudoinverse tensor if `cache_pinv` is True, otherwise None.
        """
        new_pinv = layer.update_weights(covariances=covariances, cache_pinv=cache_pinv)
        if getattr(layer.config, "use_bias", False):
            layer.update_bias(covariances=covariances)
        return new_pinv

    def _optimize_states(
        self,
        layer_idx: int,
        layer: nn.Module,
        batch_state: ADMM_BatchState,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
        labels: torch.Tensor,
        time_steps: Optional[List[int]],
    ) -> None:
        """Routes the state optimization logic based on the layer's position in the network.

        Args:
            layer_idx (int): The index of the current layer.
            layer (nn.Module): The current layer module.
            batch_state (ADMM_BatchState): The [global state][src.admm.dataclasses.ADMM_BatchState] object containing all layer states for the batch.
            state (ADMM_LayerState): The [specific state][src.admm.dataclasses.ADMM_LayerState] object for the current layer.
            a_prev (torch.Tensor): The activations from the preceding layer.
            labels (torch.Tensor): The ground truth labels for the final layer loss calculation.
            time_steps (Optional[List[int]]): Sequence of time steps for spiking networks.
        """
        if layer_idx < self.L - 1:
            next_layer = self.layers[layer_idx + 1]
            next_state = batch_state.layer_states[layer_idx + 1]
            self._optimize_a_and_z(
                layer=layer,
                next_layer=next_layer,
                next_state=next_state,
                state=state,
                a_prev=a_prev,
                time_steps=time_steps,
            )

        else:
            self._optimize_z_last(
                layer=layer,
                state=state,
                a_prev=a_prev,
                labels=labels,
                time_steps=time_steps,
            )

    def _create_cache(
        self,
        layer: nn.Module,
        state: ADMM_LayerState,
        next_layer: nn.Module,
        next_state: ADMM_LayerState,
        a_prev: torch.Tensor,
    ) -> TemporalCache:
        """Delegates cache construction to the [TemporalCache][src.admm.dataclasses.TemporalCache].

        Args:
            layer (nn.Module): The current layer being optimized.
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the current layer.
            next_layer (nn.Module): The subsequent layer in the network.
            next_state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the subsequent layer.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            TemporalCache: A typed data class containing the precomputed temporal matrices.
        """
        return TemporalCache.build(
            layer=layer,
            next_layer=next_layer,
            next_state=next_state,
            state=state,
            a_prev=a_prev,
        )

    def _optimize_a_and_z(
        self,
        layer: nn.Module,
        state: ADMM_LayerState,
        next_layer: nn.Module,
        next_state: ADMM_LayerState,
        a_prev: torch.Tensor,
        time_steps: Optional[List[int]] = None,
    ) -> None:
        """Unified interface for updating activations ($a$) and pre-activations ($z$).

        Automatically routes the execution to unrolled, decoupled, or vectorized
        solvers based on the [global configuration][src.admm.dataclasses.ADMM_Config].

        Args:
            layer (nn.Module): The current layer being optimized.
            state (ADMM_LayerState):  [State object][src.admm.dataclasses.ADMM_LayerState] of the current layer.
            next_layer (nn.Module): The subsequent layer in the network.
            next_state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] of the next layer.
            a_prev (torch.Tensor): Activations from the previous layer.
            time_steps (Optional[List[int]], optional): List of timesteps for SNN simulation. Defaults to None.
        """
        # --- PATH 1: UNROLLED ---
        if self.config.train_method.startswith("unrolled") and getattr(
            layer, "spiking", False
        ):
            cache = self._create_cache(
                layer=layer,
                state=state,
                next_layer=next_layer,
                next_state=next_state,
                a_prev=a_prev,
            )
            for t in time_steps:
                if self.config.update_z_first:
                    layer.h.update_z_unrolled(
                        t=t, cache=cache, state=state, config=layer.config
                    )
                    layer.update_a_unrolled(
                        t=t, cache=cache, next_layer=next_layer, state=state
                    )
                else:
                    layer.update_a_unrolled(
                        t=t, cache=cache, next_layer=next_layer, state=state
                    )
                    layer.h.update_z_unrolled(
                        t=t, cache=cache, state=state, config=layer.config
                    )

        # --- PATH 2: DECOUPLED ---
        elif self.config.train_method.startswith("decoupled") and getattr(
            layer, "spiking", False
        ):
            # Activation functions are not capable of computing layer dynamics. We must pass this as an argument.
            forward = layer.spatial_forward(a_prev)

            if self.config.update_z_first:
                layer.h.update_z_decoupled(
                    state=state,
                    forward=forward,
                    time_steps=time_steps,
                    config=layer.config,
                )
                layer.update_a(
                    state=state,
                    next_layer=next_layer,
                    next_state=next_state,
                    a_prev=a_prev,
                )
            else:
                layer.update_a(
                    state=state,
                    next_layer=next_layer,
                    next_state=next_state,
                    a_prev=a_prev,
                )
                layer.h.update_z_decoupled(
                    state=state,
                    forward=forward,
                    time_steps=time_steps,
                    config=layer.config,
                )

        # --- PATH 3: VECTORIZED  ---
        else:
            # Activation functions are not capable of computing layer dynamics. We must pass this as an argument.
            forward = layer.spatial_forward(a_prev)

            if self.config.update_z_first:
                layer.h.update_z(state=state, forward=forward, config=layer.config)
                layer.update_a(
                    state=state,
                    next_layer=next_layer,
                    next_state=next_state,
                    a_prev=a_prev,
                )
            else:
                layer.update_a(
                    state=state,
                    next_layer=next_layer,
                    next_state=next_state,
                    a_prev=a_prev,
                )
                layer.h.update_z(state=state, forward=forward, config=layer.config)

    def _optimize_z_last(
        self,
        layer: nn.Module,
        state: ADMM_LayerState,
        a_prev: torch.Tensor,
        labels: torch.Tensor,
        time_steps: Optional[List[int]] = None,
    ) -> None:
        """Handles state optimization specifically for the final layer's pre-activations ($z_L$).

        Delegates to the network's configured loss function to apply label-based proximal updates.

        Args:
            layer (nn.Module): The final layer of the network.
            state (ADMM_LayerState): [State object][src.admm.dataclasses.ADMM_LayerState] payload for the final layer.
            a_prev (torch.Tensor): The activations from the preceding layer.
            labels (torch.Tensor): The ground truth labels.
            time_steps (Optional[List[int]], optional): Sequence of time steps for spiking networks. Defaults to None.
        """
        # Loss functions are not capable of computing layer dynamics. We must pass this as an argument.
        forward = layer.spatial_forward(a_prev)

        if self.config.train_method != "vectorized":
            self.loss_f.update_z_last_unrolled(
                state=state,
                forward=forward,
                labels=labels,
                config=layer.config,
                time_steps=time_steps,
            )
        else:
            self.loss_f.update_z_last(
                state=state,
                forward=forward,
                labels=labels,
                config=layer.config,
                spiking=self.is_spiking,
            )

    @torch.no_grad()
    def forward_model(
        self, inputs: torch.Tensor, return_firing_rates: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[float]]]:
        """Standard Feed-Forward pass for initialization or evaluation.

        Args:
            inputs (torch.Tensor): The input tensor to the network.
            return_firing_rates (bool, optional): Whether to compute and return spiking firing rates. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[float]]]:
                If `return_firing_rates` is False, returns only the final network output (`torch.Tensor`).
                If True, returns a tuple containing:
                - **torch.Tensor**: The final network output.
                - **List[float]**: The activation firing rates per layer.
        """
        x = inputs.to(self.device)
        firing_rates = []

        # PATH 1: SPIKING NETWORK
        if self.is_spiking:
            batch_size = x.size(1)

            # We save the current state to pass it to the next time-step
            z_current = [None] * self.L
            a_current = [None] * self.L

            spike_sums = [0.0] * (self.L - 1)
            spike_counts = [0] * (self.L - 1)

            for t in range(self.T):
                a_prev = x[t]

                for i, layer in enumerate(self.layers):
                    z_t = layer.forward(
                        a_prev=a_prev, z_tminus1=z_current[i], a_tminus1=a_current[i]
                    )
                    z_current[i] = z_t
                    a_t = layer.h(z_t)
                    a_current[i] = a_t
                    a_prev = a_t

                    if return_firing_rates and i < self.L - 1:
                        spike_sums[i] += a_t.sum().item()
                        spike_counts[i] += a_t.numel()

            final_out = z_current[-1]

            if return_firing_rates:
                firing_rates = [s / c for s, c in zip(spike_sums, spike_counts)]

        # PATH 2: STATIC NETWORK
        else:
            batch_size = x.size(0)
            final_z = None

            for i, layer in enumerate(self.layers):
                z_pred = layer.spatial_forward(x)
                x = layer.h(z_pred)
                final_z = z_pred

            final_out = final_z

        if final_out.dim() > 2:
            final_out = final_out.view(batch_size, -1)

        if return_firing_rates:
            return final_out, firing_rates
        return final_out

    @torch.no_grad()
    def fit(self, dataloader: DataLoader, warming: bool = False) -> None:
        """Orchestrates the global fitting loop for the ADMM optimization process.

        Args:
            dataloader (Any): A PyTorch DataLoader yielding inputs and labels.
            warming (bool, optional): If True, bypasses the Lagrange multiplier update to stabilize initial matrices. Defaults to False.
        """
        # Initialization
        if not self.initialized:
            self.state_handler.initialize_all_batches(dataloader=dataloader)
            self.initialized = True

        time_steps = self._get_time_steps()
        layer_indices = self._get_layer_order()

        # Route to the selected mathematical optimization strategy
        if self.config.block_method == "two-block":
            self._fit_two_block(layer_indices, time_steps, warming)
        elif self.config.block_method == "multi-block":
            self._fit_multi_block(layer_indices, time_steps, warming)
        else:
            raise ValueError(f"Unknown update mode: {self.config.update_mode}")

    def _fit_two_block(
        self, layer_indices: List[int], time_steps: Optional[List[int]], warming: bool
    ) -> None:
        """Executes the two-phase ADMM algorithm:

        - Global parameter updates (Weights and Biases).
        - Local state updates (Activations, Pre-activations, and Lagrange multipliers).

        Args:
            layer_indices (List[int]): An ordered list of layer indices.
            time_steps (Optional[List[int]], optional): List of timesteps for SNN simulation. Defaults to None.
            warming (bool, optional): If True, bypasses the Lagrange multiplier update to stabilize initial matrices. Defaults to False.
        """
        # PHASE 1: COMPUTE COVARIANCES
        self.cov_handler.reset_accumulators()

        for batch_id in self.state_handler.get_batch_ids():
            inputs, labels, batch_state = self.state_handler.load_batch(batch_id)

            for layer_idx in layer_indices:
                layer = self.layers[layer_idx]
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

        # PHASE 2: GLOBAL WEIGHT & BIAS UPDATE
        for layer_idx in layer_indices:
            layer = self.layers[layer_idx]
            covariances = self.cov_handler.get_covariances(layer_idx)
            new_pinv = self._optimize_weights_and_biases(
                layer=layer, covariances=covariances, cache_pinv=(layer_idx == 0)
            )
            self.cov_handler.set_pinv(layer_idx=layer_idx, pinv=new_pinv)

        # PHASE 3: LOCAL STATE UPDATES (a, z, lambda)
        for batch_id in self.state_handler.get_batch_ids():
            inputs, labels, batch_state = self.state_handler.load_batch(batch_id)

            for layer_idx in layer_indices:
                layer = self.layers[layer_idx]
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
                    layer.update_lambda(state, a_prev)

            self.state_handler.save_batch(batch_state, inputs=inputs, labels=labels)

    def _fit_multi_block(
        self, layer_indices: List[int], time_steps: Optional[List[int]], warming: bool
    ) -> None:
        """ "Executes the multi-phase ADMM algorithm: Updates weights then states sequentially per layer.

        Args:
           layer_indices (List[int]): An ordered list of layer indices.
           time_steps (Optional[List[int]], optional): List of timesteps for SNN simulation. Defaults to None.
           warming (bool, optional): If True, bypasses the Lagrange multiplier update to stabilize initial matrices. Defaults to False.
        """

        for layer_idx in layer_indices:
            layer = self.layers[layer_idx]

            # PHASE 1: COMPUTE COVARIANCES FOR THIS SPECIFIC LAYER
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

            # PHASE 2: WEIGHT & BIAS UPDATE FOR THIS SPECIFIC LAYER
            covariances = self.cov_handler.get_covariances(layer_idx)
            new_pinv = self._optimize_weights_and_biases(
                layer=layer, covariances=covariances, cache_pinv=(layer_idx == 0)
            )
            self.cov_handler.set_pinv(layer_idx=layer_idx, pinv=new_pinv)

            # PHASE 3: STATE UPDATE FOR THIS SPECIFIC LAYER
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
