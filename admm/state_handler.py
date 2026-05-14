r"""
This module contains the state handler for ADMM.

In the standard Backpropagation framework, intermediate activations are not important
and safely discarded after the backward pass. However, in the ADMM framework, local
sequence variables ($z$, $a$, and $\lambda$) are treated as optimizable parameters
that must persist and evolve across epochs. This module handles the immense memory
footprint of these variables by seamlessly orchestrating their caching and retrieval
between RAM and persistent disk storage.
"""

import os
from typing import List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from .dataclasses import ADMM_BatchState, ADMM_LayerState
from .initializers import get_initializer


class ADMM_StateHandler:
    """Manages the persistence of ADMM variables (z, a, lambda) across epochs.

    Automatically swaps between the states in RAM and Disk to handle multiple batches.
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        init_strategy: str = "s-uniform",
        cache_dir: str = "./admm_cache",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initializes the State Handler.

        Args:
            layers (torch.nn.ModuleList): The network layers corresponding to the states.
            init_strategy (str, optional): The initialization strategy for ADMM states. Defaults to "s-uniform".
            cache_dir (Union[str, Path], optional): Directory to save disk caches. Defaults to "./admm_cache".
            device (Optional[Union[str, torch.device]], optional): The execution device.
        """
        self.device = device
        self.layers = layers
        self.init_strategy = init_strategy
        self.cache_dir = cache_dir
        self.num_batches = 0

        self.in_memory = False
        self.memory_cache = {}

    def get_batch_ids(self) -> List[int]:
        """Returns a list of all initialized batch IDs."""
        return list(range(self.num_batches))

    def initialize_all_batches(self, dataloader: DataLoader) -> None:
        """Iterates through the DataLoader, creates initial states, and persists them.

        Automatically decides whether to use only RAM or Disk based on batch count.

        Args:
            dataloader (DataLoader): The dataset loader to process.
        """

        self.in_memory = len(dataloader) == 1

        if not self.in_memory:
            os.makedirs(self.cache_dir, exist_ok=True)

        initializer = get_initializer(self.init_strategy)

        self.num_batches = 0
        for batch_id, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            layer_states = initializer.init_states(self.layers, inputs, self.device)
            batch_state = ADMM_BatchState(batch_id=batch_id, layer_states=layer_states)

            self.save_batch(batch_state, inputs, labels)
            self.num_batches += 1

    def save_batch(
        self,
        batch_state: ADMM_BatchState,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Serializes a batch state either to RAM or the hard drive.

        Args:
            batch_state (ADMM_BatchState): The current state of the ADMM variables.
            inputs (Optional[torch.Tensor], optional): Original network inputs.
            labels (Optional[torch.Tensor], optional): Ground truth labels.
        """
        batch_id = batch_state.batch_id

        if self.in_memory:
            # Preserve inputs/labels if they already exist in RAM
            if inputs is None or labels is None:
                existing = self.memory_cache.get(batch_id, {})
                inputs = existing.get("inputs")
                labels = existing.get("labels")

            self.memory_cache[batch_id] = {
                "batch_state": batch_state,
                "inputs": inputs,
                "labels": labels,
            }
        else:
            tensors_to_save = {}
            filepath = os.path.join(self.cache_dir, f"batch_{batch_id}.safetensors")

            if inputs is None or labels is None:
                existing = load_file(filepath, device=str(self.device))
                tensors_to_save["inputs"] = existing["inputs"].clone()
                tensors_to_save["labels"] = existing["labels"].clone()
                del existing
            else:
                tensors_to_save["inputs"] = inputs
                tensors_to_save["labels"] = labels

            for layer_idx, l_state in enumerate(batch_state.layer_states):
                tensors_to_save[f"layer_{layer_idx}_z"] = l_state.z
                if l_state.a is not None:
                    tensors_to_save[f"layer_{layer_idx}_a"] = l_state.a
                if l_state.lambda_lagrange is not None:
                    tensors_to_save[f"layer_{layer_idx}_lambda_lagrange"] = (
                        l_state.lambda_lagrange
                    )

            save_file(tensors_to_save, filepath)

    def load_batch(
        self, batch_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, ADMM_BatchState]:
        """Loads a batch from RAM or disk into GPU RAM.

        Args:
            batch_id (int): The integer identifier for the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, ADMM_BatchState]: A tuple containing the
                batch inputs, batch labels, and the populated ADMM batch state.
        """
        if self.in_memory:
            cached = self.memory_cache[batch_id]
            return cached["inputs"], cached["labels"], cached["batch_state"]

        filepath = os.path.join(self.cache_dir, f"batch_{batch_id}.safetensors")
        tensors = load_file(filepath, device=str(self.device))

        inputs = tensors["inputs"].clone()
        labels = tensors["labels"].clone()

        layer_states = []
        for layer_idx in range(len(self.layers)):
            z = tensors[f"layer_{layer_idx}_z"].clone()

            a = tensors.get(f"layer_{layer_idx}_a", None)
            if a is not None:
                a = a.clone()

            lambda_lagrange = tensors.get(f"layer_{layer_idx}_lambda_lagrange", None)
            if lambda_lagrange is not None:
                lambda_lagrange = lambda_lagrange.clone()

            layer_states.append(
                ADMM_LayerState(z=z, a=a, lambda_lagrange=lambda_lagrange)
            )

        del tensors

        batch_state = ADMM_BatchState(batch_id=batch_id, layer_states=layer_states)
        return inputs, labels, batch_state
