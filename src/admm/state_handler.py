r"""
This module contains the state handler for ADMM.

In the standard Backpropagation framework, intermediate activations are not important
and safely discarded after the backward pass. However, in the ADMM framework, local
sequence variables ($z$, $a$, and $\lambda$) are treated as optimizable parameters
that must persist and evolve across epochs. This module handles
these variables by seamlessly orchestrating their caching and retrieval
between RAM and persistent disk storage, allowing single-batch and multibatch
training to be agnostic to the user.
"""

import os
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Union

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
        async_writes: bool = True,
        max_queue_size: int = 2,
        num_workers: int = 1,
    ):
        """Initializes the State Handler.

        Args:
            layers (torch.nn.ModuleList): The network layers corresponding to the states.
            init_strategy (str, optional): The initialization strategy for ADMM states. Defaults to "s-uniform".
            cache_dir (Union[str, Path], optional): Directory to save disk caches. Defaults to "./admm_cache".
            device (Optional[Union[str, torch.device]], optional): The execution device.
        """
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.layers = layers
        self.init_strategy = init_strategy
        self.cache_dir = os.path.join(cache_dir, f"run_{uuid.uuid4().hex}")
        self.num_batches = 0
        self._writer = None
        self.in_memory = False
        self.memory_cache = {}
        self.async_writes = async_writes
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self._initialized_batches: set[int] = set()
        self._initializer = get_initializer(self.init_strategy)

    def is_batch_initialized(self, batch_id: int) -> bool:
        """Return whether a persistent state already exists for this batch."""
        return batch_id in self._initialized_batches

    def initialize(self, dataloader: DataLoader) -> None:
        """Prepare the state handler without creating any batch states.

        Batch states are initialized lazily when they are first requested.
        """
        if self._writer is not None:
            self._writer.shutdown()
            self._writer = None
        # Keep the existing single-batch optimization.
        self.in_memory = len(dataloader) == 1

        # Disk-backed mode needs the cache directory.
        if not self.in_memory:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._writer = (
                AsyncWriter(
                    max_queue_size=self.max_queue_size,
                    num_workers=self.num_workers,
                )
                if self.async_writes
                else None
            )

        self.num_batches = 0
        self._initialized_batches.clear()
        self.memory_cache.clear()

    def _initialize_batch(
        self,
        batch_id: int,
        inputs: torch.Tensor,
    ) -> ADMM_BatchState:
        """Create the initial ADMM state for one batch."""

        if inputs.device != self.device:
            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )

        layer_states = self._initializer.init_states(
            self.layers,
            inputs,
            self.device,
        )

        return ADMM_BatchState(
            batch_id=batch_id,
            layer_states=layer_states,
        )

    def save_batch(
        self,
        batch_state: ADMM_BatchState,
    ) -> None:
        """Serializes a batch state either to RAM or the hard drive.

        Args:
            batch_state (ADMM_BatchState): The [current state][src.admm.dataclasses.ADMM_BatchState] of the ADMM variables.
        """
        batch_id = batch_state.batch_id

        if self.in_memory:  # Single batch route
            self.memory_cache[batch_id] = {
                "batch_state": batch_state,
            }
        else:
            tensors_to_save = {}
            filepath = os.path.join(self.cache_dir, f"batch_{batch_id}.safetensors")

            for layer_idx, l_state in enumerate(batch_state.layer_states):
                tensors_to_save[f"layer_{layer_idx}_z"] = l_state.z
                if l_state.a is not None:
                    tensors_to_save[f"layer_{layer_idx}_a"] = l_state.a
                if l_state.lambda_lagrange is not None:
                    tensors_to_save[f"layer_{layer_idx}_lambda_lagrange"] = (
                        l_state.lambda_lagrange
                    )

            if self._writer is not None:
                cpu_tensors = {
                    key: value.detach().to("cpu", copy=True)
                    for key, value in tensors_to_save.items()
                }

                self._writer.enqueue(
                    cpu_tensors,
                    filepath,
                )

            else:
                save_file(tensors_to_save, filepath)

        self._initialized_batches.add(batch_id)

        self.num_batches = max(
            self.num_batches,
            batch_id + 1,
        )

    def _load_batch_from_disk(self, batch_id: int) -> ADMM_BatchState:
        """Loads a batch state from disk. Caller must not use this in in-memory mode."""
        filepath = os.path.join(
            self.cache_dir,
            f"batch_{batch_id}.safetensors",
        )

        if self._writer is not None:
            self._writer.flush(filepath)

        tensors = load_file(
            filepath,
            device=str(self.device),
        )

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
        return ADMM_BatchState(batch_id=batch_id, layer_states=layer_states)

    def load_batch(self, batch_id: int, inputs: torch.Tensor) -> ADMM_BatchState:
        if batch_id not in self._initialized_batches:
            return self._initialize_batch(
                batch_id=batch_id,
                inputs=inputs,
            )

        if self.in_memory:
            return self.memory_cache[batch_id]["batch_state"]
        return self._load_batch_from_disk(batch_id)


class AsyncWriter:
    """Asynchronously writes tensors to disk using a bounded thread pool.

    The class provides three guarantees:

    1. Writes happen in background threads.
    2. The number of outstanding writes is bounded, preventing unbounded RAM usage.
    3. Two writes to the same filepath are never active simultaneously.

    Exceptions raised by background writes are propagated through ``flush()``.
    """

    def __init__(
        self,
        max_queue_size: int = 2,
        num_workers: int = 1,
    ):
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1.")

        if num_workers < 1:
            raise ValueError("num_workers must be >= 1.")

        self._num_workers = num_workers

        # Keep the same semantics as the previous implementation:
        # ``max_queue_size`` items could wait in the queue, while
        # ``num_workers`` items could already be running.
        self._max_in_flight = max_queue_size + num_workers

        # Bounds the number of submitted-but-not-finished writes.
        # This is important because each queued task owns references
        # to the CPU tensors that it is going to write.
        self._slots = threading.BoundedSemaphore(self._max_in_flight)

        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="admm-writer",
        )

        # At most one pending Future is allowed for a given filepath.
        self._futures: Dict[str, Future] = {}

        # Protects ``_futures`` and ``_shutdown``.
        self._lock = threading.Lock()

        self._shutdown = False

    @staticmethod
    def _normalize_path(filepath: str) -> str:
        """Convert a path to a stable absolute representation."""
        return os.path.abspath(filepath)

    @staticmethod
    def _write_file(
        tensors_dict: dict,
        filepath: str,
    ) -> None:
        """Write a complete file atomically.

        The data is first written to a temporary file in the same directory.
        Once that succeeds, ``os.replace`` swaps it into place.
        """

        directory = os.path.dirname(filepath) or "."
        os.makedirs(directory, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(filepath) + ".tmp-",
            dir=directory,
        )
        os.close(fd)

        try:
            save_file(tensors_dict, tmp_path)

            # Atomic replacement on the same filesystem.
            os.replace(tmp_path, filepath)

        finally:
            # Remove the temporary file if something failed before os.replace.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def enqueue(
        self,
        tensors_dict: dict,
        filepath: str,
    ) -> Future:
        """Schedule a background write.

        Blocks when the bounded in-flight window is full.

        If another write to the same filepath is still pending, this method
        waits for that write to finish before scheduling the new one.
        """

        filepath = self._normalize_path(filepath)

        with self._lock:
            if self._shutdown:
                raise RuntimeError("AsyncWriter has already been shut down.")

            # Prevent overlapping writes to the same file.
            previous = self._futures.get(filepath)

            if previous is not None:
                try:
                    previous.result()
                except Exception as exc:
                    raise RuntimeError(f"Async write failed for '{filepath}'.") from exc
                finally:
                    # Remove the old completed Future, whether it succeeded
                    # or failed, so a failed write doesn't permanently block
                    # every future write to this same file.
                    if self._futures.get(filepath) is previous:
                        del self._futures[filepath]

            # Apply backpressure before submitting another task.
            self._slots.acquire()

            try:
                future = self._executor.submit(
                    self._write_file,
                    tensors_dict,
                    filepath,
                )
            except BaseException:
                # If submission failed, give the slot back.
                self._slots.release()
                raise

            self._futures[filepath] = future

            # Free one slot when the write finishes, regardless of success/failure.
            future.add_done_callback(lambda _: self._slots.release())

            return future

    def flush(self, filepath: Optional[str] = None) -> None:
        """Wait for pending writes.

        Args:
            filepath:
                If provided, wait only for that file.
                If None, wait for every pending write.

        Any background exception is re-raised in the calling thread.
        """

        if filepath is not None:
            filepath = self._normalize_path(filepath)

            with self._lock:
                future = self._futures.get(filepath)

            if future is None:
                return

            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Async write failed for '{filepath}'.") from exc
            finally:
                with self._lock:
                    if self._futures.get(filepath) is future:
                        del self._futures[filepath]

            return

        # Flush all currently known files.
        while True:
            with self._lock:
                pending = list(self._futures.items())

            if not pending:
                return

            first_error = None

            for filepath, future in pending:
                try:
                    future.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = (filepath, exc)
                finally:
                    with self._lock:
                        if self._futures.get(filepath) is future:
                            del self._futures[filepath]

            if first_error is not None:
                filepath, exc = first_error
                raise RuntimeError(f"Async write failed for '{filepath}'.") from exc

    def shutdown(self) -> None:
        """Flush all pending writes and shut down the executor."""

        with self._lock:
            if self._shutdown:
                return

            self._shutdown = True

        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


class AsyncStatePrefetcher:
    """Asynchronously prefetches ADMM batch states from disk.

    The manager uses this class with a one-batch lookahead:

        current batch → being consumed
        next batch    → being loaded

    ``max_in_flight`` bounds the number of submitted loads.
    """

    def __init__(
        self,
        state_handler,
        max_in_flight: int = 2,
        num_workers: int = 1,
    ):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1.")

        if num_workers < 1:
            raise ValueError("num_workers must be >= 1.")

        self._handler = state_handler

        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="admm-reader",
        )

        self._slots = threading.BoundedSemaphore(max_in_flight)

        # Only the main training thread accesses this dictionary.
        self._futures: Dict[int, Future] = {}

        self._shutdown = False

    def schedule(self, batch_id: int) -> None:
        """Start loading a batch in the background."""

        if self._shutdown:
            raise RuntimeError("AsyncStatePrefetcher has already been shut down.")

        if batch_id in self._futures:
            raise RuntimeError(f"Batch {batch_id} is already scheduled.")

        # Apply backpressure.
        self._slots.acquire()

        try:
            future = self._executor.submit(
                self._handler._load_batch_from_disk,
                batch_id,
            )
        except BaseException:
            self._slots.release()
            raise

        self._futures[batch_id] = future

    def get(self, batch_id: int) -> ADMM_BatchState:
        """Wait for a previously scheduled batch and return its state."""

        future = self._futures.pop(batch_id, None)

        if future is None:
            raise RuntimeError(f"Batch {batch_id} was not scheduled.")

        try:
            return future.result()
        finally:
            # Whether loading succeeded or failed, one slot is now free.
            self._slots.release()

    def shutdown(self) -> None:
        """Stop the reader threads."""

        if self._shutdown:
            return

        self._shutdown = True

        self._executor.shutdown(
            wait=True,
            cancel_futures=True,
        )
