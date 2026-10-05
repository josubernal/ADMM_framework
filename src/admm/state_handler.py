# r"""
# This module contains the state handler for ADMM.

# In the standard Backpropagation framework, intermediate activations are not important
# and safely discarded after the backward pass. However, in the ADMM framework, local
# sequence variables ($z$, $a$, and $\lambda$) are treated as optimizable parameters
# that must persist and evolve across epochs. This module handles
# these variables by seamlessly orchestrating their caching and retrieval
# between RAM and persistent disk storage, allowing single-batch and multibatch
# training to be agnostic to the user.
# """

# import os
# import tempfile
# import threading
# import time
# import uuid
# from concurrent.futures import Future, ThreadPoolExecutor
# from typing import Dict, Optional, Union

# import torch
# from safetensors.torch import load_file, save_file
# from torch.utils.data import DataLoader

# from .dataclasses import ADMM_BatchState, ADMM_LayerState
# from .initializers import get_initializer


# class ADMM_StateHandler:
#     """Manages the persistence of ADMM variables (z, a, lambda) across epochs.

#     Automatically swaps between the states in RAM and Disk to handle multiple batches.
#     """

#     def __init__(
#         self,
#         layers: torch.nn.ModuleList,
#         init_strategy: str = "s-uniform",
#         cache_dir: str = "./admm_cache",
#         device: Optional[Union[str, torch.device]] = None,
#         async_writes: bool = True,
#         max_queue_size: int = 1,
#         num_workers: int = 1,
#     ):
#         """Initializes the State Handler.

#         Args:
#             layers (torch.nn.ModuleList): The network layers corresponding to the states.
#             init_strategy (str, optional): The initialization strategy for ADMM states. Defaults to "s-uniform".
#             cache_dir (Union[str, Path], optional): Directory to save disk caches. Defaults to "./admm_cache".
#             device (Optional[Union[str, torch.device]], optional): The execution device.
#         """
#         self.device = (
#             torch.device(device) if device is not None else torch.device("cpu")
#         )
#         self.layers = layers
#         self.init_strategy = init_strategy
#         self.cache_dir = os.path.join(cache_dir, f"run_{uuid.uuid4().hex}")
#         self.num_batches = 0
#         self._writer = None
#         self.in_memory = False
#         self.memory_cache = {}
#         self.async_writes = async_writes
#         self.max_queue_size = max_queue_size
#         self.num_workers = num_workers
#         self._initialized_batches: set[int] = set()
#         self._initializer = get_initializer(self.init_strategy)
#         print(f"ADMM CACHE = {self.cache_dir}")

#     def is_batch_initialized(self, batch_id: int) -> bool:
#         """Return whether a persistent state already exists for this batch."""
#         return batch_id in self._initialized_batches

#     def initialize(self, dataloader: DataLoader) -> None:
#         """Prepare the state handler without creating any batch states.

#         Batch states are initialized lazily when they are first requested.
#         """
#         if self._writer is not None:
#             self._writer.shutdown()
#             self._writer = None
#         # Keep the existing single-batch optimization.
#         self.in_memory = len(dataloader) == 1

#         # Disk-backed mode needs the cache directory.
#         if not self.in_memory:
#             os.makedirs(self.cache_dir, exist_ok=True)
#             self._writer = (
#                 AsyncWriter(
#                     max_queue_size=self.max_queue_size,
#                     num_workers=self.num_workers,
#                 )
#                 if self.async_writes
#                 else None
#             )

#         self.num_batches = 0
#         self._initialized_batches.clear()
#         self.memory_cache.clear()

#     def _initialize_batch(
#         self,
#         batch_id: int,
#         inputs: torch.Tensor,
#     ) -> ADMM_BatchState:
#         """Create the initial ADMM state for one batch."""

#         if inputs.device != self.device:
#             inputs = inputs.to(
#                 self.device,
#                 non_blocking=True,
#             )

#         layer_states = self._initializer.init_states(
#             self.layers,
#             inputs,
#             self.device,
#         )

#         return ADMM_BatchState(
#             batch_id=batch_id,
#             layer_states=layer_states,
#         )

#     def save_batch(
#         self,
#         batch_state: ADMM_BatchState,
#     ) -> None:
#         """Serializes a batch state either to RAM or the hard drive.

#         Args:
#             batch_state (ADMM_BatchState): The [current state][src.admm.dataclasses.ADMM_BatchState] of the ADMM variables.
#         """
#         batch_id = batch_state.batch_id

#         if self.in_memory:  # Single batch route
#             self.memory_cache[batch_id] = {
#                 "batch_state": batch_state,
#             }
#         else:
#             tensors_to_save = {}
#             filepath = os.path.join(self.cache_dir, f"batch_{batch_id}.safetensors")

#             for layer_idx, l_state in enumerate(batch_state.layer_states):
#                 tensors_to_save[f"layer_{layer_idx}_z"] = l_state.z
#                 if l_state.a is not None:
#                     tensors_to_save[f"layer_{layer_idx}_a"] = l_state.a
#                 if l_state.lambda_lagrange is not None:
#                     tensors_to_save[f"layer_{layer_idx}_lambda_lagrange"] = (
#                         l_state.lambda_lagrange
#                     )

#             # if self._writer is not None:
#             #     cpu_tensors = {
#             #         key: value.detach().to("cpu", copy=True)
#             #         for key, value in tensors_to_save.items()
#             #     }

#             #     self._writer.enqueue(
#             #         cpu_tensors,
#             #         filepath,
#             #     )
#             if self._writer is not None:
#                 if self.device.type == "cuda":
#                     torch.cuda.synchronize(self.device)

#                 t0 = time.perf_counter()

#                 cpu_tensors = {
#                     key: value.detach().to("cpu", copy=True)
#                     for key, value in tensors_to_save.items()
#                 }

#                 if self.device.type == "cuda":
#                     torch.cuda.synchronize(self.device)

#                 t1 = time.perf_counter()

#                 self._writer.enqueue(
#                     cpu_tensors,
#                     filepath,
#                 )

#                 t2 = time.perf_counter()

#                 print(
#                     f"[SAVE {batch_id}] "
#                     f"GPU->CPU={t1 - t0:.3f}s | "
#                     f"enqueue={t2 - t1:.3f}s"
#                 )
#             else:
#                 save_file(tensors_to_save, filepath)

#         self._initialized_batches.add(batch_id)

#         self.num_batches = max(
#             self.num_batches,
#             batch_id + 1,
#         )

#     def _load_batch_from_disk(self, batch_id: int) -> ADMM_BatchState:
#         """Loads a batch state from disk. Caller must not use this in in-memory mode."""
#         filepath = os.path.join(
#             self.cache_dir,
#             f"batch_{batch_id}.safetensors",
#         )

#         if self._writer is not None:
#             self._writer.flush(filepath)

#         tensors = load_file(
#             filepath,
#             device=str(self.device),
#         )

#         layer_states = []
#         for layer_idx in range(len(self.layers)):
#             z = tensors[f"layer_{layer_idx}_z"].clone()

#             a = tensors.get(f"layer_{layer_idx}_a", None)
#             if a is not None:
#                 a = a.clone()

#             lambda_lagrange = tensors.get(f"layer_{layer_idx}_lambda_lagrange", None)
#             if lambda_lagrange is not None:
#                 lambda_lagrange = lambda_lagrange.clone()

#             layer_states.append(
#                 ADMM_LayerState(z=z, a=a, lambda_lagrange=lambda_lagrange)
#             )

#         del tensors
#         return ADMM_BatchState(batch_id=batch_id, layer_states=layer_states)

#     def load_batch(self, batch_id: int, inputs: torch.Tensor) -> ADMM_BatchState:
#         if batch_id not in self._initialized_batches:
#             return self._initialize_batch(
#                 batch_id=batch_id,
#                 inputs=inputs,
#             )

#         if self.in_memory:
#             return self.memory_cache[batch_id]["batch_state"]
#         return self._load_batch_from_disk(batch_id)


# class AsyncWriter:
#     """Asynchronously writes tensors to disk using a bounded thread pool.

#     The class provides three guarantees:

#     1. Writes happen in background threads.
#     2. The number of outstanding writes is bounded, preventing unbounded RAM usage.
#     3. Two writes to the same filepath are never active simultaneously.

#     Exceptions raised by background writes are propagated through ``flush()``.
#     """

#     def __init__(
#         self,
#         max_queue_size: int = 1,
#         num_workers: int = 1,
#     ):
#         if max_queue_size < 1:
#             raise ValueError("max_queue_size must be >= 1.")

#         if num_workers < 1:
#             raise ValueError("num_workers must be >= 1.")

#         self._num_workers = num_workers

#         # Keep the same semantics as the previous implementation:
#         # ``max_queue_size`` items could wait in the queue, while
#         # ``num_workers`` items could already be running.
#         self._max_in_flight = max_queue_size + num_workers

#         # Bounds the number of submitted-but-not-finished writes.
#         # This is important because each queued task owns references
#         # to the CPU tensors that it is going to write.
#         self._slots = threading.BoundedSemaphore(self._max_in_flight)

#         self._executor = ThreadPoolExecutor(
#             max_workers=num_workers,
#             thread_name_prefix="admm-writer",
#         )

#         # At most one pending Future is allowed for a given filepath.
#         self._futures: Dict[str, Future] = {}

#         # Protects ``_futures`` and ``_shutdown``.
#         self._lock = threading.Lock()

#         self._shutdown = False

#     @staticmethod
#     def _normalize_path(filepath: str) -> str:
#         """Convert a path to a stable absolute representation."""
#         return os.path.abspath(filepath)

#     # @staticmethod
#     # def _write_file(
#     #     tensors_dict: dict,
#     #     filepath: str,
#     # ) -> None:
#     #     """Write a complete file atomically.

#     #     The data is first written to a temporary file in the same directory.
#     #     Once that succeeds, ``os.replace`` swaps it into place.
#     #     """

#     #     directory = os.path.dirname(filepath) or "."
#     #     os.makedirs(directory, exist_ok=True)

#     #     fd, tmp_path = tempfile.mkstemp(
#     #         prefix=os.path.basename(filepath) + ".tmp-",
#     #         dir=directory,
#     #     )
#     #     os.close(fd)

#     #     try:
#     #         save_file(tensors_dict, tmp_path)

#     #         # Atomic replacement on the same filesystem.
#     #         os.replace(tmp_path, filepath)

#     #     finally:
#     #         # Remove the temporary file if something failed before os.replace.
#     #         if os.path.exists(tmp_path):
#     #             try:
#     #                 os.remove(tmp_path)
#     #             except OSError:
#     #                 pass
#     @staticmethod
#     def _write_file(tensors_dict: dict, filepath: str) -> None:
#         t0 = time.perf_counter()

#         total_bytes = sum(
#             tensor.numel() * tensor.element_size() for tensor in tensors_dict.values()
#         )

#         total_gb = total_bytes / (1024**3)

#         directory = os.path.dirname(filepath) or "."
#         os.makedirs(directory, exist_ok=True)

#         fd, tmp_path = tempfile.mkstemp(
#             prefix=os.path.basename(filepath) + ".tmp-",
#             dir=directory,
#         )
#         os.close(fd)

#         try:
#             t1 = time.perf_counter()

#             save_file(tensors_dict, tmp_path)

#             t2 = time.perf_counter()

#             os.replace(tmp_path, filepath)

#             t3 = time.perf_counter()

#             throughput = total_gb / (t2 - t1)

#             print(
#                 f"[WRITER] {os.path.basename(filepath)} "
#                 f"size={total_gb:.3f} GB | "
#                 f"save={t2 - t1:.3f}s | "
#                 f"throughput={throughput:.1f} GB/s | "
#                 f"replace={t3 - t2:.3f}s"
#             )

#         finally:
#             if os.path.exists(tmp_path):
#                 try:
#                     os.remove(tmp_path)
#                 except OSError:
#                     pass

#     def enqueue(
#         self,
#         tensors_dict: dict,
#         filepath: str,
#     ) -> Future:
#         """Schedule a background write.

#         Blocks when the bounded in-flight window is full.

#         If another write to the same filepath is still pending, this method
#         waits for that write to finish before scheduling the new one.
#         """

#         filepath = self._normalize_path(filepath)

#         with self._lock:
#             if self._shutdown:
#                 raise RuntimeError("AsyncWriter has already been shut down.")

#             # Prevent overlapping writes to the same file.
#             previous = self._futures.get(filepath)

#             if previous is not None:
#                 try:
#                     previous.result()
#                 except Exception as exc:
#                     raise RuntimeError(f"Async write failed for '{filepath}'.") from exc
#                 finally:
#                     # Remove the old completed Future, whether it succeeded
#                     # or failed, so a failed write doesn't permanently block
#                     # every future write to this same file.
#                     if self._futures.get(filepath) is previous:
#                         del self._futures[filepath]

#             # Apply backpressure before submitting another task.
#             self._slots.acquire()

#             try:
#                 future = self._executor.submit(
#                     self._write_file,
#                     tensors_dict,
#                     filepath,
#                 )
#             except BaseException:
#                 # If submission failed, give the slot back.
#                 self._slots.release()
#                 raise

#             self._futures[filepath] = future

#             # Free one slot when the write finishes, regardless of success/failure.
#             future.add_done_callback(lambda _: self._slots.release())

#             return future

#     def flush(self, filepath: Optional[str] = None) -> None:
#         """Wait for pending writes.

#         Args:
#             filepath:
#                 If provided, wait only for that file.
#                 If None, wait for every pending write.

#         Any background exception is re-raised in the calling thread.
#         """

#         if filepath is not None:
#             filepath = self._normalize_path(filepath)

#             with self._lock:
#                 future = self._futures.get(filepath)

#             if future is None:
#                 return

#             try:
#                 future.result()
#             except Exception as exc:
#                 raise RuntimeError(f"Async write failed for '{filepath}'.") from exc
#             finally:
#                 with self._lock:
#                     if self._futures.get(filepath) is future:
#                         del self._futures[filepath]

#             return

#         # Flush all currently known files.
#         while True:
#             with self._lock:
#                 pending = list(self._futures.items())

#             if not pending:
#                 return

#             first_error = None

#             for filepath, future in pending:
#                 try:
#                     future.result()
#                 except Exception as exc:
#                     if first_error is None:
#                         first_error = (filepath, exc)
#                 finally:
#                     with self._lock:
#                         if self._futures.get(filepath) is future:
#                             del self._futures[filepath]

#             if first_error is not None:
#                 filepath, exc = first_error
#                 raise RuntimeError(f"Async write failed for '{filepath}'.") from exc

#     def shutdown(self) -> None:
#         """Flush all pending writes and shut down the executor."""

#         with self._lock:
#             if self._shutdown:
#                 return

#             self._shutdown = True

#         try:
#             self.flush()
#         finally:
#             self._executor.shutdown(wait=True)

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.shutdown()


# class AsyncStatePrefetcher:
#     """Asynchronously prefetches ADMM batch states from disk.

#     The manager uses this class with a one-batch lookahead:

#         current batch → being consumed
#         next batch    → being loaded

#     ``max_in_flight`` bounds the number of submitted loads.
#     """

#     def __init__(
#         self,
#         state_handler,
#         max_in_flight: int = 2,
#         num_workers: int = 1,
#     ):
#         if max_in_flight < 1:
#             raise ValueError("max_in_flight must be >= 1.")

#         if num_workers < 1:
#             raise ValueError("num_workers must be >= 1.")

#         self._handler = state_handler

#         self._executor = ThreadPoolExecutor(
#             max_workers=num_workers,
#             thread_name_prefix="admm-reader",
#         )

#         self._slots = threading.BoundedSemaphore(max_in_flight)

#         # Only the main training thread accesses this dictionary.
#         self._futures: Dict[int, Future] = {}

#         self._shutdown = False

#     def schedule(self, batch_id: int) -> None:
#         """Start loading a batch in the background."""

#         if self._shutdown:
#             raise RuntimeError("AsyncStatePrefetcher has already been shut down.")

#         if batch_id in self._futures:
#             raise RuntimeError(f"Batch {batch_id} is already scheduled.")

#         # Apply backpressure.
#         self._slots.acquire()

#         try:
#             future = self._executor.submit(
#                 self._handler._load_batch_from_disk,
#                 batch_id,
#             )
#         except BaseException:
#             self._slots.release()
#             raise

#         self._futures[batch_id] = future

#     def get(self, batch_id: int) -> ADMM_BatchState:
#         """Wait for a previously scheduled batch and return its state."""

#         future = self._futures.pop(batch_id, None)

#         if future is None:
#             raise RuntimeError(f"Batch {batch_id} was not scheduled.")

#         try:
#             return future.result()
#         finally:
#             # Whether loading succeeded or failed, one slot is now free.
#             self._slots.release()

#     def shutdown(self) -> None:
#         """Stop the reader threads."""

#         if self._shutdown:
#             return

#         self._shutdown = True

#         self._executor.shutdown(
#             wait=True,
#             cancel_futures=True,
#         )

"""
ADMM state persistence using raw binary (.bin) files.

Each run creates:

    ./admm_cache/run_<uuid>/
        index.json
        batch_0.bin
        batch_1.bin
        ...

The binary files contain raw tensor bytes. ``index.json`` stores the
metadata required to reconstruct every tensor:

    - tensor name
    - byte offset
    - number of bytes
    - shape
    - dtype

The writer is asynchronous and bounded. By default it uses:

    max_queue_size = 1
    num_workers = 1

This intentionally allows only one large state write to be active/pending,
which avoids excessive contention for CPU memory bandwidth and filesystem I/O.
"""

import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataclasses import ADMM_BatchState, ADMM_LayerState
from .initializers import get_initializer

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to a numpy dtype."""

    mapping = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
        torch.bfloat16: np.dtype(np.uint16),
        torch.float8_e4m3fn: np.dtype(np.uint8),
        torch.float8_e4m3fnuz: np.dtype(np.uint8),
        torch.float8_e5m2: np.dtype(np.uint8),
        torch.float8_e5m2fnuz: np.dtype(np.uint8),
        torch.int8: np.dtype(np.int8),
        torch.int16: np.dtype(np.int16),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
        torch.bool: np.dtype(np.bool_),
    }

    if dtype not in mapping:
        raise TypeError(f"Unsupported torch dtype: {dtype}")

    return mapping[dtype]


def _numpy_dtype_to_torch(dtype: str) -> torch.dtype:
    """Convert a numpy dtype string back to torch dtype."""

    dtype = np.dtype(dtype)

    mapping = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }

    # bfloat16 is stored as uint16 because numpy has no native bfloat16
    # representation in standard installations.
    if dtype == np.dtype(np.uint16):
        return torch.bfloat16

    if dtype not in mapping:
        raise TypeError(f"Unsupported numpy dtype: {dtype}")

    return mapping[dtype]


# ---------------------------------------------------------------------------
# ADMM State Handler
# ---------------------------------------------------------------------------


class ADMM_StateHandler:
    """Manages ADMM states between RAM and binary disk storage.

    For a single-batch DataLoader, states remain in RAM.

    For multiple batches, every batch is stored in a separate raw binary
    file:

        batch_<id>.bin

    A run-level index.json stores the metadata required to read those files.
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        init_strategy: str = "s-uniform",
        cache_dir: str = "./admm_cache",
        device: Optional[Union[str, torch.device]] = None,
        async_writes: bool = True,
        max_queue_size: int = 1,
        num_workers: int = 1,
        ram_budget_gb: float = 0,
    ):
        self._ram_budget = int(ram_budget_gb * 1024**3)
        print(f"RAM tier budget = {self._ram_budget / 1024**3:.1f} GB")
        self._ram_used = 0
        self._ram: Dict[int, Dict[str, torch.Tensor]] = {}
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        self.layers = layers
        self.init_strategy = init_strategy

        self.cache_dir = os.path.join(
            cache_dir,
            f"run_{uuid.uuid4().hex}",
        )

        self.index_path = os.path.join(
            self.cache_dir,
            "index.json",
        )

        self.num_batches = 0

        self._writer = None

        self.in_memory = False
        self.memory_cache = {}

        self.async_writes = async_writes
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers

        self._initialized_batches: set[int] = set()

        self._initializer = get_initializer(self.init_strategy)

        print(f"ADMM CACHE = {self.cache_dir}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def is_batch_initialized(self, batch_id: int) -> bool:
        """Return whether a batch state has already been initialized."""

        return batch_id in self._initialized_batches

    def initialize(self, dataloader: DataLoader) -> None:
        """Prepare the state handler."""

        if self._writer is not None:
            self._writer.shutdown()
            self._writer = None

        self.in_memory = len(dataloader) == 1

        if not self.in_memory:
            os.makedirs(
                self.cache_dir,
                exist_ok=True,
            )

            # Create the initial index.
            self._write_index(
                {
                    "version": 1,
                    "format": "raw_binary",
                    "batches": {},
                }
            )

            if self.async_writes:
                self._writer = AsyncWriter(
                    max_queue_size=self.max_queue_size,
                    num_workers=self.num_workers,
                )

        self.num_batches = 0
        self._initialized_batches.clear()
        self.memory_cache.clear()
        self._ram.clear()
        self._ram_used = 0

    # ------------------------------------------------------------------
    # Initialization of individual batches
    # ------------------------------------------------------------------
    def _tensors_to_batch_state(
        self, batch_id: int, tensors: Dict[str, torch.Tensor]
    ) -> ADMM_BatchState:
        """Build an ADMM_BatchState from a flat tensor dictionary."""
        layer_states = []
        for layer_idx in range(len(self.layers)):
            layer_states.append(
                ADMM_LayerState(
                    z=tensors[f"layer_{layer_idx}_z"],
                    a=tensors.get(f"layer_{layer_idx}_a"),
                    lambda_lagrange=tensors.get(f"layer_{layer_idx}_lambda_lagrange"),
                )
            )
        return ADMM_BatchState(batch_id=batch_id, layer_states=layer_states)

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

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _state_to_tensors(
        batch_state: ADMM_BatchState,
    ) -> Dict[str, torch.Tensor]:
        """Convert an ADMM batch state into a flat tensor dictionary."""

        tensors = {}

        for layer_idx, layer_state in enumerate(batch_state.layer_states):
            tensors[f"layer_{layer_idx}_z"] = layer_state.z

            if layer_state.a is not None:
                tensors[f"layer_{layer_idx}_a"] = layer_state.a

            if layer_state.lambda_lagrange is not None:
                tensors[f"layer_{layer_idx}_lambda_lagrange"] = (
                    layer_state.lambda_lagrange
                )

        return tensors

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _create_metadata(
        tensors: Dict[str, torch.Tensor],
    ) -> Dict:
        """Create the binary layout metadata for a tensor dictionary."""

        offset = 0
        metadata = {}

        for name, tensor in tensors.items():
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"Tensor '{name}' must be on CPU before creating binary metadata."
                )

            tensor = tensor.detach()

            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            dtype = _torch_dtype_to_numpy(tensor.dtype)

            nbytes = tensor.numel() * tensor.element_size()

            metadata[name] = {
                "offset": offset,
                "nbytes": nbytes,
                "shape": list(tensor.shape),
                "dtype": dtype.str,
            }

            offset += nbytes

        return {
            "tensors": metadata,
            "total_bytes": offset,
        }

    # ------------------------------------------------------------------
    # Index handling
    # ------------------------------------------------------------------

    def _read_index(self) -> Dict:
        """Read the run-level metadata index."""

        if not os.path.exists(self.index_path):
            return {
                "version": 1,
                "format": "raw_binary",
                "batches": {},
            }

        with open(
            self.index_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _try_save_ram(self, batch_id: int, tensors: Dict[str, torch.Tensor]) -> bool:
        """Copy into a persistent pinned buffer. Returns False if over budget."""
        buf = self._ram.get(batch_id)

        # Layout changed (e.g. lambda appears after warming): rebuild the slot.
        if buf is not None and (
            buf.keys() != tensors.keys()
            or any(
                buf[k].shape != t.shape or buf[k].dtype != t.dtype
                for k, t in tensors.items()
            )
        ):
            self._ram_used -= sum(v.numel() * v.element_size() for v in buf.values())
            del self._ram[batch_id]
            buf = None

        if buf is None:
            need = sum(t.numel() * t.element_size() for t in tensors.values())
            if self._ram_used + need > self._ram_budget:
                return False
            buf = {
                k: torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                for k, t in tensors.items()
            }
            self._ram[batch_id] = buf
            self._ram_used += need

        for k, t in tensors.items():
            buf[k].copy_(t.detach(), non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return True

    def _load_batch_from_ram(self, batch_id: int) -> ADMM_BatchState:
        buf = self._ram[batch_id]
        tensors = {k: v.to(self.device, non_blocking=True) for k, v in buf.items()}
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return self._tensors_to_batch_state(batch_id, tensors)

    def _write_index(self, index: Dict) -> None:
        """Atomically write the run-level index."""

        directory = os.path.dirname(self.index_path)

        fd, tmp_path = tempfile.mkstemp(
            prefix="index.tmp-",
            dir=directory,
            text=True,
        )

        try:
            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    index,
                    f,
                    indent=2,
                )

                f.flush()
                os.fsync(f.fileno())

            os.replace(
                tmp_path,
                self.index_path,
            )

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _register_batch_metadata(
        self,
        batch_id: int,
        metadata: Dict,
    ) -> None:
        """Register a batch in index.json."""

        index = self._read_index()

        index["batches"][str(batch_id)] = metadata

        self._write_index(index)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_batch(
        self,
        batch_state: ADMM_BatchState,
    ) -> None:
        """Save one ADMM batch state."""

        batch_id = batch_state.batch_id

        # --------------------------------------------------------------
        # Single-batch RAM mode
        # --------------------------------------------------------------

        if self.in_memory:
            self.memory_cache[batch_id] = {
                "batch_state": batch_state,
            }

        # --------------------------------------------------------------
        # Disk-backed mode
        # --------------------------------------------------------------

        else:
            tensors_to_save = self._state_to_tensors(batch_state)
            if self._try_save_ram(batch_id, tensors_to_save):
                self._initialized_batches.add(batch_id)
                self.num_batches = max(self.num_batches, batch_id + 1)
                return

            filepath = os.path.join(
                self.cache_dir,
                f"batch_{batch_id}.bin",
            )

            # ----------------------------------------------------------
            # GPU -> CPU
            # ----------------------------------------------------------

            if self._writer is not None:
                cpu_tensors = {
                    key: value.detach().to("cpu", copy=True).contiguous()
                    for key, value in tensors_to_save.items()
                }

                metadata = self._create_metadata(cpu_tensors)

                self._register_batch_metadata(
                    batch_id,
                    metadata,
                )

                self._writer.enqueue(
                    cpu_tensors,
                    filepath,
                )

            # ----------------------------------------------------------
            # Synchronous mode
            # ----------------------------------------------------------

            else:
                cpu_tensors = {
                    key: value.detach().to("cpu", copy=True).contiguous()
                    for key, value in tensors_to_save.items()
                }

                metadata = self._create_metadata(cpu_tensors)

                self._register_batch_metadata(
                    batch_id,
                    metadata,
                )

                AsyncWriter._write_file(
                    cpu_tensors,
                    filepath,
                )

        self._initialized_batches.add(batch_id)

        self.num_batches = max(
            self.num_batches,
            batch_id + 1,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tensor_from_bin(
        filepath: str,
        metadata: Dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Read one tensor from a binary file."""
        np_dtype = np.dtype(metadata["dtype"])

        offset = metadata["offset"]
        nbytes = metadata["nbytes"]
        shape = tuple(metadata["shape"])

        # --------------------------------------------------------------
        # Use mmap so the complete 1.7 GB file does not need to be
        # copied into Python memory before extracting the tensor.
        # --------------------------------------------------------------

        mm = np.memmap(
            filepath,
            dtype=np_dtype,
            mode="r",
            offset=offset,
            shape=(nbytes // np_dtype.itemsize,),
        )

        tensor = torch.from_numpy(mm)

        tensor = tensor.reshape(shape)

        # --------------------------------------------------------------
        # GPU:
        #
        # Copy directly from the mmap-backed CPU tensor to GPU.
        #
        # After .to(device), the GPU tensor owns its memory, so the
        # mmap can safely disappear.
        # --------------------------------------------------------------

        if device.type == "cuda":
            tensor = tensor.to(
                device,
                non_blocking=True,
            )

            del mm

            return tensor

        # --------------------------------------------------------------
        # CPU:
        #
        # Make an independent copy because returning a tensor backed by
        # a temporary mmap would leave its lifetime unclear.
        # --------------------------------------------------------------

        tensor = tensor.clone()

        del mm

        return tensor

    def _load_batch_from_disk(
        self,
        batch_id: int,
    ) -> ADMM_BatchState:
        """Load one ADMM state from its binary file."""
        if batch_id in self._ram:
            return self._load_batch_from_ram(batch_id)
        filepath = os.path.join(
            self.cache_dir,
            f"batch_{batch_id}.bin",
        )

        # --------------------------------------------------------------
        # Make absolutely sure a pending write has completed.
        # --------------------------------------------------------------

        if self._writer is not None:
            self._writer.flush(filepath)

        # --------------------------------------------------------------
        # Read metadata.
        # --------------------------------------------------------------

        index = self._read_index()

        batch_metadata = index["batches"].get(str(batch_id))

        if batch_metadata is None:
            raise RuntimeError(f"No metadata found for batch {batch_id}.")

        # --------------------------------------------------------------
        # Reconstruct layer states.
        # --------------------------------------------------------------

        tensors = {}

        for name, metadata in batch_metadata["tensors"].items():
            tensors[name] = self._load_tensor_from_bin(
                filepath,
                metadata,
                self.device,
            )

        # --------------------------------------------------------------
        # Reconstruct ADMM_BatchState.
        # --------------------------------------------------------------

        layer_states = []

        for layer_idx in range(len(self.layers)):
            z = tensors[f"layer_{layer_idx}_z"]

            a = tensors.get(
                f"layer_{layer_idx}_a",
                None,
            )

            lambda_lagrange = tensors.get(
                f"layer_{layer_idx}_lambda_lagrange",
                None,
            )

            layer_states.append(
                ADMM_LayerState(
                    z=z,
                    a=a,
                    lambda_lagrange=lambda_lagrange,
                )
            )

        return ADMM_BatchState(
            batch_id=batch_id,
            layer_states=layer_states,
        )

    # ------------------------------------------------------------------
    # Public loading API
    # ------------------------------------------------------------------

    def load_batch(
        self,
        batch_id: int,
        inputs: torch.Tensor,
    ) -> ADMM_BatchState:

        if batch_id not in self._initialized_batches:
            return self._initialize_batch(
                batch_id=batch_id,
                inputs=inputs,
            )

        if self.in_memory:
            return self.memory_cache[batch_id]["batch_state"]

        return self._load_batch_from_disk(batch_id)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush and shutdown the asynchronous writer."""

        if self._writer is not None:
            self._writer.shutdown()
            self._writer = None


# =========================================================================
# Async Writer
# =========================================================================


class AsyncWriter:
    """Asynchronously writes raw binary tensor files.

    The default configuration is intentionally conservative:

        max_queue_size = 1
        num_workers = 1

    This means there is at most one active/pending large write.

    The writer performs:

        CPU tensors
            ↓
        raw binary write
            ↓
        atomic os.replace()

    so readers never observe a partially written .bin file.
    """

    def __init__(
        self,
        max_queue_size: int = 1,
        num_workers: int = 1,
    ):

        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1.")

        if num_workers < 1:
            raise ValueError("num_workers must be >= 1.")

        self._num_workers = num_workers

        self._max_in_flight = max_queue_size + num_workers

        self._slots = threading.BoundedSemaphore(self._max_in_flight)

        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="admm-writer",
        )

        self._futures: Dict[
            str,
            Future,
        ] = {}

        self._lock = threading.Lock()

        self._shutdown = False

    @staticmethod
    def _normalize_path(
        filepath: str,
    ) -> str:
        return os.path.abspath(filepath)

    # ------------------------------------------------------------------
    # Binary writer
    # ------------------------------------------------------------------

    @staticmethod
    def _write_file(
        tensors_dict: dict,
        filepath: str,
    ) -> None:

        directory = os.path.dirname(filepath) or "."

        os.makedirs(
            directory,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Temporary file in the SAME directory.
        #
        # This is important because os.replace() is then atomic.
        # --------------------------------------------------------------

        fd, tmp_path = tempfile.mkstemp(
            prefix=(os.path.basename(filepath) + ".tmp-"),
            dir=directory,
        )

        os.close(fd)

        try:
            with open(
                tmp_path,
                "wb",
                buffering=1024 * 1024,
            ) as f:
                for tensor in tensors_dict.values():
                    tensor = tensor.detach()

                    if tensor.device.type != "cpu":
                        raise ValueError("AsyncWriter received a non-CPU tensor.")

                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()

                    # --------------------------------------------------
                    # Write raw bytes.
                    # --------------------------------------------------

                    array = tensor.numpy()

                    f.write(memoryview(array))

                f.flush()

                # Make sure the data has reached the filesystem
                # before exposing the file.
                os.fsync(f.fileno())

            os.replace(
                tmp_path,
                filepath,
            )

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(
        self,
        tensors_dict: dict,
        filepath: str,
    ) -> Future:

        filepath = self._normalize_path(filepath)

        with self._lock:
            if self._shutdown:
                raise RuntimeError("AsyncWriter has already been shut down.")

            # ----------------------------------------------------------
            # Prevent two writes to the same file.
            # ----------------------------------------------------------

            previous = self._futures.get(filepath)

            if previous is not None:
                try:
                    previous.result()

                except Exception as exc:
                    raise RuntimeError(f"Async write failed for '{filepath}'.") from exc

                finally:
                    if self._futures.get(filepath) is previous:
                        del self._futures[filepath]

            # ----------------------------------------------------------
            # Backpressure.
            # ----------------------------------------------------------

            self._slots.acquire()

            try:
                future = self._executor.submit(
                    self._write_file,
                    tensors_dict,
                    filepath,
                )

            except BaseException:
                self._slots.release()

                raise

            self._futures[filepath] = future

            future.add_done_callback(lambda _: self._slots.release())

            return future

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(
        self,
        filepath: Optional[str] = None,
    ) -> None:

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

        # --------------------------------------------------------------
        # Flush every pending file.
        # --------------------------------------------------------------

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
                        first_error = (
                            filepath,
                            exc,
                        )

                finally:
                    with self._lock:
                        if self._futures.get(filepath) is future:
                            del self._futures[filepath]

            if first_error is not None:
                filepath, exc = first_error

                raise RuntimeError(f"Async write failed for '{filepath}'.") from exc

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:

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

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.shutdown()


# =========================================================================
# Async State Prefetcher
# =========================================================================


class AsyncStatePrefetcher:
    """Asynchronously prefetches ADMM states from .bin files.

    Typical usage:

        prefetcher.schedule(next_batch)

        state = prefetcher.get(current_batch)

    This allows the next batch to be loaded while the current batch
    is being processed.
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

        self._futures: Dict[
            int,
            Future,
        ] = {}

        self._shutdown = False

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def schedule(self, batch_id: int) -> None:

        print(
            f"[PREFETCH SCHEDULE] batch={batch_id} "
            f"existing={list(self._futures.keys())}"
        )

        if self._shutdown:
            raise RuntimeError("AsyncStatePrefetcher has already been shut down.")

        if batch_id in self._futures:
            raise RuntimeError(f"Batch {batch_id} is already scheduled.")

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

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(self, batch_id: int) -> ADMM_BatchState:

        print(f"[PREFETCH GET] batch={batch_id} existing={list(self._futures.keys())}")

        future = self._futures.pop(batch_id, None)

        if future is None:
            raise RuntimeError(f"Batch {batch_id} was not scheduled.")

        try:
            return future.result()
        finally:
            self._slots.release()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:

        if self._shutdown:
            return

        self._shutdown = True

        self._executor.shutdown(
            wait=True,
            cancel_futures=True,
        )
