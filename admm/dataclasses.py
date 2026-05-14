"""
This module defines the universal configuration and state containers used
across the ADMM framework. By using strict data classes, we ensure type safety,
fail-fast validation, and a clean separation between network hyperparameters
and dynamic optimization states.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class ADMM_Config:
    r"""Universal global configuration class for the ADMM Manager.

    Defines the global behavior of the ADMM optimization loop, dictating how
    layers are sequenced, how auxiliary states are initialized, and which
    mathematical solver methods are deployed.

    Attributes:
        init (str): The strategy used to initialize the auxiliary states ($z$ and $a$)
            before the first optimization step. Options include `"zeros"`, `"xavier"`,
            `"pytorch"`, `"z-uniform"`, `"s-uniform"`, and `"relaxed"`. Defaults to `"s-uniform"`.
            For specific information on each strategy, refer to the [initializers][admm.initializers].
        train_method (str): The specific algorithmic sequence used to update the network.
            For static networks, this must be `"vectorized"`, `"unrolled-random"`, `"unrolled-sequential"`,
            `"decoupled-random"`, `"decoupled-sequential"`, `"unrolled-backwards"` and `"decoupled-backwards"`.
            Defaults to `"decoupled-backwards"`.
        layer_order (str): The sequence in which the layers are optimized during a
            single ADMM sweep. Options include `"backwards"`, `"sequential"`, `"random"`,
            and `"random-last"`. Defaults to `"backwards"`.
        update_z_first (bool): A boolean flag determining the sub-step order. If True,
            the pre-activations ($z$) are updated before the activations ($a$).
            Defaults to `False`.
    """

    init: str = "s-uniform"
    train_method: str = "decoupled-backwards"
    layer_order: str = "backwards"
    update_z_first: bool = False

    def __post_init__(self):
        # Validate initialization strategies
        valid_inits = {
            "zeros",
            "xavier",
            "wrandom",
            "pytorch",
            "z-uniform",
            "s-uniform",
            "relaxed",
        }
        if self.init not in valid_inits:
            raise ValueError(f"Invalid init '{self.init}'. Allowed: {valid_inits}")

        # Validate training methods
        valid_methods = {
            "vectorized",
            "unrolled-random",
            "unrolled-sequential",
            "decoupled-random",
            "decoupled-sequential",
            "unrolled-backwards",
            "decoupled-backwards",
        }
        if self.train_method not in valid_methods:
            raise ValueError(
                f"Invalid train_method '{self.train_method}'. Allowed: {valid_methods}"
            )

        # Validate layer orders
        valid_orders = {"backwards", "random-last", "random", "sequential"}
        if self.layer_order not in valid_orders:
            raise ValueError(
                f"Invalid layer_order '{self.layer_order}'. Allowed: {valid_orders}"
            )

        # Enforce boolean types
        if not isinstance(self.update_z_first, bool):
            raise TypeError(
                f"update_z_first must be a boolean, got {type(self.update_z_first)}"
            )


@dataclass
class ADMM_LayerConfig:
    r"""Universal configuration container for layer-specific ADMM hyperparameters.

    Each mathematical layer in the network maintains its own instance of this
    configuration, allowing for localized tuning of ADMM penalties and physics parameters.

    Attributes:
        rho (float): The penalty parameter enforcing the affine consensus constraint
            ($z = Wx + b$). Higher values force stricter compliance to the spatial weights.
            Defaults to `1.0`.
        beta (float): The penalty parameter enforcing the non-linear activation constraint
            ($a = h(z)$). Higher values force stricter compliance to the activation function.
            Defaults to `1.0`.
        thetas (float, optional): The spiking threshold parameter used strictly for
            Spiking Neural Networks (SNNs). Dictates the membrane potential required
            to emit a spike. Defaults to `None`.
        deltas (float, optional): The temporal leakage parameter for SNNs. Represents
            how much membrane potential is retained between time steps. Defaults to `None`.
        use_lagrange (bool): Flag to enable or disable the dual variable ($\lambda$) updates.
            If disabled, the algorithm operates as an unconstrained penalty method. Defaults to `False`.
        use_bias (bool): Flag determining if the specific layer should maintain and
            optimize a trainable bias vector. Defaults to `False`.
        use_reset (bool): Flag determining if the layer should apply a voltage reset
            penalty after a spike is emitted (soft-reset mechanics). Defaults to `True`. Only applicable for SNNs.
        use_cholesky (bool): Flag indicating if the spatial weights solver should use
            a Cholesky decomposition (faster) instead of a standard Pseudo-Inverse. Defaults to `True`.
    """

    rho: float = 1.0
    beta: float = 1.0
    thetas: Optional[float] = None
    deltas: Optional[float] = None
    use_lagrange: bool = False
    use_bias: bool = False
    use_reset: bool = True
    use_cholesky: bool = True
    use_fft: bool = True  # Only applicable for convolutional layers

    def __post_init__(self):
        # Validate math penalties (must be strictly positive)
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {self.rho}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")

        # Validate optional temporal physics parameters
        if self.thetas is not None and not isinstance(self.thetas, (int, float)):
            raise TypeError(f"thetas must be a numeric value, got {type(self.thetas)}")
        if self.deltas is not None and not isinstance(self.deltas, (int, float)):
            raise TypeError(f"deltas must be a numeric value, got {type(self.deltas)}")

        # Strictly enforce booleans for flags to prevent silent type-casting bugs
        for flag in ["use_lagrange", "use_bias", "use_reset", "use_cholesky"]:
            val = getattr(self, flag)
            if not isinstance(val, bool):
                raise TypeError(f"{flag} must be a boolean, got {type(val)}")


@dataclass
class ADMM_LayerState:
    r"""Holds the persistent ADMM memory for a single layer for a specific batch.

    This replaces the stateful `self.z`, `self.a`, and `self.lambda_lagrange`
    attributes that previously lived inside the nn.Module.

    Attributes:
        z (torch.Tensor): The pre-activation / membrane potential tensor.
        a (torch.Tensor): The activation / spike tensor.
        lambda_lagrange (torch.Tensor, optional): The dual variable (Lagrange multiplier) tensor.
    """

    z: torch.Tensor
    a: Optional[torch.Tensor] = None
    lambda_lagrange: Optional[torch.Tensor] = None


@dataclass
class ADMM_BatchState:
    r"""Holds the complete network state for a single batch of data.

    Attributes:
        batch_id (int): The unique identifier for this batch on the hard drive.
        layer_states (list[ADMM_LayerState]): The memory states for each layer,
            ordered from input (layer 0) to output (layer L-1).
    """

    batch_id: int
    layer_states: list[ADMM_LayerState]


@dataclass
class ADMM_LayerCovariance:
    r"""Holds the global accumulated matrices for a layer's parameter updates.

    Attributes:
        numerator (Union[float, torch.Tensor]): The accumulated $Y^T P$ numerator matrix.
        denominator (Union[float, torch.Tensor]): The accumulated $P^T P$ denominator matrix.
        bias_sum (Union[float, torch.Tensor]): The accumulated residual sum for bias updates.
        bias_count (int): The number of elements accumulated in the bias sum.
        pinv (torch.Tensor, optional): The cached pseudo-inverse matrix to speed up solves.
    """

    numerator: Union[float, torch.Tensor] = 0.0
    denominator: Union[float, torch.Tensor] = 0.0
    bias_sum: Optional[Union[float, torch.Tensor]] = None
    bias_count: Optional[int] = None
    pinv: Optional[torch.Tensor] = None


@dataclass
class TemporalCache:
    r"""Holds precomputed tensors to accelerate unrolled SNN temporal loops.

    Provides type safety and IDE auto-completion while preventing redundant
    matrix calculations during time-step iterations.

    Attributes:
        forward_pass (torch.Tensor): The precomputed spatial forward pass for the entire sequence.
        denominator_main (torch.Tensor): The left-hand side (LHS) system matrix for time steps $t < T$.
        denominator_last (torch.Tensor): The left-hand side (LHS) system matrix for time step $t = T$.
        adjoint (torch.Tensor): The precomputed adjoint operator mapping the next layer's errors backwards.
    """

    forward_pass: torch.Tensor
    denominator_main: torch.Tensor
    denominator_last: torch.Tensor
    adjoint: torch.Tensor

    @classmethod
    def build(
        cls,
        layer: torch.nn.Module,
        next_layer: torch.nn.Module,
        next_state: ADMM_LayerState,
        a_prev: torch.Tensor,
        state: ADMM_LayerState,
    ) -> "TemporalCache":
        """Precomputes and distributes operations to accelerate the unrolled loop.

        This factory method calculates the static portions of the ADMM activation
        update that do not change during the unrolled sequence.

        Args:
            layer (nn.Module): The current layer.
            next_layer (nn.Module): The subsequent layer in the network.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            TemporalCache: A typed data class containing the precomputed matrices.
        """
        # 1- Forward Pass
        forward_pass = layer.spatial_forward(a_prev)

        # 2- Denominator
        denominator_main, denominator_last, _ = next_layer.get_a_denominator(
            beta_current=layer.config.beta,
            rho_current=layer.config.rho,
            thetas_current=layer.config.thetas,
            a_shape=state.a.shape,
            unrolled=True,
        )

        # 3- Term 2
        adjoint = layer._get_a_adjoint(
            next_layer=next_layer, next_state=next_state, a_shape=state.a.shape
        )  # forward_pass=forward_pass)  DEPRECATED

        return cls(
            forward_pass=forward_pass,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            adjoint=adjoint,
        )
