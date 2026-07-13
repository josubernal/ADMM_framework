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
            `"wrandom"`, `"pytorch"`, `"z-uniform"`, `"s-uniform"`, and `"relaxed"`.
            Defaults to `"s-uniform"`. For specific information on each strategy,
            refer to the [initializers][src.admm.initializers].
        block_method (str): The block splitting strategy for the ADMM framework.
            Options include `"two-block"` and `"multi-block"`. Defaults to `"two-block"`.
        train_method (str): The specific algorithmic sequence used to update the network.
            Options include `"vectorized"`, `"unrolled"`, and `"decoupled"`.
            Defaults to `"decoupled"`.
        layer_order (str): The sequence in which the layers are optimized during a
            single ADMM sweep. Options include `"backwards"`, `"sequential"`, `"random"`,
            and `"random-last"`. Defaults to `"backwards"`.
        time_order (str): The sequence in which time steps are optimized during a
            single ADMM sweep for unrolled/spiking networks. Options include `"backwards"`,
            `"random"`, `"random-last"`, and `"sequential"`. Defaults to `"backwards"`.
        update_z_first (bool): A boolean flag determining the sub-step order. If True,
            the pre-activations ($z$) are updated before the activations ($a$).
            Defaults to `False`.
        solver (str): The solver used to compute the weight update. Options are `"standard"`,
            `"conjugate-gradient"`, and `"cholesky"`. Defaults to `"standard"`. For specific
            information on each solver, refer to the [weights solvers][src.admm.functional.weights_solvers].
        cache_pinv (bool): Serves to activate and deactivate the inverse caching of the first layer. Defaults to `True`.
    """

    init: str = "s-uniform"
    block_method: str = "two-block"
    train_method: str = "decoupled"
    layer_order: str = "backwards"
    time_order: str = "backwards"
    update_z_first: bool = False
    solver: str = "standard"
    cache_pinv: bool = True

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
            "unrolled",
            "decoupled",
        }
        if self.train_method not in valid_methods:
            raise ValueError(
                f"Invalid train_method '{self.train_method}'. Allowed: {valid_methods}"
            )
        # Validate block methods
        valid_block_methods = {"multi-block", "two-block", "distributed"}
        if self.block_method not in valid_block_methods:
            raise ValueError(
                f"Invalid block_method '{self.block_method}'. Allowed: {valid_block_methods}"
            )

        # Validate layer orders
        valid_orders = {"backwards", "random-last", "random", "sequential"}
        if self.layer_order not in valid_orders:
            raise ValueError(
                f"Invalid layer_order '{self.layer_order}'. Allowed: {valid_orders}"
            )
        # Validate time orders
        valid_orders = {"backwards", "random-last", "random", "sequential"}
        if self.time_order not in valid_orders:
            raise ValueError(
                f"Invalid time_order '{self.time_order}'. Allowed: {valid_orders}"
            )

        # Validate solver
        valid_solvers = {"standard", "conjugate-gradient", "cholesky"}
        if self.solver not in valid_solvers:
            raise ValueError(
                f"Invalid solver '{self.solver}'. Allowed: {valid_solvers}"
            )

        # Enforce boolean types
        if not isinstance(self.update_z_first, bool):
            raise TypeError(
                f"update_z_first must be a boolean, got {type(self.update_z_first)}"
            )

        if not isinstance(self.cache_pinv, bool):
            raise TypeError(
                f"cache_pinv must be a boolean, got {type(self.cache_pinv)}"
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
    """

    rho: float = 1.0
    beta: float = 1.0
    thetas: Optional[float] = None
    deltas: Optional[float] = None
    use_lagrange: bool = False
    use_bias: bool = False
    use_reset: bool = True  # This should be updated in the future to be authomatic. Maybe creating some kind of computation graph
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
        for flag in ["use_lagrange", "use_bias", "use_reset"]:
            val = getattr(self, flag)
            if not isinstance(val, bool):
                raise TypeError(f"{flag} must be a boolean, got {type(val)}")


@dataclass
class ADMM_LayerState:
    r"""Holds the persistent ADMM memory for a single layer for a specific batch.

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
    r"""Holds the global accumulated matrices for a layer's parameter (weights and bias) updates.

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
        W_expanded (torch.Tensor): The expanded weight matrix of the next layer.
        denominator_main (torch.Tensor): The left-hand side (LHS) system matrix for time steps $t < T$.
        denominator_last (torch.Tensor): The left-hand side (LHS) system matrix for time step $t = T$.
        adjoint (torch.Tensor): The precomputed adjoint operator mapping the next layer's errors backwards.
        route (str): Solving strategy for the activation parameter.
    """

    forward_pass: torch.Tensor
    W_expanded: torch.Tensor
    adjoint: torch.Tensor
    route: str
    denominator_main: Optional[torch.Tensor] = None
    denominator_last: Optional[torch.Tensor] = None

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
            state (ADMM_LayerState): The current layer state.
            next_state (ADMM_LayerState): The subsequent layer state.

        Returns:
            TemporalCache: A typed data class containing the precomputed matrices.
        """
        # Delay the import to prevent circular dependencies
        from .functional.utils import should_use_woodbury

        # 1- Forward Pass
        forward_pass = layer.spatial_forward(a_prev)

        # Expand weights for the next layer
        W_expanded = next_layer.pool_op.expand_weights(next_layer.W, state.a.shape)

        # 3- Denominator
        denominator_main, denominator_last = None, None
        if (
            layer.config.use_fft
            and getattr(layer, "convolution", False)
            and getattr(next_layer, "convolution", False)
        ):
            route = "fft"
            den_main, den_last = next_layer._get_fft_a_denominator(
                config_prev=layer.config, a_shape=state.a.shape
            )
            # We want to cached it inverted
            denominator_main = 1.0 / den_main
            denominator_last = 1.0 / den_last

        elif should_use_woodbury(W=next_layer.W, W_expanded=W_expanded):
            route = "woodbury"
            # Main timesteps (t < T) absorb the SNN temporal reset penalty directly into β.
            temporal_penalty = layer.config.rho * (layer.config.thetas**2)
            beta_main = layer.config.beta + temporal_penalty
            beta_last = layer.config.beta
            # ρ_{l+1} * W_{l+1} * W_{l+1}^T
            WtW = torch.matmul(W_expanded, W_expanded.t()).mul_(next_layer.config.rho)

            # (β* I_M + ρ_{l+1} * W_{l+1} * W_{l+1}^T)^{-1}
            mat_main = WtW.clone()
            mat_main.diagonal().add_(beta_main)
            denominator_main = torch.linalg.inv(mat_main)

            mat_last = WtW.clone()
            mat_last.diagonal().add_(beta_last)
            denominator_last = torch.linalg.inv(mat_last)
        else:
            route = "standard"
            denominator_main, denominator_last, _ = next_layer._get_a_denominator(
                config_prev=layer.config,
                W_expanded=W_expanded,
            )
            # We want to cached the inverted matrix
            denominator_main = torch.linalg.inv(denominator_main).transpose(-2, -1)
            denominator_last = torch.linalg.inv(denominator_last).transpose(-2, -1)

        # 3- Adjoint
        inside_adjoint = next_layer.get_v(next_state)
        adjoint = next_layer.adjoint_operator(
            inside_adjoint, original_input_shape=state.a.shape
        )

        return cls(
            forward_pass=forward_pass,
            W_expanded=W_expanded,
            route=route,
            denominator_main=denominator_main,
            denominator_last=denominator_last,
            adjoint=adjoint,
        )
