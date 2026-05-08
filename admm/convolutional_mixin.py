"""
This mixin provides Fast Fourier Transform (FFT) accelerated operations for
convolutional layers. By leveraging the Convolution Theorem, it performs heavy
matrix inversions and spatial adjoint operations in the frequency domain,
drastically improving numerical performance.
"""

import warnings

import torch
import torch.fft
import torch.nn.functional as F


class ADMM_Convolution:
    """A unified Mixin to handle FFTs and performance enhancing functions designed for convolution.

    Attributes:
        use_fft (bool): Flag determining whether spatial operations should be
            converted to the frequency domain. Defaults to True.
        padding_mode (str): The padding strategy. 'circular' is highly recommended
            to ensure exact equivalence between spatial operations and FFTs.
            Defaults to 'circular'.
    """

    def __init__(self, *args, use_fft=True, padding_mode="circular", **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fft = use_fft
        self.padding_mode = padding_mode

        if self.padding_mode == "circular" and not self.use_fft:
            warnings.warn(
                "Using circular padding without FFTs (use_fft=False) is highly inefficient. "
                "The spatial solver must construct a massive dense Gram matrix to compute the adjoint. "
                "Consider setting use_fft=True."
            )

    def _setup(self, global_config=None) -> None:
        """Initializes configuration and validates convolution hyperparameters."""
        if (
            getattr(self, "padding_mode", "zeros") == "circular"
            and getattr(self, "s", 1) != 1
        ):
            warnings.warn(
                f"Circular padding requires stride=1 for the ADMM solver. "
                f"Changing stride from {self.s} to 1."
            )
            self.s = 1
        super()._setup(global_config)

    def _circular_forward(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        """Applies a circular padded convolution.

        Args:
            x (torch.Tensor): The input spatial sequence or image tensor.
            use_bias (bool, optional): Whether to apply the layer's bias. Defaults to True.

        Returns:
            torch.Tensor: The convoluted output tensor.
        """
        b = (
            getattr(self, "b", None)
            if (
                isinstance(use_bias, bool) and use_bias and getattr(self, "bias", False)
            )
            else None
        )
        is_spiking = x.dim() == 5
        x_flat, tb_shape = self._fold_time(x) if is_spiking else (x, None)

        x_padded = F.pad(x_flat, (self.p, self.p, self.p, self.p), mode="circular")
        out_flat = F.conv2d(x_padded, self.W, bias=b, padding=0, stride=self.s)

        return self._unfold_time(out_flat, tb_shape) if is_spiking else out_flat

    def _circular_adjoint(self, target: torch.Tensor) -> torch.Tensor:
        """Computes the exact mathematical transpose (adjoint) using FFTs.

        Args:
            target (torch.Tensor): The tensor to be mapped backwards.

        Returns:
            torch.Tensor: The adjoint spatial tensor.
        """
        is_spiking = target.dim() == 5
        target_flat, tb_shape = (
            self._fold_time(target) if is_spiking else (target, None)
        )

        H, W_dim = target_flat.shape[-2:]
        pad_h = H - self.k
        pad_w = W_dim - self.k
        padded_W = F.pad(self.W, (0, pad_w, 0, pad_h))
        padded_W = torch.roll(
            padded_W, shifts=(-(self.k // 2), -(self.k // 2)), dims=(-2, -1)
        )

        W_fft = torch.fft.fft2(padded_W)
        target_fft = torch.fft.fft2(target_flat)

        out_fft = torch.einsum("bohw,oihw->bihw", target_fft, W_fft)
        out_flat = torch.fft.ifft2(out_fft).real

        return self._unfold_time(out_flat, tb_shape) if is_spiking else out_flat

    def _circular_compute_P(self, a_prev: torch.Tensor) -> torch.Tensor:
        """Extracts circular padded image patches for the weight update step."""
        is_spiking = a_prev.dim() == 5
        a_flat = self._fold_time(a_prev)[0] if is_spiking else a_prev

        a_padded = F.pad(a_flat, (self.p, self.p, self.p, self.p), mode="circular")
        patches = F.unfold(a_padded, kernel_size=self.k, padding=0, stride=self.s)

        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)

    def _get_WtW(self, a_shape: tuple) -> tuple[torch.Tensor, int]:
        r"""Computes the scaled weight covariance matrix $W^H W$.

        If `use_fft=True`, it transforms the weights to the frequency domain
        for massively accelerated, element-wise matrix decomposition.

        Args:
            a_shape (tuple): The physical geometry of the activations.

        Returns:
            tuple:
                - torch.Tensor: The computed covariance matrix (either dense or FFT based).
                - int: The number of tracked input features.
        """
        if not getattr(self, "use_fft", True):
            C, H, W_dim = a_shape[-3:]
            CHW = C * H * W_dim
            I = torch.eye(CHW, device=self.W.device, dtype=self.W.dtype)
            I_images = I.view(CHW, C, H, W_dim)

            A_I = self.spatial_forward(I_images, use_bias=False)
            WtW_images = self.adjoint_operator(A_I, original_input_shape=I_images.shape)

            WtW_dense = WtW_images.view(CHW, CHW)
            return WtW_dense, CHW

        height, width = a_shape[-2:]
        _, in_channels, kernel_h, kernel_w = self.W.shape

        pad_h = height - kernel_h
        pad_w = width - kernel_w
        padded_W = F.pad(self.W, (0, pad_w, 0, pad_h))

        shift_h = -(kernel_h // 2)
        shift_w = -(kernel_w // 2)
        padded_W = torch.roll(padded_W, shifts=(shift_h, shift_w), dims=(-2, -1))

        W_fft = torch.fft.fft2(padded_W)
        W_fft = W_fft.permute(2, 3, 0, 1)
        WtW_fft = torch.matmul(W_fft.transpose(-2, -1), W_fft.conj())

        return WtW_fft, in_channels

    def _solve_activation_system(
        self,
        numerator: torch.Tensor,
        denominator_main: torch.Tensor,
        denominator_last: torch.Tensor,
        a_shape: tuple,
        in_features: int,
    ) -> torch.Tensor:
        r"""Universal Solver for the activation ($a$) update in the frequency domain.

        Solves the linear system: $A x = B$ in Fourier space, bypassing $O(N^3)$
        inversion bottlenecks.

        Args:
            numerator (torch.Tensor): The precomputed numerator tensor.
            denominator_main (torch.Tensor): The FFT denominator matrix for $t < T$.
            denominator_last (torch.Tensor): The FFT denominator matrix for $t = T$.
            a_shape (tuple): The shape of the activation tensor.
            in_features (int): Number of input channels.

        Returns:
            torch.Tensor: The exact updated activations mapped back to the spatial domain.
        """
        if not self.use_fft:
            return super()._solve_activation_system(
                numerator, denominator_main, denominator_last, a_shape, in_features
            )

        original_dim = numerator.dim()
        if original_dim == 4:
            numerator = numerator.unsqueeze(0)
        _, batch_size = numerator.shape[:2]
        numerator_fft = torch.fft.fft2(numerator).permute(0, 1, 3, 4, 2).unsqueeze(-1)

        A_main = denominator_main.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        A_last = denominator_last.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        a_fft_main = torch.linalg.solve(A_main, numerator_fft[:-1])
        a_fft_last = torch.linalg.solve(A_last, numerator_fft[-1:])

        a_fft = torch.cat([a_fft_main, a_fft_last], dim=0)
        a_fft = a_fft.squeeze(-1).permute(0, 1, 4, 2, 3)
        a_spatial = torch.fft.ifft2(a_fft).real

        return a_spatial.squeeze(0) if original_dim == 4 else a_spatial

    def _solve_activation_system_unrolled(
        self, numerator: torch.Tensor, denominator: torch.Tensor
    ) -> torch.Tensor:
        r"""FFT-based unrolled step solver for convolutions.

        Executes a batched frequency-domain matrix multiplication for a single timestep.

        Args:
            numerator (torch.Tensor): The numerator vector for the current timestep.
            denominator (torch.Tensor): The precomputed inverted denominator matrix.

        Returns:
            torch.Tensor: The updated activation tensor for the current timestep in the spatial domain.
        """
        if not self.use_fft:
            return super()._solve_activation_system_unrolled(numerator, denominator)

        numerator_fft = torch.fft.fft2(numerator)
        numerator_fft = numerator_fft.permute(2, 3, 0, 1).unsqueeze(
            -1
        )  # [Batch, C, H, W] -> [H, W, Batch, C, 1]
        denominator_expanded = denominator.unsqueeze(
            2
        )  # [H, W, C, C] -> [H, W, 1, C, C]
        a_t_fft = torch.matmul(denominator_expanded, numerator_fft)
        a_t_fft = a_t_fft.squeeze(-1).permute(
            2, 3, 0, 1
        )  # [H, W, Batch, C, 1] -> [Batch, C, H, W]
        return torch.fft.ifft2(a_t_fft).real

    def _compute_covariances(
        self, Y: torch.Tensor, a_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the numerator and denominator for the weight update.

        Overrides the base method to compute covariances in chunks, preventing
        Out Of Memory (OOM) errors during heavy 5D spatiotemporal operations.

        Args:
            Y (torch.Tensor): The target spatial tensor.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            tuple:
                - torch.Tensor: The computed numerator matrix ($Y^T P$).
                - torch.Tensor: The computed denominator matrix ($P^T P$).
        """
        if a_prev.dim() == 4:
            a_prev = a_prev.unsqueeze(0)  # [1, Batch, C, H, W]
            Y = Y.unsqueeze(0)  # [1, Batch, C_out, H, W]

        T, _ = a_prev.shape[:2]
        patch_dim = self.in_c * self.k * self.k
        out_channels = self.W.shape[0]

        denominator = torch.zeros(
            (patch_dim, patch_dim), device=a_prev.device, dtype=a_prev.dtype
        )
        numerator = torch.zeros(
            (out_channels, patch_dim), device=Y.device, dtype=Y.dtype
        )

        for t in range(T):
            if getattr(self, "padding_mode", "zeros") == "circular":
                a_padded = torch.nn.functional.pad(
                    a_prev[t], (self.p, self.p, self.p, self.p), mode="circular"
                )
                P_t = torch.nn.functional.unfold(
                    a_padded, kernel_size=self.k, padding=0, stride=self.s
                )
            else:
                P_t = torch.nn.functional.unfold(
                    a_prev[t], kernel_size=self.k, padding=self.p, stride=self.s
                )

            P_t = P_t.transpose(1, 2).reshape(-1, patch_dim)
            Y_flat = Y[t].movedim(self.channel_dim, -1).reshape(-1, out_channels)

            denominator += P_t.t() @ P_t
            numerator += Y_flat.t() @ P_t

        return numerator, denominator
