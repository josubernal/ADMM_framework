import torch
import torch.fft
import torch.nn.functional as F

class ADMM_Convolution:
    """A unified Mixin to handle FFT-based activation solvers for both 
    standard 2D (4D tensors) and Spiking (5D tensors) convolutions.
    
    Leverages the Convolution Theorem to perform heavy matrix inversions 
    in the frequency domain as element-wise operations.
    """

    def _get_fft_matrices(self, target_shape: tuple):
        """Pads weights, transforms them to the frequency domain, and computes W^H W.
        
        Formula:
            W_padded = pad(W) shifted to origin
            W_fft = FFT(W_padded)
            WtW_fft = W_fft^H @ W_fft
            
        Args:
            target_shape (tuple): The spatial shape (H, W) of the target activations.
            
        Returns:
            tuple:
                - torch.Tensor: The frequency-domain covariance matrix (WtW_fft).
                - int: The number of input channels.
        """
        height, width = target_shape[-2:]
        out_channels, in_channels, kernel_h, kernel_w = self.W.shape
        
        pad_h = height - kernel_h
        pad_w = width - kernel_w
        padded_W = F.pad(self.W, (0, pad_w, 0, pad_h))
        
        shift_h = - (kernel_h // 2)
        shift_w = - (kernel_w // 2)
        padded_W = torch.roll(padded_W, shifts=(shift_h, shift_w), dims=(-2, -1))
        
        W_fft = torch.fft.fft2(padded_W)
        
        W_fft = W_fft.permute(2, 3, 0, 1)
        
        WtW_fft = torch.matmul(W_fft.conj().transpose(-2, -1), W_fft)
        
        return WtW_fft, in_channels

    def _get_a_denominator(self, beta_current: float, a_shape: tuple, temp_den=0.0, rho_current=None, thetas_current=None, unrolled=False, **kwargs):
        """Computes the denominator matrix for the activation (a) update step in the frequency domain.
        
        Formula:
            denominator = beta * I + rho * W^H W
            
        Args:
            beta_current (float): The penalty parameter beta.
            a_shape (tuple): The shape of the activation tensor.
            temp_den (float or torch.Tensor, optional): Explicit temporal penalty. Defaults to 0.0.
            rho_current (float, optional): Spiking layer rho penalty.
            thetas_current (float, optional): Spiking layer theta penalty.
            unrolled (bool, optional): Whether to explicitly invert the matrix for unrolled loops.
            **kwargs: Absorbs any extra Spiking variables safely via duck-typing.
            
        Returns:
            tuple:
                - torch.Tensor: The computed denominator_main matrix.
                - torch.Tensor: The computed denominator_last matrix.
                - int: The number of input channels.
        """
        WtW_fft, in_channels = self._get_fft_matrices(a_shape)
        I = torch.eye(in_channels, device=self.W.device, dtype=WtW_fft.dtype)
        
        if rho_current is not None and thetas_current is not None:
            temp_den = rho_current * (thetas_current ** 2)
            
        temporal_penalty_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
        
        denominator_last = beta_current * I + self.rho * WtW_fft
        denominator_main = denominator_last + (temporal_penalty_scalar * I)
        
        if unrolled:
            denominator_main = torch.linalg.inv(denominator_main)
            denominator_last = torch.linalg.inv(denominator_last)
            
        return denominator_main, denominator_last, in_channels
    
    def solve_activation_system(self, numerator: torch.Tensor, denominator_main: torch.Tensor, denominator_last: torch.Tensor, a_shape: tuple, in_features: int) -> torch.Tensor:
        """Universal Solver for the a update in the frequency domain.

        Solves the linear system: 
        denominator * a = numerator
        
        Args:
            numerator (torch.Tensor): The precomputed numerator tensor.
            denominator_main (torch.Tensor): The FFT denominator matrix for t < T.
            denominator_last (torch.Tensor): The FFT denominator matrix for t = T.
            a_shape (tuple): The shape of the activation tensor.
            in_features (int): Number of input channels (kept for spatial duck-typing).

        Returns:
            torch.Tensor: The exact updated activations in the spatial domain.
        """
        is_spiking = (numerator.dim() == 5)

        if not is_spiking:
            # 4D STANDARD SPATIAL SOLVE
            batch_size = a_shape[0]
            numerator_fft = torch.fft.fft2(numerator).permute(0, 2, 3, 1).unsqueeze(-1)
            
            # Use denominator_last directly
            denominator_expanded = denominator_last.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            a_fft = torch.linalg.solve(denominator_expanded, numerator_fft)
            
            a_fft = a_fft.squeeze(-1).permute(0, 3, 1, 2)
            return torch.fft.ifft2(a_fft).real

        else:
            # 5D SPIKING TEMPORAL SOLVE
            T, batch_size = a_shape[:2]
            numerator_fft = torch.fft.fft2(numerator).permute(0, 1, 3, 4, 2).unsqueeze(-1)
            
            denominator_main_expanded = denominator_main.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            denominator_last_expanded = denominator_last.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            
            a_fft_main = torch.linalg.solve(denominator_main_expanded, numerator_fft[:-1])
            a_fft_last = torch.linalg.solve(denominator_last_expanded, numerator_fft[-1:])
            
            a_fft = torch.cat([a_fft_main, a_fft_last], dim=0)
            a_fft = a_fft.squeeze(-1).permute(0, 1, 4, 2, 3)
            return torch.fft.ifft2(a_fft).real

    def solve_activation_system_unrolled(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """FFT-based unrolled step solver for convolutions.
        
        Executes a batched frequency-domain matrix multiplication for a single timestep.
        
        Args:
            numerator (torch.Tensor): The numerator vector for the current timestep.
            denominator (torch.Tensor): The precomputed inverted denominator matrix.
            
        Returns:
            torch.Tensor: The updated activation tensor for the current timestep in the spatial domain.
        """
        # 1. numerator to frequency domain (operates on spatial H, W dims)
        numerator_fft = torch.fft.fft2(numerator)
        
        # 2. Permute numerator_fft to (H, W, Batch, in_channels, 1) for batched multiplication
        numerator_fft = numerator_fft.permute(2, 3, 0, 1).unsqueeze(-1)
        
        # 3. Expand the inverted denominator to broadcast over Batch
        denominator_expanded = denominator.unsqueeze(2)
        
        # 4. Multiply: Denominator @ Numerator
        a_t_fft = torch.matmul(denominator_expanded, numerator_fft)
        
        # 5. Permute back to standard spatial shape (Batch, in_channels, H, W)
        a_t_fft = a_t_fft.squeeze(-1).permute(2, 3, 0, 1)
        
        # 6. Inverse FFT back to the spatial domain and extract real parts
        return torch.fft.ifft2(a_t_fft).real
         
    def _compute_covariances(self, Y: torch.Tensor, a_prev: torch.Tensor):
        """Computes the numerator and denominator for the weight update.

        Overrides the base method to compute covariances in chunks, preventing 
        Out Of Memory (OOM) errors during the heavy 5D spatiotemporal operations.

        Args:
            Y (torch.Tensor): The target tensor.
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            tuple:
                - torch.Tensor: The computed numerator matrix (Y^T @ P).
                - torch.Tensor: The computed denominator matrix (P^T @ P).
        """        
        if a_prev.dim() == 4:
            a_prev = a_prev.unsqueeze(0) # [1, Batch, C, H, W]
            Y = Y.unsqueeze(0)           # [1, Batch, C_out, H, W]

        T, _ = a_prev.shape[:2]
        patch_dim = self.in_c * self.k * self.k
        out_channels = self.W.shape[0]
        
        denominator = torch.zeros((patch_dim, patch_dim), device=a_prev.device, dtype=a_prev.dtype)
        numerator = torch.zeros((out_channels, patch_dim), device=Y.device, dtype=Y.dtype)
        
        for t in range(T):
            P_t = torch.nn.functional.unfold(a_prev[t], kernel_size=self.k, padding=self.p, stride=self.s)
            
            # P_t is now 3D [Batch, patch_dim, L]. Transpose works perfectly!
            P_t = P_t.transpose(1, 2).reshape(-1, patch_dim) 
            
            Y_flat = Y[t].movedim(self.channel_dim, -1).reshape(-1, out_channels) 
            
            denominator += P_t.t() @ P_t
            numerator += Y_flat.t() @ P_t
            
        return numerator, denominator
