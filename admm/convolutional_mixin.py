
import torch.fft
import torch.nn.functional as F


class ADMM_Convolution:
    """
    A unified Mixin to handle FFT-based activation solves for both 
    standard 2D (4D tensors) and Spiking (5D tensors) convolutions.
    """
    def _get_fft_matrices(self, target_shape: tuple):
        """Pads weights, transforms to frequency domain, and computes W^H W."""
        # target_shape[-2:] safely grabs (H, W) for both 4D and 5D shapes
        H, W_dim = target_shape[-2:]
        out_c, in_c, k_h, k_w = self.W.shape
        
        # Pad right and bottom to match the target spatial shape
        pad_h = H - k_h
        pad_w = W_dim - k_w
        padded_W = F.pad(self.W, (0, pad_w, 0, pad_h))
        
        # Shift the kernel to align the phase (center to origin)
        shift_h = - (k_h // 2)
        shift_w = - (k_w // 2)
        padded_W = torch.roll(padded_W, shifts=(shift_h, shift_w), dims=(-2, -1))
        
        # 2D FFT on the spatial dims
        W_fft = torch.fft.fft2(padded_W)
        
        # Rearrange to (H, W, out_c, in_c) for channel-wise broadcasting
        W_fft = W_fft.permute(2, 3, 0, 1)
        
        # W^H @ W -> Shape: (H, W, in_c, in_c)
        WtW_fft = torch.matmul(W_fft.conj().transpose(-2, -1), W_fft)
        return WtW_fft, in_c

    def solve_activation_system(self, numerator: torch.Tensor, a_shape: tuple, beta_current: float, temp_den=0.0) -> torch.Tensor:
        """Solves the system in the frequency domain, adapting to 4D or 5D."""
        WtW_fft, in_c = self._get_fft_matrices(a_shape)
        
        I = torch.eye(in_c, device=self.W.device, dtype=WtW_fft.dtype)
        td_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
        
        # Base Spatial Denominator (Used for t=T in Spiking, or standard Spatial)
        A_last = beta_current * I + self.rho * WtW_fft
        
        is_spiking = (numerator.dim() == 5)

        if not is_spiking:
            # ==========================================
            # STANDARD SPATIAL SOLVE (4D: B, C, H, W)
            # ==========================================
            A = A_last + td_scalar * I 
            B_dim, _, H, W_dim = a_shape
            
            # fft2 operates on last two dims by default: (B, C, H, W)
            # Permute to (B, H, W, C, 1) for batched linear solve
            num_fft = torch.fft.fft2(numerator).permute(0, 2, 3, 1).unsqueeze(-1)
            
            A_expanded = A.unsqueeze(0).expand(B_dim, -1, -1, -1, -1)
            a_fft = torch.linalg.solve(A_expanded, num_fft)
            
            a_fft = a_fft.squeeze(-1).permute(0, 3, 1, 2)
            return torch.fft.ifft2(a_fft).real

        else:
            # ==========================================
            # SPIKING TEMPORAL SOLVE (5D: T, B, C, H, W)
            # ==========================================
            T, B_dim, _, H, W_dim = a_shape
            
            # SNNs apply the temp_den penalty to all timesteps EXCEPT the last one
            A_main = A_last + td_scalar * I
            
            # Permute to (T, B, H, W, C, 1)
            num_fft = torch.fft.fft2(numerator).permute(0, 1, 3, 4, 2).unsqueeze(-1)
            
            A_main_exp = A_main.unsqueeze(0).expand(B_dim, -1, -1, -1, -1)
            A_last_exp = A_last.unsqueeze(0).expand(B_dim, -1, -1, -1, -1)
            
            # Solve t < T-1 using A_main
            a_fft_main = torch.linalg.solve(A_main_exp, num_fft[:-1])
            
            # Solve t == T-1 using A_last
            a_fft_last = torch.linalg.solve(A_last_exp, num_fft[-1:])
            
            # Recombine time and reformat to (T, B, C, H, W)
            a_fft = torch.cat([a_fft_main, a_fft_last], dim=0)
            a_fft = a_fft.squeeze(-1).permute(0, 1, 4, 2, 3)
            return torch.fft.ifft2(a_fft).real
    
    def _get_a_denominator(self, beta_current, a_shape, temp_den=0.0, unrolled=False):
        """
        FFT-aware denominator computation for unrolled SNN temporal caches.
        """
        WtW_fft, in_c = self._get_fft_matrices(a_shape)
        
        I = torch.eye(in_c, device=self.W.device, dtype=WtW_fft.dtype)
        td_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
        
        # Base spatial denominator (used for t=T)
        denominator_last = beta_current * I + self.rho * WtW_fft
        
        # Main temporal denominator (used for t < T)
        denominator_main = denominator_last + (td_scalar * I)
        
        if unrolled:
            # Invert the [in_c, in_c] matrices at each spatial frequency [H, W]
            # torch.linalg.inv operates perfectly on the last two dimensions of complex tensors!
            denominator_main = torch.linalg.inv(denominator_main)
            denominator_last = torch.linalg.inv(denominator_last)
            
        return denominator_main, denominator_last, in_c

    def solve_unrolled_step(self, RHS: torch.Tensor, math_box: torch.Tensor) -> torch.Tensor:
        """
        FFT-based unrolled step solver for convolutions.
        Executes a batched frequency-domain matrix multiplication for a single timestep.
        """
        # 1. RHS to frequency domain (operates on spatial H, W dims)
        RHS_fft = torch.fft.fft2(RHS)
        
        # 2. Permute RHS_fft to (H, W, Batch, in_c, 1) for batched multiplication
        RHS_fft = RHS_fft.permute(2, 3, 0, 1).unsqueeze(-1)
        
        # 3. Expand the inverted math_box from (H, W, in_c, in_c) to broadcast over Batch
        math_box_exp = math_box.unsqueeze(2)
        
        # 4. Multiply: (H, W, 1, in_c, in_c) @ (H, W, Batch, in_c, 1)
        a_t_fft = torch.matmul(math_box_exp, RHS_fft)
        
        # 5. Permute back to standard spatial shape (Batch, in_c, H, W)
        a_t_fft = a_t_fft.squeeze(-1).permute(2, 3, 0, 1)
        
        # 6. Inverse FFT back to the spatial domain and extract real parts
        return torch.fft.ifft2(a_t_fft).real
         