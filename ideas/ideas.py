#In this file I left ideas that can be usefull in the future.
#They must be tested for production.

#####################################################################
# Possible layers
#####################################################################

import torch


class Spiking_ADMMAvgPool2d(Spiking_ADMMConv2d):
    def __init__(self, channels, k, s, p=0, h: nn.Module=None, scale: float=1.0, use_reset: bool=True, init: str="zeros"):
        # An Average Pool is just a Conv layer where in_c == out_c
        super().__init__(in_c=channels, out_c=channels, k=k, p=p, s=s, h=h, scale=scale, use_reset=use_reset, bias=False, init=init)
        
    def setup(self, config: dict, is_last_layer: bool = False):
        super().setup(config, is_last_layer)
        
        # 1. Manually shape the weights to act strictly as Average Pooling
        with torch.no_grad():
            self.W.fill_(0.0) # Zero out cross-channel connections
            val = 1.0 / (self.k * self.k) # The average divisor (e.g., 1/4 for a 2x2 pool)
            
            # Map the 1/k^2 weight strictly to the matching input/output channel
            for c in range(self.in_c):
                self.W[c, c, :, :].fill_(val) 
        
        # 2. Lock the parameters (Pooling doesn't learn)
        self.W.requires_grad = False
        
        # THE FIX: Check if it's a Tensor, since bias=False sets it to the integer 0
        if isinstance(self.b, torch.Tensor):
            self.b.requires_grad = False
            
    def update_weights(self, a_prev, lambda_lagrange=None, cache_pinv=False):
        pass # Override to prevent weight learning

    def update_bias(self, a_prev, lambda_lagrange=None):
        pass # Override to prevent bias learning

########################################################################
# FFTs for deeper convolutional layers (not implemented yet)
########################################################################

def _psf2otf(self, kernel, target_shape):
        """
        Pads a spatial kernel with zeros to match the image shape, 
        and rolls the center to the origin (0,0) for correct FFT phase.
        """
        H, W = target_shape
        k_h, k_w = kernel.shape[-2:]
        
        pad_h = H - k_h
        pad_w = W - k_w
        padded_kernel = torch.nn.functional.pad(kernel, (0, pad_w, 0, pad_h))
        
        shift_h = -(k_h // 2)
        shift_w = -(k_w // 2)
        rolled_kernel = torch.roll(padded_kernel, shifts=(shift_h, shift_w), dims=(-2, -1))
        
        return torch.fft.fft2(rolled_kernel)
    
def _compute_exact_a_fft(self, target, next_layer, temp_num=0.0, temp_den=0.0):
        """
        Exact ADMM Solver using Fast Fourier Transforms (FFTs)
        Assumes next_layer is a Conv2d with stride=1 and padding='same'
        """
        T = self.z.size(0)
        B = self.z.size(1)
        H, W = self.a.shape[-2:]
        
        # 1. Get the exact spatial RHS
        W_T_target = next_layer.adjoint_operator(target, original_input_shape=self.a.shape)
        h_z = self.h(self.z)
        RHS = self.gamma * h_z + next_layer.beta * W_T_target + temp_num
        
        # 2. Prepare the Kernel in Frequency Domain
        # F_W shape: [Out_C, In_C, H, W]
        F_W = self._psf2otf(next_layer.W, (H, W))
        
        # We need the complex conjugate for W^T
        F_W_conj = torch.conj(F_W)
        
        # 3. Calculate Frequencies for LHS Denominator: (gamma * I + beta * W^T * W)
        # Sum over the Out_C dimension to simulate the W^T W dot product
        td_scalar = temp_den[0].view(-1)[0].item() if isinstance(temp_den, torch.Tensor) else temp_den
        
        # F_WtW shape: [In_C, H, W]
        F_WtW = torch.sum(F_W_conj * F_W, dim=0) 
        
        # 4. Transform RHS to Frequency Domain
        # FFT over the spatial dimensions (H, W)
        F_RHS = torch.fft.fft2(RHS, dim=(-2, -1))
        
        # 5. SOLVE! (Exact Division in Complex Space)
        # Handle the main timesteps and the last timestep exactly as before
        F_a_exact = torch.zeros_like(F_RHS)
        
        if T > 1:
            LHS_main = (self.gamma + td_scalar) + next_layer.beta * F_WtW
            F_a_exact[:-1] = F_RHS[:-1] / LHS_main
            
        LHS_last = self.gamma + next_layer.beta * F_WtW
        F_a_exact[-1] = F_RHS[-1] / LHS_last
        
        # 6. Return to Spatial Domain
        # IFFT back to spatial, taking the real part to discard tiny imaginary floating-point errors
        a_exact = torch.fft.ifft2(F_a_exact, dim=(-2, -1)).real
        
        return a_exact

def _update_a_unrolled(self, t, cache): 
        h = self.h(self.z[t])
        is_fft = "F_LHS_main" in cache
        
        if is_fft:
            # --- EXACT FFT DIVISION ---
            TERM3 = self.gamma * h  
            F_TERM3 = torch.fft.fft2(TERM3, dim=(-2, -1))
            
            if t < self.z.size(0) - 1:
                F_a_t = (F_TERM3 / cache["F_LHS_main"]) + cache["F_TERM1*TERM2_a_main"][t]
            else:
                F_a_t = (F_TERM3 / cache["F_LHS_last"]) + cache["F_TERM1*TERM2_a_last"]
                
            a_t_exact = torch.fft.ifft2(F_a_t, dim=(-2, -1)).real
            self.a[t].data.copy_(torch.clamp(a_t_exact, min=0.0, max=1.0))
            
        else:
            # --- EXACT DENSE MATRIX MULTIPLICATION ---
            TERM3 = self.gamma * h.view(h.size(0), -1) 
            
            if t < self.z.size(0) - 1:
                a_t_flat = torch.matmul(TERM3, cache["TERM1_a_main"]) + cache["TERM1*TERM2_a_main"][t]
            else:
                a_t_flat = torch.matmul(TERM3, cache["TERM1_a_last"]) + cache["TERM1*TERM2_a_last"]
            
            # This safely un-flattens the matrix back into the 5D spatial grid!
            self.a[t].data.copy_(torch.clamp(a_t_flat.view_as(self.a[t]), min=0.0, max=1.0))
            
            
def _create_cache(self, next_layer: nn.Module, a_prev: torch.Tensor, lambda_lagrange: torch.Tensor):
        """Precomputes and distributes operations to accelerate the unrolled loop."""
        from .layers import ADMM_SpikingConv2d
        T = self.z.size(0)
        batch_size = self.z.size(1)
        cache = {}

        # 1. CACHE FORWARD PASS
        cache["FORWARD"] = self.spatial_forward(a_prev)

        # 2. CACHE TARGET RHS (TERM 2)
        target_z = next_layer.z - next_layer._format_bias()
        
        if getattr(next_layer, 'spiking', False):
            z_shifted = torch.zeros_like(next_layer.z)
            z_shifted[1:] = next_layer.z[:-1]
            target_z = target_z - next_layer.deltas * z_shifted
        
        if lambda_lagrange is not None:
            target_z[-1] = target_z[-1] + (lambda_lagrange / next_layer.beta)

        # THE CRITICAL SHAPE FIX: Map back to self.a.shape, NOT a_prev.shape!
        # Because we use the new Object's adjoint, this gracefully maps flat features back to 5D.
        projected_target = next_layer.adjoint_operator(target_z, original_input_shape=self.a.shape)
        base_RHS = next_layer.beta * projected_target

        temp_num, _ = self._get_temporal_a_penalties(cache)
        if getattr(self, 'use_reset', False) and isinstance(temp_num, torch.Tensor):
            TERM2 = base_RHS + temp_num
        else:
            TERM2 = base_RHS

        # 3. CACHE INVERSES (TERM 1) AND DISTRIBUTE
        is_next_spatial = isinstance(next_layer,ADMM_SpikingConv2d)

        if is_next_spatial:
            # --- EXACT FFT SOLVER (NO MAJORIZERS) ---
            H, W = self.a.shape[-2:]
            F_W = next_layer._psf2otf(next_layer.W, (H, W))
            F_WtW = torch.sum(torch.conj(F_W) * F_W, dim=0) 
            
            LHS_last_freq = self.gamma + next_layer.beta * F_WtW
            LHS_main_freq = LHS_last_freq + self.beta * (self.thetas ** 2) if getattr(self, 'use_reset', False) else LHS_last_freq
            
            cache["F_LHS_main"] = LHS_main_freq
            cache["F_LHS_last"] = LHS_last_freq
            
            F_TERM2 = torch.fft.fft2(TERM2, dim=(-2, -1))
            cache["F_TERM1*TERM2_a_main"] = F_TERM2 / LHS_main_freq
            cache["F_TERM1*TERM2_a_last"] = F_TERM2[-1] / LHS_last_freq
            
        else:
            # --- EXACT DENSE MATRIX INVERSION (NO MAJORIZERS) ---
            
            # THE MAGIC TRICK: Universal Weight Expansion for ANY Pooling Object
            is_pooled = hasattr(next_layer, 'pool_op') and next_layer.pool_op.__class__.__name__ != 'ADMM_Flatten'
            
            if is_pooled and self.a.dim() > 3:
                H, W_dim = self.a.shape[-2:]
                out_f = next_layer.W.shape[0]
                in_c = self.a.shape[-3]
                
                # Dynamically get the pool size, default to (1, 1) for GAP
                pool_h, pool_w = next_layer.pool_op.output_size if hasattr(next_layer.pool_op, 'output_size') else (1, 1)
                
                # 1. Reshape weights to the pooled spatial grid [Out, Channels, Pool_H, Pool_W]
                W_spatial = next_layer.W.view(out_f, in_c, pool_h, pool_w)
                
                # 2. Upsample weights to match the full spatial grid [Out, Channels, H, W]
                W_expanded = torch.nn.functional.interpolate(W_spatial, size=(H, W_dim), mode='nearest')
                
                # 3. Scale by the pooling factor (Mean property)
                scale = (pool_h * pool_w) / (H * W_dim)
                W_flat = (W_expanded * scale).view(out_f, -1)
            else:
                W_flat = next_layer.W.view(next_layer.W.size(0), -1)
                
            I = torch.eye(W_flat.size(1), device=next_layer.W.device, dtype=W_flat.dtype)
            WtW = torch.matmul(W_flat.t(), W_flat) 
                
            LHS_last = self.gamma * I + next_layer.beta * WtW
            LHS_main = LHS_last + self.beta * (self.thetas ** 2) * I if getattr(self, 'use_reset', False) else LHS_last
            
            TERM1_a_main = torch.linalg.inv(LHS_main).t()          
            TERM1_a_last = torch.linalg.inv(LHS_last).t()
            
            cache["TERM1_a_main"] = TERM1_a_main
            cache["TERM1_a_last"] = TERM1_a_last
            
            # Flatten TERM2 safely for matrix multiplication
            TERM2_flat = TERM2.view(T * batch_size, -1)
            A_main_cache_flat = torch.matmul(TERM2_flat, TERM1_a_main)
            
            cache["TERM1*TERM2_a_main"] = A_main_cache_flat.view(T, batch_size, -1)
            cache["TERM1*TERM2_a_last"] = torch.matmul(TERM2[-1].view(batch_size, -1), TERM1_a_last)
            
        return cache
    
            