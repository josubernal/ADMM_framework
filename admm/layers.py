"""
ADMM Layer Definitions

This module contains the linear and convolutional layers for ADMM.

Each layer must provide:
1. A forward pass (`spatial_forward`).
2. An `adjoint_operator` (transpose) to map targets/errors back to the input space.
3. A `_compute_P` method to extract the "patch matrix", which is used to solve the weight update step.

INDEX:
- ADMMLinear
- ADMMConv2d
- Spiking_ADMMLinear
- Spiking_ADMMConv2d
"""

import warnings
import torch
import torch.nn as nn

from .spiking_mixin import ADMM_Spiking
from .affine import ADMM_AffineLayer

from .convolutional_mixin import ADMM_Convolution
####################################################################################################
# LAYERS
####################################################################################################

class ADMM_Linear(ADMM_AffineLayer):
    """
    Standard Fully Connected Layer for ADMM.
    
    How it works:
    - Forward: Applies a linear transformation (y = xW^T + b) to the input.
    - Adjoint: Maps the target vector back to the input space using the transpose of the weights.
    """
    def __init__(self, in_f, out_f, h: nn.Module=None, bias: bool=False, init: str="pytorch", pool_op=None):
        super().__init__(h=h, bias=bias, pool_op=pool_op) 
        self.init = init
        self.channel_dim = -1 # Targets the [B, C] dimension
        self.in_f = in_f
        self.out_f = out_f

        
    def setup(self, config: dict, is_last_layer: bool = False):
        super().setup(config, is_last_layer)
        self._init_weights_and_bias((self.out_f, self.in_f), (self.out_f,))

    def spatial_forward(self, x, use_bias=True):
        b = self.b if (isinstance(use_bias, bool) and use_bias and self.bias) else None
        return torch.nn.functional.linear(self.pool_op(x), self.W, bias=b)
    
    def adjoint_operator(self, target, original_input_shape=None):
        out = torch.matmul(target, self.W)
        return self.pool_op.adjoint(out, original_input_shape=original_input_shape)
    
    def _compute_P(self, a_prev): 
        return self.pool_op(a_prev)
    
    def _get_bias_reduction_dims(self): 
        return 0

class ADMM_Conv2d( ADMM_Convolution, ADMM_AffineLayer):
    """
    Standard 2D Convolutional Layer for ADMM.
    
    How it works:
    - Forward: Applies a standard 2D convolution.
    - Adjoint: Applies a Transposed Convolution (Deconvolution) to map the feature space 
      back to the spatial image space. Dynamically calculates output padding to reconstruct
      the original input shape.
    - Compute P: Uses `unfold` (im2col) to extract sliding local blocks from the image 
      into a flat patch matrix for the weight update step.
    """
    def __init__(self, in_c, out_c, k, p, s, h: nn.Module=None, bias: bool=False, init: str="pytorch",  pool_op=None, use_fft=True, padding_mode="circular"):
        super().__init__(h=h, bias=bias,  pool_op=pool_op, use_fft=use_fft, padding_mode=padding_mode)
        self.init = init
        self.p = p
        self.s = s
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.channel_dim = -3 # Targets the [B, C, H, W] dimension    
        

    def setup(self, config: dict, is_last_layer: bool = False):
        super().setup(config, is_last_layer)
        self._init_weights_and_bias((self.out_c, self.in_c, self.k, self.k), (self.out_c,))

    def spatial_forward(self, x, use_bias=True):
        if self.padding_mode == 'circular':
            return self._circular_forward(x, use_bias)
        b = self.b if (isinstance(use_bias, bool) and use_bias and self.bias) else None
        return torch.nn.functional.conv2d(x, self.W, bias=b, padding=self.p, stride=self.s)

    def adjoint_operator(self, target, original_input_shape=None):
        if self.padding_mode == 'circular':
            return self._circular_adjoint(target)
      
        out_pad = (0, 0)
        if original_input_shape is not None:
            H_in, W_in = original_input_shape[-2:]
            H_dim, W_dim = target.shape[-2:]
            H_calc = (H_dim - 1) * self.s - 2 * self.p + self.k
            W_calc = (W_dim - 1) * self.s - 2 * self.p + self.k
            out_pad = (max(0, H_in - H_calc), max(0, W_in - W_calc))
            
        return torch.nn.functional.conv_transpose2d(
            target, self.W, padding=self.p, stride=self.s, output_padding=out_pad
        )
    def _compute_P(self, a_prev):
        """Extracts image patches using unfold (im2col) for localized weight updates.

        Args:
            a_prev (torch.Tensor): The previous layer's activations.

        Returns:
            torch.Tensor: The flattened patch matrix $P$.
        """
        if self.padding_mode == 'circular':
            return self._circular_compute_P(a_prev)
        patches = torch.nn.functional.unfold(a_prev, kernel_size=self.k, padding=self.p, stride=self.s)
        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)
    
    def _get_bias_reduction_dims(self): 
        return (0, 2, 3) 

class ADMM_SpikingLinear(ADMM_Spiking, ADMM_AffineLayer):
    """
    Spiking Fully Connected Layer.
    
    How it works:
    - Designed for 5D tensors: (Time, Batch, Features).
    - Folds the Time and Batch dimensions together to process the entire sequence as 
      a standard 2D matrix operation, then unfolds it back to the temporal sequence.
    """
    def __init__(self, in_f, out_f, h: nn.Module=None, use_reset: bool=True, bias: bool=False, init: str="s-uniform", pool_op=None):
        super().__init__(h=h, bias=bias, pool_op=pool_op)
        self.init = init
        self.T = None
        self.spiking = True
        self.use_reset = use_reset
        self.in_f = in_f
        self.out_f = out_f
        self.channel_dim = -1 

     
    def setup(self, config: dict, is_last_layer: bool = False):
        super().setup(config, is_last_layer)
        self._init_weights_and_bias((self.out_f, self.in_f), (self.out_f,))

    def spatial_forward(self, x, use_bias=True):
        b = self.b if (isinstance(use_bias, bool) and use_bias and self.bias) else None
        x_flat, tb_shape = self._fold_time(x)
        x_pooled = self.pool_op(x_flat)
        
        out_flat = torch.matmul(x_pooled, self.W.t())
        if b is not None:
            out_flat += b
            
        return self._unfold_time(out_flat, tb_shape)
    
    def adjoint_operator(self, target, original_input_shape=None):
        target_flat, tb_shape = self._fold_time(target)
        deconv_flat = torch.matmul(target_flat, self.W)
        return self.pool_op.adjoint(deconv_flat, original_input_shape, tb_shape)

    def _compute_P(self, a_prev): 
        a_flat, _ = self._fold_time(a_prev)
        return self.pool_op(a_flat)

    def _get_bias_reduction_dims(self):
        return (0, 1)

class ADMM_SpikingConv2d(ADMM_Convolution, ADMM_Spiking, ADMM_AffineLayer):
    """
    Spiking 2D Convolutional Layer.
    
    How it works:
    - Designed for 5D tensors: (Time, Batch, Channels, Height, Width).
    - Uses time folding to apply standard 2D convolutions efficiently across all timesteps.
    - Includes a chunked covariance computation to prevent Out-Of-Memory (OOM) errors 
      during the ADMM weight update step.
    """
    def __init__(self, in_c, out_c, k, p, s, h: nn.Module=None, use_reset: bool=True, bias: bool=False, init: str="s-uniform", pool_op=None, use_fft=True, padding_mode="circular"):
        super().__init__(h=h, bias=bias, pool_op=pool_op, use_fft=use_fft, padding_mode=padding_mode) 
        self.init = init
        self.T = None
        self.spiking = True
        self.use_reset = use_reset
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.p = p
        self.s = s
        self.channel_dim = -3          
    
    def setup(self, config: dict, is_last_layer: bool = False):
        super().setup(config, is_last_layer)
        self._init_weights_and_bias((self.out_c, self.in_c, self.k, self.k), (self.out_c,))
        
    def spatial_forward(self, x, use_bias=True):
        if self.padding_mode == 'circular':
            return self._circular_forward(x, use_bias)
        b = self.b if (isinstance(use_bias, bool) and use_bias and self.bias) else None
        x_flat, tb_shape = self._fold_time(x)
        out_flat = torch.nn.functional.conv2d(x_flat, self.W, bias=b, padding=self.p, stride=self.s)
        return self._unfold_time(out_flat, tb_shape)

    def adjoint_operator(self, target, original_input_shape=None):
        if self.padding_mode == 'circular':
            return self._circular_adjoint(target)
        target_flat, tb_shape = self._fold_time(target)
        out_pad = (0, 0)
        if original_input_shape is not None:
            H_in, W_in = original_input_shape[-2:]
            H_dim, W_dim = target_flat.shape[-2:]
            out_pad = (max(0, H_in - ((H_dim - 1) * self.s - 2 * self.p + self.k)), 
                       max(0, W_in - ((W_dim - 1) * self.s - 2 * self.p + self.k)))
            
        out_flat = torch.nn.functional.conv_transpose2d(target_flat, self.W, padding=self.p, stride=self.s, output_padding=out_pad)
        return self._unfold_time(out_flat, tb_shape)
    
    def _compute_P(self, a_prev):
        if self.padding_mode == 'circular':
            return self._circular_compute_P(a_prev)
        a_flat, _ = self._fold_time(a_prev)
        patches = torch.nn.functional.unfold(a_flat, kernel_size=self.k, padding=self.p, stride=self.s)
        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)
    
    def _get_bias_reduction_dims(self): 
        return (0, 1, 3, 4)
