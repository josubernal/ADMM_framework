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

