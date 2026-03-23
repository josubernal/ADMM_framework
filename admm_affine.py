import torch
import random
import torch.nn as nn
import configparser
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import tonic
from tonic import DiskCachedDataset
import os
import json
import matplotlib.pyplot as plt


class ADMM(nn.Module):
    """
    Universal Manager. Handles both Static and Spiking ADMM networks automatically.
    """
    def __init__(self, layers: nn.ModuleList, beta: float = 1.0, gamma: float = 1.0, device=None, **kwargs):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.layers = layers
        self.L = len(self.layers)
        self.initialized = False
        self.lambda_lagrange = None
        
        self.beta = beta
        self.gamma = gamma
   
        config = {'beta': self.beta, 'gamma': self.gamma}
        config.update(kwargs)
        
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._configure_layers(config)

    def _configure_layers(self, config_dict: dict):
        """Explicitly passes parameters to layers."""
        for i, layer in enumerate(self.layers):
            layer.to(self.device)
            for key, val in config_dict.items():
                if hasattr(layer, key):
                    setattr(layer, key, val)
                
                # Handle activation function parameters
                if key == 'thetas' and hasattr(layer, 'h') and hasattr(layer.h, 'thetas'):
                    layer.h.thetas = val
                                       
            # Turn off reset for the final continuous readout layer
            if i == len(self.layers) - 1 and hasattr(layer, 'use_reset'):
                layer.use_reset = False
    
    def _is_spiking(self):
        """Helper to determine if the network has a temporal dimension."""
        return hasattr(self, 'T') and self.T is not None

    def _init_states(self, inputs):
        """Populates the initial z and a variables and initializes lambda."""
        x = inputs.to(self.device)
        self.initialized = True
        
        with torch.no_grad():
            for layer in self.layers:
                z_pred = layer.forward(x)
                a_pred = layer.h(z_pred)
                layer.z = z_pred.clone()
                layer.a = a_pred.clone()
                x = a_pred 
                    
        last_z = self.layers[-1].z
        target_shape = last_z[-1] if self._is_spiking() else last_z
        self.lambda_lagrange = torch.zeros_like(target_shape, device=self.device)
        
    def _get_batchsize(self, inputs):
        return inputs.shape[1] if self._is_spiking() else inputs.shape[0]
    
    def forward_model(self, inputs):
        """Standard feed-forward pass (Inference)."""
        x = inputs.to(self.device)
        final_z = None
        
        # 1. Initialize a list to store the firing rates for each hidden layer
        firing_rates = [] 
        
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                z_pred = layer.forward(x)
                x = layer.h(z_pred) 
                
                if i < len(self.layers) - 1:
                    # Sum of all spikes divided by the total number of elements
                    layer_firing_rate = x.sum().item() / x.numel()
                    firing_rates.append(layer_firing_rate)
                    
                final_z = z_pred
                
        final_out = final_z[-1] if self._is_spiking() else final_z
        batch_size = inputs.size(1) if self._is_spiking() else inputs.size(0)
            
        if final_out.dim() > 2:
            final_out = final_out.view(batch_size, -1)

        return final_out, firing_rates

    def _lambda_update(self, last_layer, a_prev_L):
        spatial_out = last_layer.spatial_forward(a_prev_L)
        if self._is_spiking():
            z_T = last_layer.z[-1]
            z_T_minus_1 = last_layer.z[-2]
            F_a_T = spatial_out[-1]
            
            residual = z_T - (last_layer.deltas * z_T_minus_1) - F_a_T
        else:
            residual = last_layer.z - spatial_out

        self.lambda_lagrange += 2 * self.beta * residual
        self.lambda_lagrange = torch.clamp(self.lambda_lagrange,-0.05 * self.beta, 0.05 * self.beta)

    
    def fit(self, inputs: torch.Tensor, labels: torch.Tensor, warming: bool = False):
        """Orchestrates the fitting loop."""
        with torch.no_grad():
            if self.lambda_lagrange is None or self.lambda_lagrange.shape[0] != self._get_batchsize(inputs):
                self._init_states(inputs)
            
            random_layers = random.sample(range(self.L - 1), self.L - 1)

            for l in random_layers:
                layer = self.layers[l]
                next_layer = self.layers[l+1]
                a_prev = inputs if l == 0 else self.layers[l - 1].a

                layer.update_weights(a_prev)
                layer.update_bias(a_prev)

                lagrange = self.lambda_lagrange if l == self.L - 2 else None
                layer.update_a(next_layer, a_prev, lagrange)
                layer.update_z(a_prev)  
            
            # Update last layer
            last_layer = self.layers[-1]
            a_prev_L = self.layers[-2].a if len(self.layers) > 1 else inputs
            
            last_layer.update_weights(a_prev_L, self.lambda_lagrange)
            last_layer.update_bias(a_prev_L, self.lambda_lagrange)
            last_layer.update_z_last(a_prev_L, labels, self.lambda_lagrange)
            
            # Lambda update
            if not warming:
                self._lambda_update(last_layer, a_prev_L)
    
    #Metrics
    def lagrangian_cost(self, inputs, labels):
        """Calculates the ADMM energy/cost to track convergence."""
        cost = 0.0
        batch_size = self._get_batchsize(inputs)
        last_layer = self.layers[-1]
        
        z_last_flat = last_layer.z[-1].view(batch_size, -1) if self._is_spiking() else last_layer.z.view(batch_size, -1)
        cost += torch.norm(z_last_flat - labels)**2 
        
        a_prev_L = self.layers[-2].a if len(self.layers) > 1 else inputs
        last_out = last_layer.vectorized_forward(a_prev_L)
        last_layer_cost = last_layer.z - last_out
        
        lambda_term = last_layer_cost[-1] if self._is_spiking() else last_layer_cost

        cost += torch.sum(self.lambda_lagrange * lambda_term)
        cost += self.beta * torch.norm(last_layer_cost)**2
        
        for l in range(self.L - 1):
            a_prev = inputs if l == 0 else self.layers[l - 1].a
            layer = self.layers[l]
            
            physical_out = layer.vectorized_forward(a_prev)

            cost += self.beta * torch.norm((layer.z - physical_out))**2
            cost += self.gamma * torch.norm((layer.a - layer.h(layer.z))) ** 2
            
        return cost.item()
    
    
    def primal_residual_norm(self, inputs):
        """
        Calculates the normalized L1 sum of the primal residual.
        """
        last_layer = self.layers[-1]
        a_prev_L = self.layers[-2].a if len(self.layers) > 1 else inputs
        last_out = last_layer.vectorized_forward(a_prev_L)
        residual = last_layer.z - last_out
        r = residual[-1] if self._is_spiking() else residual
        norm_factor = r.numel() ** 0.5
        
        return (torch.norm(r) / norm_factor).item()

    def preactivation_constraint_sum(self, inputs):
        """
        Calculates the normalized L2 norm of the pre-activation constraint for each layer.
        """
        constraints_residuals = []
        
        with torch.no_grad():
            for l, layer in enumerate(self.layers):
                a_prev = inputs if l == 0 else self.layers[l - 1].a
                predicted_z = layer.vectorized_forward(a_prev)
                residual = layer.z - predicted_z
                norm_factor = residual.numel() ** 0.5
                val = (torch.norm(residual) / norm_factor).item()
                constraints_residuals.append(val)
                
        return constraints_residuals
  
    def activation_constraint_sum(self):
        """
        Calculates the normalized L2 norm of the activation constraint for hidden layers.
        """
        constraints_residuals = []
        
        with torch.no_grad():
            for l in range(self.L - 1):
                layer = self.layers[l]
                residual = layer.a - layer.h(layer.z)     
                norm_factor = residual.numel() ** 0.5
                val = (torch.norm(residual) / norm_factor).item()
                constraints_residuals.append(val)
                
        return constraints_residuals

class ADMMLayer(nn.Module):
    """The Base Layer Interface."""
    def __init__(self, h: nn.Module = None):
        super().__init__()
        self.device = self.beta = self.gamma = None
        self.z = self.a = None
        self.deltas = self.thetas = None
        self.spiking = False 
        self.h = h if h is not None else ADMM_ReLU()
    
    def forward(self, x):    
        """
        STANDARD SEQUENTIAL PASS (Initialization / Inference)  
        """
        y = self.spatial_forward(x)     
        if self.spiking:
            z = torch.zeros_like(y)
            z_prev = torch.zeros_like(y[0])
            a_prev = torch.zeros_like(y[0])
            
            for t in range(self.T):
                reset = self.thetas * a_prev if (t > 0 and self.use_reset) else 0
                z_t = y[t] + self.deltas * z_prev - reset
                
                z[t] = z_t
                z_prev = z_t
                a_prev = self.h(z_t)
                
        return y
   
    def vectorized_forward(self, a_prev):
        """
        DECOUPLED ADMM PASS (Optimization / Constraint Evaluation)
        Evaluates the network's time as vector dimensionality.
        """
        A = self.spatial_forward(a_prev) 
        if self.spiking: 
            A += self._compute_temporal_dependencies()
        return A

    #-----------Helper-------------------
    def _broadcast_to_match(self, tensor, target_tensor):
        if tensor.dim() < target_tensor.dim():
            missing_dims = target_tensor.dim() - tensor.dim()
            return tensor.view(*tensor.shape, *([1] * missing_dims))
        return tensor 
    
    #------------ Updates--------------
    def update_z_last(self, a_prev, labels, lambda_lagrange):
        spatial_res = self.spatial_forward(a_prev)
        pred = self.vectorized_forward(a_prev)
        shape = self.z[-1] if self.spiking else pred
        
        labels_sp = self._broadcast_to_match(labels, shape)
        lambda_sp = self._broadcast_to_match(lambda_lagrange, shape)
        
        if self.spiking:
            num, den = self._get_temporal_z_penalties(spatial_res, labels_sp, lambda_sp)
        else:
            num = labels_sp - (lambda_sp / 2.0)
            den = 1.0
            
        numerator = (self.beta * pred) + num
        denominator = self.beta + den
        
        self.z.data.copy_(numerator / denominator)
            
class ADMMAffineLayer(ADMMLayer):
    """Affine layer interface."""
    def __init__(self, h: nn.Module= None):
        super().__init__(h)
        self.use_exact_solver = False

    def _init_weights_and_bias(self, weight_shape, bias_shape, scale=1.0):
        w = torch.empty(*weight_shape)
        nn.init.xavier_uniform_(w)
        self.W = nn.Parameter(w * scale)
        self.b = nn.Parameter(torch.zeros(*bias_shape))
        
    def _format_bias(self):
        target_shape = [1] * self.z.dim()
        target_shape[self.channel_dim] = self.b.size(0)
        return self.b.view(*target_shape)
    
    def _compute_exact_a(self, target,next_layer):
        h_z = self.h(self.z)
        target_flat = target.view(target.size(0), -1)
        h_z_flat = h_z.view(h_z.size(0), -1)
            
        W_T_target = torch.matmul(target_flat, next_layer.W)
        RHS = self.gamma * h_z_flat + next_layer.beta * W_T_target 
            
        I = torch.eye(next_layer.W.size(1), device=next_layer.W.device)
        WtW = torch.matmul(next_layer.W.t(), next_layer.W) 
        LHS_operator = self.gamma * I + next_layer.beta * WtW   
        new_a_flat = torch.linalg.solve(LHS_operator, RHS.t()).t()
        return new_a_flat.view_as(self.a)

    def update_weights(self, a_prev, lambda_lagrange=None):
        P = self._compute_P(a_prev)
        Y = self.z - self._format_bias()
        
        if self.spiking:
            Y = Y - self._compute_temporal_dependencies()
            if lambda_lagrange is not None:
                lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z[-1])
                Y[-1] = Y[-1] + (lam_spatial / (2 * self.beta))
        else:
            if lambda_lagrange is not None:
                lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z)
                Y = Y + (lam_spatial / (2 * self.beta))

        Y = Y.movedim(self.channel_dim, -1).reshape(P.size(0), self.W.shape[0])
        
        W_old_flat = self.W.reshape(self.W.shape[0], -1).t()

        PtP = torch.matmul(P.t(), P) 
        PtY = torch.matmul(P.t(), Y) 
        
        PtR = PtY - torch.matmul(PtP, W_old_flat)
        
        pinv_PtP = torch.linalg.pinv(PtP)
        
        delta_W_flat = torch.matmul(pinv_PtP, PtR)
        new_W_flat = W_old_flat + delta_W_flat

        self.W.data.copy_(new_W_flat.t().reshape(self.W.shape))
        
    def update_bias(self, a_prev, lambda_lagrange=None):
        target = self.z - self.spatial_forward(a_prev, bias=False)
        
        if self.spiking:
            target = target - self._compute_temporal_dependencies()
            if lambda_lagrange is not None:
                lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z[-1])
                target[-1] = target[-1] + (lam_spatial / (2 * self.beta))
        else:
            if lambda_lagrange is not None:
                lam_spatial = self._broadcast_to_match(lambda_lagrange, self.z)
                target = target + (lam_spatial / (2 * self.beta))
                
        new_bias = torch.mean(target, dim=self._get_bias_reduction_dims())
        self.b.data.copy_(new_bias)
    
    def update_z(self, a_prev):
        res = self.spatial_forward(a_prev)
        new_z = self.h.proximal_z_update(
            a=self.a, res=res, gamma=self.gamma, beta=self.beta,
            deltas=getattr(self, 'deltas', None), thetas=getattr(self, 'thetas', None), z=self.z
        )
        self.z.data.copy_(new_z)

    def update_a(self, next_layer, a_prev, lambda_lagrange=None):
        numerator = next_layer.z - next_layer._format_bias()
        if getattr(next_layer, 'spiking', False):
            numerator = numerator - next_layer._compute_temporal_dependencies()
            if lambda_lagrange is not None:
                lam_sp = next_layer._broadcast_to_match(lambda_lagrange, next_layer.z[-1])
                numerator[-1] = numerator[-1] + (lam_sp / (2 * next_layer.beta))
        else:
            if lambda_lagrange is not None:
                lam_sp = next_layer._broadcast_to_match(lambda_lagrange, numerator)
                numerator = numerator + (lam_sp / (2 * next_layer.beta))
                
        temp_num, temp_den = 0.0, 0.0
        if self.spiking:
            temp_num, temp_den = self._get_temporal_a_penalties(a_prev)
            
        if getattr(next_layer, 'use_exact_solver', False) and not getattr(next_layer, 'spiking', False):
            new_a= self._compute_exact_a(numerator, next_layer)
        else:
            backward_pass = next_layer.adjoint_operator(numerator, original_input_shape=self.a.shape)
            backward_pass = backward_pass.view_as(self.a)
    
            energy = next_layer._get_W_energy_for_backward()
            denominator = self.gamma + next_layer.beta * energy + temp_den
            numerator = self.gamma * self.h(self.z) + next_layer.beta * backward_pass + temp_num
            new_a= numerator / denominator
        
        max_val = 1.0 if self.spiking else 10.0
        self.a.data.copy_(torch.clamp(new_a, min=0.0, max=max_val))
    
    def _get_W_energy_for_backward(self):
        if self.W.dim() == 2:
            return torch.linalg.matrix_norm(self.W, ord=2)**2
        else:
            return torch.sum(self.W**2)
class Spiking:
    def _fold_time(self, x):
        if x.dim() == 5 or x.dim() == 3: return x.reshape(x.size(0) * x.size(1), *x.shape[2:]), x.shape[:2]
        return x, None

    def _unfold_time(self, x_flat, tb_shape):
        if tb_shape is None: return x_flat
        return x_flat.reshape(tb_shape[0], tb_shape[1], *x_flat.shape[1:])

    def _compute_temporal_dependencies(self):     
        M = torch.zeros_like(self.z)
        M[1:] = self.deltas * self.z[:-1]
        if getattr(self, 'use_reset', False):
            M[1:] -= self.thetas * self.a[:-1]
        return M
    
    def _get_temporal_z_penalties(self, pred, labels_spatial, lambda_spatial):
        r = self.z - pred 
        
        num, den = torch.zeros_like(self.z), torch.zeros_like(self.z)
        num[:-1] = self.deltas * self.beta * r[1:]
        den[:-1] = (self.deltas ** 2) * self.beta  
        
        num[-2] += (lambda_spatial * self.deltas) / 2
        num[-1] = labels_spatial - (lambda_spatial / 2)
        den[-1] = 1.0
        return num, den

    def _get_temporal_a_penalties(self, a_prev):
        if not getattr(self, 'use_reset', False): return 0.0, 0.0
        
        spatial_out = self.spatial_forward(a_prev)
        r = self.z - spatial_out
        r[1:] -= self.deltas * self.z[:-1]
        
        r_next = torch.zeros_like(r)
        r_next[:-1] = r[1:]
        
        num, den = torch.zeros_like(self.z), torch.zeros_like(self.z)
        num[:-1] = -self.thetas * self.beta * r_next[:-1]
        den[:-1] = (self.thetas ** 2) * self.beta
        return num, den
    
class ADMMLinear(ADMMAffineLayer):
    def __init__(self, in_f, out_f, h: nn.Module= None, scale: float=1.0):
        super().__init__(h)     
        self._init_weights_and_bias((out_f, in_f), (out_f,), scale=scale)
        self.channel_dim = -1 # [B, C]
        self.use_exact_solver = (in_f <= 2000)
        
    def spatial_forward(self, x, bias=True):
        if isinstance(bias, bool):
            bias = self.b if bias else None
        return torch.nn.functional.linear(x.view(x.size(0), -1), self.W, bias=bias)
    
    def adjoint_operator(self, target, original_input_shape=None):
        out = torch.matmul(target, self.W)
        if original_input_shape is not None:
            return out.reshape(original_input_shape)
        return out
    
    def _compute_P(self, a_prev): 
        return a_prev.view(a_prev.size(0), -1)
    
    def _get_bias_reduction_dims(self): 
        return 0

class ADMMConv2d(ADMMAffineLayer):
    def __init__(self, in_c, out_c, k, p, s, h: nn.Module= None, scale: float=1.0):
        super().__init__(h)
        self.p, self.s = p, s
        self.in_c, self.out_c, self.k = in_c, out_c, k
        self.channel_dim = -3 # [B, C, H, W]
        
        self._init_weights_and_bias((out_c, in_c, k, k), (out_c,), scale=scale)

    def spatial_forward(self, x, bias=True):
        if isinstance(bias, bool):
            bias = self.b if bias else None
        return torch.nn.functional.conv2d(x, self.W, bias=bias, padding=self.p, stride=self.s)

    def adjoint_operator(self, target, original_input_shape=None):
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
        patches = torch.nn.functional.unfold(a_prev, kernel_size=self.k, padding=self.p, stride=self.s)
        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)
    
    def _get_bias_reduction_dims(self): 
        return (0, 2, 3)

class Spiking_ADMMLinear(Spiking, ADMMAffineLayer):
    def __init__(self, in_f, out_f, scale: float=1.0, h: nn.Module=None, use_reset: bool=True):
        super().__init__(h=h)
        
        self.T = None
        self.spiking = True
        self.use_reset = use_reset
        self.use_exact_solver =  (in_f <= 2000)
        
        self.in_f, self.out_f = in_f, out_f
        
        self.channel_dim = -1 
        self._init_weights_and_bias((out_f, in_f), (out_f,), scale=scale)

    def spatial_forward(self, x, bias=True):
        if isinstance(bias, bool):
            bias = self.b if bias else None
        x_flat, tb_shape = self._fold_time(x)
        out_flat = torch.matmul(x_flat.reshape(x_flat.size(0), -1), self.W.t())
        if bias is not None:
            out_flat += bias
        return self._unfold_time(out_flat, tb_shape)

    def adjoint_operator(self, target, original_input_shape=None):
        target_flat, tb_shape = self._fold_time(target)
        deconv_flat = torch.matmul(target_flat, self.W)
        if original_input_shape is not None: return deconv_flat.reshape(original_input_shape)
        return self._unfold_time(deconv_flat, tb_shape)

    def _compute_P(self, a_prev): 
        return self._fold_time(a_prev)[0].reshape(a_prev.size(0) * a_prev.size(1), -1)
    
    def _get_bias_reduction_dims(self):
        return (0, 1)
    


class Spiking_ADMMConv2d(Spiking, ADMMAffineLayer):
    def __init__(self, in_c, out_c, k, p, s, h: nn.Module=None, scale: float=1.0, use_reset: bool=True):
        super().__init__( h=h) 
        
        self.T = None
        self.spiking = True
        self.use_reset = use_reset
        
        self.in_c, self.out_c, self.k, self.p, self.s = in_c, out_c, k, p, s

        self.channel_dim = -3 
        self._init_weights_and_bias((out_c, in_c, k, k), (out_c,), scale=scale)
          
    def spatial_forward(self, x, bias=True):
        if isinstance(bias, bool):
            bias = self.b if bias else None
        x_flat, tb_shape = self._fold_time(x)
        out_flat = torch.nn.functional.conv2d(x_flat, self.W, bias=bias, padding=self.p, stride=self.s)
        return self._unfold_time(out_flat, tb_shape)

    def adjoint_operator(self, target, original_input_shape=None):
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
        patches = torch.nn.functional.unfold(self._fold_time(a_prev)[0], kernel_size=self.k, padding=self.p, stride=self.s)
        return patches.transpose(1, 2).reshape(-1, self.in_c * self.k * self.k)
    
    def _get_bias_reduction_dims(self): 
        return (0, 1, 3, 4)

class ADMM_ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.relu(x)
   
    def proximal_z_update(self, a, res, gamma, beta, **kwargs):
        z_pos = (gamma * a + beta * res) / (gamma + beta)
        return torch.where(z_pos > 0, z_pos, res)

class ADMM_Heaviside(nn.Module):
    def __init__(self, thetas):
        super().__init__()  
        self.thetas = thetas
   
    def forward(self, x):
        return (x > self.thetas).float()
   
    def check_entries(self, z, a_l, q_l, r_l, deltas, beta, gamma, thetas): 
        delta1 = gamma * (1 - 2 * a_l)
        r_l_next = torch.zeros_like(r_l)
        r_l_next[:-1] = r_l[1:]

        temporal_mask = torch.ones_like(z)
        temporal_mask[-1] = 0.0

        energy_z = beta * (z - q_l)**2 + \
                   temporal_mask * beta * (r_l_next - deltas * z + thetas * a_l)**2
        
        energy_theta = beta * (thetas - q_l)**2 + \
                       temporal_mask * beta * (r_l_next - deltas * thetas + thetas * a_l)**2
                       
        delta2 = energy_z - energy_theta

        mask_z_greater = (z > thetas).float()
        mask_deltas1 = (delta1 + delta2 > 0).float()
        mask_intersection1 = mask_z_greater * mask_deltas1
        
        mask_deltas2 = (-delta1 + delta2 > 0).float()
        mask_intersection2 = (1 - mask_z_greater) * mask_deltas2

        z = torch.where(mask_intersection1 == 1, thetas, z)
        z = torch.where(mask_intersection2 == 1, thetas + 1e-5, z)
        return z
    
    def proximal_z_update(self, res, deltas, beta, gamma, z, a, thetas=None, **kwargs):
        """Specialized z-update using ADMM temporal penalties."""
        t_val = thetas if thetas is not None else self.thetas
        q = res.clone()
        q[1:] += deltas * z[:-1] 
        q[1:] -= t_val * a[:-1] 

        r = z - res
        
        numerator = beta * q
        denominator = beta

        temporal_penalty_num = torch.zeros_like(numerator)
        temporal_penalty_den = torch.zeros_like(numerator)
        temporal_penalty_num[:-1] =  deltas * beta * (r[1:] + t_val * a[:-1]) 
        temporal_penalty_den[:-1] = (deltas**2) * beta  

        numerator = numerator + temporal_penalty_num
        denominator = denominator + temporal_penalty_den

        return self.check_entries(numerator / denominator, a, q, r, deltas, beta, gamma, t_val)
    



if __name__ == "__main__":

    seed = 8281003564
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    section=config.get('general', 'section')

    epochs = config.getint(section, 'epochs')
    warming_iters = config.getint(section, 'warming_iters')
    batch_size = config.getint(section, 'batch_size')
    n_timesteps = config.getint(section, 'n_timesteps')
    print(f"Performing training for {batch_size} images...")
    thetas = config.getfloat(section, 'thetas')   
    deltas = config.getfloat(section, 'deltas')


    linear_gamma = config.getfloat(section, 'linear_gamma')  
    linear_beta= config.getfloat(section, 'linear_beta')
    hidden_dims = config.getint(section, 'hidden_dims')
    linear_scale= config.getfloat(section, 'linear_scale')
    linear_scale_last_layer= config.getfloat(section, 'linear_scale_last_layer')

    conv_gamma = config.getfloat(section, 'conv_gamma')  
    conv_beta= config.getfloat(section, 'conv_beta')
    hidden_channels = config.getint(section, 'hidden_channels')
    conv_scale= config.getfloat(section, 'conv_scale')
    conv_scale_last_layer= config.getfloat(section, 'conv_scale_last_layer')

    # Define transformations
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
        ])

    trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
    cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
    train_loader = DataLoader(cached_trainset, batch_size=batch_size,
                                collate_fn=tonic.collation.PadTensors(), shuffle=True, drop_last=True, generator = torch.Generator().manual_seed(seed))


    linear_layers =  nn.ModuleList([
        Spiking_ADMMLinear(in_f=34*34*2, out_f=hidden_dims, scale=linear_scale, h=ADMM_Heaviside(thetas=thetas)),
        Spiking_ADMMLinear(in_f=hidden_dims,  out_f=10, scale=linear_scale_last_layer)
    ])

    conv_layers = nn.ModuleList([
        Spiking_ADMMConv2d(in_c=2, out_c=hidden_channels, k=5, p=2, s=2, 
                           scale=conv_scale, h=ADMM_Heaviside(thetas=thetas)),       
        Spiking_ADMMLinear(in_f=hidden_channels * 17 * 17, out_f=10, 
                           scale=conv_scale_last_layer)
    ])


    model = ADMM(linear_layers, T=n_timesteps,  beta= linear_beta, thetas=thetas, deltas=deltas, gamma=linear_gamma).to(device)
    criterion = nn.MSELoss()
    data, targets = next(iter(train_loader))
    data, targets = data.to(device), targets.to(device)
    data = data.view(data.size(0), data.size(1), -1)
    targets = torch.nn.functional.one_hot(targets.to(torch.long), num_classes=10).float()

    if data.size(1) > n_timesteps:
            data = data[:, :n_timesteps, :]

    data = data.permute(1, 0, 2).float()
    data += 0.01 * torch.randn_like(data) 

    metrics = {}

    metrics_path = f'metrics_test/{batch_size}/linear'
    os.makedirs(metrics_path, exist_ok = True)
    
    print("Linear - Starting Training...")
    lagrangians, lambdas = [], []
    soft_constraints = {"a": [], "z":[]}
    losses = []
    accuracy_list = []
    firing_rate_list = []
    for epoch in range(epochs):
        model.fit(data, targets, warming=epoch < warming_iters) 
                
        if (epoch + 1) % 2 == 0 or epoch == 0: 
            with torch.no_grad():
                raw_outputs, firing_rates = model.forward_model(data)
                flat_outputs = raw_outputs.view(batch_size, -1)
                mse_train = criterion(flat_outputs, targets).item()
                
                _, preds = flat_outputs.max(dim=1)
                acc_train = 100. * (preds == targets.argmax(dim=1)).sum().item() / batch_size
                    
                lagr = model.lagrangian_cost(data, targets)
                primal = model.primal_residual_norm(data)
                preactivation_constraint_sum = model.preactivation_constraint_sum(data)
                activation_constraint_sum = model.activation_constraint_sum()
                
                print(f"Epoch [{epoch+1:3d}/{epochs}] "
                    f"| MSE: {mse_train:.4f} "
                    f"| Acc: {acc_train:6.2f}% "
                    f"| Firing rate: {[f'{v:.4f}' for v in firing_rates]}"
                    f"| Lagr: {lagr:10.2f} "
                    f"| Lamb: {primal:10.2f}"
                    f"| Pre: {[f'{v:.4f}' for v in preactivation_constraint_sum]} "
                    f"| Act: {[f'{v:.4f}' for v in activation_constraint_sum]}") 

                losses.append(mse_train)
                accuracy_list.append(acc_train)
                firing_rate_list.append([f'{v:.4f}' for v in firing_rates])
                soft_constraints["a"].append(activation_constraint_sum)
                soft_constraints["z"].append(preactivation_constraint_sum)
                lagrangians.append(lagr)
                lambdas.append(primal)
    
    metrics["lagrangians"] = lagrangians
    metrics["lambdas"] = lambdas
    metrics["soft_constraints"] = soft_constraints
    metrics["losses"] = losses
    metrics["accuracy_list"] = accuracy_list
    metrics["firing_rate"] = firing_rate_list

    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plots to visualize training metrics
    fig, ax = plt.subplots(2, 3, figsize=(30, 5))
    ax[0, 0].semilogy(lagrangians)
    ax[0, 0].set_title("Lagrangian")
    ax[0, 1].semilogy(lambdas)
    ax[0, 1].set_title("Primal Residual Norm")
    ax[0, 2].semilogy(soft_constraints["a"])
    ax[0, 2].set_title(r"$||a - h(z, \theta)||_2$")
    ax[1, 0].semilogy(soft_constraints["z"])
    ax[1, 0].set_title(r"$||z_1 - \delta*z_{shifted} - W_1a_0 + \theta*a_{1,shifted}||_2$") 
    ax[1, 1].semilogy(losses)
    ax[1, 1].set_title("Loss")
    ax[1, 2].plot(accuracy_list)
    ax[1, 2].set_title("Train Accuracy")
        
    # Run final evaluation on training data
    pot, firing_rate = model.forward_model(data)
    _, predicted = (pot.view(batch_size, -1)).max(1)
    labels = torch.argmax(targets, dim=1)
    print("----------------------------")
    print("Train accuracy:", (predicted == labels).sum().item() / labels.size(0))
    print("----------------------------")
    plt.tight_layout()
    plt.show()

    model = ADMM(conv_layers, T=n_timesteps, beta= conv_beta, thetas=thetas, deltas=deltas, gamma=conv_gamma).to(device)
    criterion = nn.MSELoss()

    data, targets = next(iter(train_loader))
    data, targets = data.to(device), targets.to(device)


    if data.size(1) > n_timesteps:
        data = data[:, :n_timesteps, :, :, :]
    elif data.size(1) < n_timesteps:
        padding = torch.zeros((data.size(0), n_timesteps - data.size(1), *data.shape[2:]), device=device)
        data = torch.cat([data, padding], dim=1)

    data = data.permute(1, 0, 2, 3, 4).float() 
    data += 0.01 * torch.randn_like(data)  
    targets = torch.nn.functional.one_hot(targets.to(torch.long), num_classes=10).float()


    metrics = {}

    metrics_path = f'metrics_test/{batch_size}/conv'
    os.makedirs(metrics_path, exist_ok = True)
    
    print("\nStandard Convolutional - Starting Training...")
    lagrangians, lambdas = [], []
    soft_constraints = {"a": [], "z":[]}
    losses = []
    accuracy_list = []
    firing_rate_list = []
    for epoch in range(epochs):
        model.fit(data, targets, warming=epoch < warming_iters) 
                
        if (epoch + 1) % 2 == 0 or epoch == 0: 
            with torch.no_grad():
                raw_outputs, firing_rates = model.forward_model(data)
                flat_outputs = raw_outputs.view(batch_size, -1)
                mse_train = criterion(flat_outputs, targets).item()
                
                _, preds = flat_outputs.max(dim=1)
                acc_train = 100. * (preds == targets.argmax(dim=1)).sum().item() / batch_size
                    
                lagr = model.lagrangian_cost(data, targets)
                primal = model.primal_residual_norm(data)
                preactivation_constraint_sum = model.preactivation_constraint_sum(data)
                activation_constraint_sum = model.activation_constraint_sum()
                
                print(f"Epoch [{epoch+1:3d}/{epochs}] "
                    f"| MSE: {mse_train:.4f} "
                    f"| Acc: {acc_train:6.2f}% "
                    f"| Firing rate: {[f'{v:.4f}' for v in firing_rates]}"
                    f"| Lagr: {lagr:10.2f} "
                    f"| Lamb: {primal:10.2f}"
                    f"| Pre: {[f'{v:.4f}' for v in preactivation_constraint_sum]} "
                    f"| Act: {[f'{v:.4f}' for v in activation_constraint_sum]}") 
                
                losses.append(mse_train)
                accuracy_list.append(acc_train)
                firing_rate_list.append([f'{v:.4f}' for v in firing_rates])
                soft_constraints["a"].append(activation_constraint_sum)
                soft_constraints["z"].append(preactivation_constraint_sum)
                lagrangians.append(lagr)
                lambdas.append(primal)
      
    metrics["lagrangians"] = lagrangians
    metrics["lambdas"] = lambdas
    metrics["soft_constraints"] = soft_constraints
    metrics["losses"] = losses
    metrics["accuracy_list"] = accuracy_list
    metrics["firing_rate"] = firing_rate_list
    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plots to visualize training metrics
    fig, ax = plt.subplots(2, 3, figsize=(30, 5))
    ax[0, 0].semilogy(lagrangians)
    ax[0, 0].set_title("Lagrangian")
    ax[0, 1].semilogy(lambdas)
    ax[0, 1].set_title("Primal Residual Norm")
    ax[0, 2].semilogy(soft_constraints["a"])
    ax[0, 2].set_title(r"$||a - h(z, \theta)||_2$")
    ax[1, 0].semilogy(soft_constraints["z"])
    ax[1, 0].set_title(r"$||z_1 - \delta*z_{shifted} - W_1a_0 + \theta*a_{1,shifted}||_2$") 
    ax[1, 1].semilogy(losses)
    ax[1, 1].set_title("Loss")
    ax[1, 2].plot(accuracy_list)
    ax[1, 2].set_title("Train Accuracy")
        
    # Run final evaluation on training data
    pot, firing_rate = model.forward_model(data)
    _, predicted = (pot.view(batch_size, -1)).max(1)
    labels = torch.argmax(targets, dim=1)
    print("----------------------------")
    print("Train accuracy:", (predicted == labels).sum().item() / labels.size(0))
    print("----------------------------")
    plt.tight_layout()
    plt.show()