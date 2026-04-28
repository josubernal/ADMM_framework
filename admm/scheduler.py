import torch

class ADMM_Scheduler:
    """
    Dynamically balances Primal and Dual residuals by scaling rho and beta.
    Upgraded for Deep Learning: Uses frequency control and a stop-epoch to 
    prevent 'Ping-Pong' oscillation in non-convex landscapes.
    """
    def __init__(self, model, mu: float = 10.0, tau: float = 2.0, balance_freq: int = 5, stop_epoch: int = 150):
        self.model = model
        self.mu = mu
        self.tau = tau
        
        # Deep Learning Controls
        self.balance_freq = balance_freq
        self.stop_epoch = stop_epoch
        self.current_epoch = 0
        
        # State placeholders
        self.z_old = None
        self.a_old = None

    def capture_state(self):
        """Called BEFORE the epoch starts to snapshot the network state."""
        self.z_old = [layer.z.detach().clone() for layer in self.model.layers if hasattr(layer, 'z')]
        self.a_old = [layer.a.detach().clone() for layer in self.model.layers[:-1] if hasattr(layer, 'a')]

    def step(self, primal_rho_list: list, primal_beta_list: list):
        """Called AFTER the epoch ends to compute duals and balance penalties."""
        self.current_epoch += 1
        
        if self.z_old is None or self.a_old is None:
            print("Warning: capture_state() must be called before step(). Skipping balance.")
            return

        # =========================================================
        # Deep Learning Control: Only balance every N epochs, and 
        # stop balancing near the end of training to allow fine-tuning!
        # =========================================================
        if (self.current_epoch % self.balance_freq != 0) or (self.current_epoch > self.stop_epoch):
            self.z_old = None
            self.a_old = None
            return

        # 1. Compute NORMALIZED Dual Residuals on the fly
        z_new = [layer.z.detach() for layer in self.model.layers if hasattr(layer, 'z')]
        a_new = [layer.a.detach() for layer in self.model.layers[:-1] if hasattr(layer, 'a')]
        
        # ---------------------------------------------------------
        # 2. Balance RHO per layer (Applied to ALL layers)
        # ---------------------------------------------------------
        for layer, zn, zo, p_rho in zip(self.model.layers, z_new, self.z_old, primal_rho_list):
            norm_factor = zn.numel() ** 0.5
            d_rho = layer.rho * (torch.norm(zn - zo).item() / norm_factor)
            
            if p_rho > self.mu * d_rho:
                layer.rho *= self.tau
                
                # --- NEW: Safe Layer-wise Lagrange Scaling ---
                if getattr(layer, 'use_lagrange', False) and getattr(layer, 'lambda_lagrange', None) is not None:
                    layer.lambda_lagrange /= self.tau
                    
            elif d_rho > self.mu * p_rho:
                layer.rho /= self.tau
                
                # --- NEW: Safe Layer-wise Lagrange Scaling ---
                if getattr(layer, 'use_lagrange', False) and getattr(layer, 'lambda_lagrange', None) is not None:
                    layer.lambda_lagrange *= self.tau

        # ---------------------------------------------------------
        # 3. Balance BETA per layer (Applied to HIDDEN layers only)
        # ---------------------------------------------------------
        for layer, an, ao, p_beta in zip(self.model.layers[:-1], a_new, self.a_old, primal_beta_list):
            norm_factor = an.numel() ** 0.5
            d_beta = layer.beta * (torch.norm(an - ao).item() / norm_factor)
            
            if p_beta > self.mu * d_beta:
                layer.beta *= self.tau
            elif d_beta > self.mu * p_beta:
                layer.beta /= self.tau
                
        # Free memory immediately
        self.z_old = None
        self.a_old = None