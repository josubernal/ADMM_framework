import torch

class ADMM_Scheduler:
    """
    Dynamically balances Primal and Dual residuals by scaling rho and beta.
    """
    def __init__(self, model, mu: float = 10.0, tau: float = 2.0):
        self.model = model
        self.mu = mu
        self.tau = tau
        
        # State placeholders
        self.z_old = None
        self.a_old = None

    def capture_state(self):
        """Called BEFORE the epoch starts to snapshot the network state."""
        self.z_old = [layer.z.detach().clone() for layer in self.model.layers if hasattr(layer, 'z')]
        self.a_old = [layer.a.detach().clone() for layer in self.model.layers[:-1] if hasattr(layer, 'a')]

    def step(self, primal_rho: list, primal_beta: list):
        """Called AFTER the epoch ends to compute duals and balance penalties."""
        if self.z_old is None or self.a_old is None:
            print("Warning: capture_state() must be called before step(). Skipping balance.")
            return

        z_new = [layer.z.detach() for layer in self.model.layers if hasattr(layer, 'z')]
        a_new = [layer.a.detach() for layer in self.model.layers[:-1] if hasattr(layer, 'a')]
                
        for i, (layer, zn, zo, p_rho) in enumerate(zip(self.model.layers, z_new, self.z_old, primal_rho)):
            norm_factor = zn.numel() ** 0.5
            d_rho = layer.rho * (torch.norm(zn - zo).item() / norm_factor)
            
            if p_rho > self.mu * d_rho:
                layer.rho *= self.tau
                if i == len(self.model.layers) - 1 and hasattr(self.model, 'lambda_lagrange'):
                    self.model.lambda_lagrange /= self.tau
                    
            elif d_rho > self.mu * p_rho:
                layer.rho /= self.tau
                if i == len(self.model.layers) - 1 and hasattr(self.model, 'lambda_lagrange'):
                    self.model.lambda_lagrange *= self.tau
  
        for layer, an, ao, p_beta in zip(self.model.layers[:-1], a_new, self.a_old, primal_beta):
            norm_factor = an.numel() ** 0.5
            d_beta = layer.beta * (torch.norm(an - ao).item() / norm_factor)
            
            if p_beta > self.mu * d_beta:
                layer.beta *= self.tau
            elif d_beta > self.mu * p_beta:
                layer.beta /= self.tau

        self.z_old = None
        self.a_old = None
    
