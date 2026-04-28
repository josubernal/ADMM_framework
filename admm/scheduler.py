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
        self.a_old = [layer.a.detach().clone() for layer in self.model.layers if hasattr(layer, 'a')]

    def step(self, primal_rho: float, primal_beta: float):
        """Called AFTER the epoch ends to compute duals and balance penalties."""
        if self.z_old is None or self.a_old is None:
            print("Warning: capture_state() must be called before step(). Skipping balance.")
            return

        # 1. Compute Dual Residuals on the fly
        z_new = [layer.z.detach() for layer in self.model.layers if hasattr(layer, 'z')]
        a_new = [layer.a.detach() for layer in self.model.layers if hasattr(layer, 'a')]
        
        dual_rho = 0.0
        for zn, zo in zip(z_new, self.z_old):
            dual_rho += self.model.rho * torch.norm(zn - zo).item()
            
        dual_beta = 0.0
        for an, ao in zip(a_new, self.a_old):
            dual_beta += self.model.beta * torch.norm(an - ao).item()

        # 2. Balance RHO (Pre-activations)
        if primal_rho > self.mu * dual_rho:
            self.model.rho *= self.tau         
            self._scale_lambda_lagrage(1.0 / self.tau)
            
        elif dual_rho > self.mu * primal_rho:
            self.model.rho /= self.tau         
            self._scale_lambda_lagrage(self.tau)
            
        # 3. Balance BETA (Activations)
        if primal_beta > self.mu * dual_beta:
            self.model.beta *= self.tau          
          
        elif dual_beta > self.mu * primal_beta:
            self.model.beta /= self.tau         
            
        self.z_old = None
        self.a_old = None
    
    def _scale_lambda_lagrage(self, scale_factor):
        self.model.lambda_lagrange.mul_(scale_factor)
