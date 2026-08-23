import math
import torch
from torch.optim import Optimizer

class StochasticFokkerPlanckDescent(Optimizer):
    """
    Implements Stochastic Fokker-Planck Descent (SFPD) based on the SCCI-2026 Framework.
    Formalized by Mohamed Nour Kayad (MNK), August 2026.
    """
    def __init__(self, params, lr=1e-3, sigma=1e-4, det_threshold=1e-5, reinjection_rate=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if sigma < 0.0:
            raise ValueError(f"Invalid noise amplitude (sigma): {sigma}")
            
        defaults = dict(
            lr=lr, 
            sigma=sigma, 
            det_threshold=det_threshold, 
            reinjection_rate=reinjection_rate
        )
        super(StochasticFokkerPlanckDescent, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            sigma = group['sigma']
            det_threshold = group['det_threshold']
            reinjection_rate = group['reinjection_rate']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                f_i = -p.grad.data
                grad_norm = torch.norm(f_i)
                noise = torch.randn_like(p.data) * sigma * math.sqrt(lr)
                
                if grad_norm < det_threshold:
                    reinjection_vector = torch.randn_like(p.data) * reinjection_rate
                    p.data.add_(reinjection_vector)
                else:
                    p.data.add_(f_i, alpha=lr)
                    p.data.add_(noise)

        return loss
