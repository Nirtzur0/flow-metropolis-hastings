import torch
import numpy as np

class FunnelTarget:
    """
    Neal's Funnel.
    y ~ N(0, 3)
    x_i ~ N(0, exp(y/2)) for i in 1..D-1
    """
    def __init__(self, dim=2):
        self.dim = dim
        
    def log_prob(self, x):
        # x: (..., D)
        # Convention: x[0] is y, x[1:] are the other components.
        
        if x.dim() == 1:
            x = x.unsqueeze(0) # (1, D)
            
        y = x[:, 0]
        others = x[:, 1:]
        
        # y ~ N(0, 3^2) -> log_p(y) = -0.5 * y^2 / 9 - log(sqrt(2pi)*3)
        log_prob_y = -0.5 * (y**2) / 9.0 - np.log(3 * np.sqrt(2 * np.pi))
        
        # others ~ N(0, exp(y))
        # variance = exp(y)
        # log_p(x|y) = -0.5 * x^2 / exp(y) - 0.5 * log(2pi * exp(y))
        #            = -0.5 * x^2 * exp(-y) - 0.5 * (log(2pi) + y)
        
        variance = torch.exp(y).unsqueeze(1) # (B, 1)
        log_prob_others = -0.5 * (others**2) / variance - 0.5 * (np.log(2 * np.pi) + y.unsqueeze(1))
        
        return log_prob_y + torch.sum(log_prob_others, dim=1)
