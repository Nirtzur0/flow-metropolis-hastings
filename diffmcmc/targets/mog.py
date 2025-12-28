import torch
import numpy as np

class GaussianMixtureTarget:
    """
    Mixture of Gaussians.
    """
    def __init__(self, dim=2, centers=None, scale=1.0):
        self.dim = dim
        if centers is None:
            # Default: 2 widely separated modes
            self.centers = torch.tensor([[-5.0] * dim, [5.0] * dim])
        else:
            self.centers = torch.tensor(centers)
            
        self.scale = scale
        self.n_modes = self.centers.shape[0]
        self.log_weight = -np.log(self.n_modes) # Uniform weights
        
    def log_prob(self, x):
        # x: (..., D)
        if x.dim() == 1:
             x = x.unsqueeze(0)
        
        # p(x) = sum_k w_k N(x | mu_k, sigma^2 I)
        # log p(x) = logsumexp( log w_k + log N(...) )
        
        # (B, 1, D) - (1, K, D) -> (B, K, D)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0).to(x.device)
        
        # Squared distance: (B, K)
        sq_dist = torch.sum(diff**2, dim=2)
        
        # log N = -0.5 * sq_dist / scale^2 - C
        const = -0.5 * self.dim * np.log(2 * np.pi) - self.dim * np.log(self.scale)
        log_n = -0.5 * sq_dist / (self.scale**2) + const
        
        # log weight + log N
        log_terms = self.log_weight + log_n
        
        return torch.logsumexp(log_terms, dim=1)
