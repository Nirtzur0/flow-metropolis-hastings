import torch
import numpy as np

class BananaTarget:
    """
    Rosenbrock-like / Banana distribution.
    log_prob = -sum( (x_{i+1} - (x_i^2 - 1))^2 + x_i^2 ) ?
    Or the simple 2D banana:
    y = x2 - b(x1^2 - a)
    """
    def __init__(self, dim=2, b=0.1):
        self.dim = dim
        self.b = b
        
    def log_prob(self, x):
        # x: (..., D)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Simple 2D Banana shape:
        # P(x1, x2) propto exp( -0.5 * (x1^2/sigma1 + (x2 - b(x1^2 - a))^2 / sigma2) )
        # Standard formulation:
        # warp N(0, I):
        # x1 = z1
        # x2 = z2 + b(z1^2 - a)
        
        # Here lets just implement a known form:
        # -0.5 * (x1^2 + (x2 - b*x1^2 + 100*b)^2) # rough example
        
        # Let's use the standard "warped gaussian" banana
        # z ~ N(0, I)
        # x = (z1, z2 + b*z1^2 - b)
        # We need log_prob(x).
        # Inverse:
        # z1 = x1
        # z2 = x2 - b*x1^2 + b
        # Jacobian is triangular with diagonal 1, so log_det = 0.
        # log_prob(x) = log_prob_z(z(x))
        
        # Dimensions: 
        # Even indices: x[2i]
        # Odd indices: x[2i+1]
        
        # Let's stick to 2D for simplicity in logic, or pairs.
        # Assume D=2 for canonical banana.
        
        z1 = x[:, 0]
        z2 = x[:, 1]
        
        # Warp back to Gaussian
        # z2_gauss = x2 - b * (x1^2)
        # Common formulation: x2 = z2 + b * z1^2  (approx)
        
        term1 = z1
        term2 = z2 - self.b * (z1**2)
        
        # Let's assume unit variance for z1, z2
        lp = -0.5 * (term1**2 + term2**2) - np.log(2 * np.pi)
        
        return lp
