import torch
import numpy as np
from diffmcmc.physics.optics import TransferMatrixMethod

class ThinFilmTarget:
    """
    Inverse Design / Inference for Thin Film Multilayer Stacks.
    
    Target:
        Given an observed Reflectance Spectrum R_obs(\lambda),
        infer the layer thicknesses d = [d_1, ..., d_N].
        
    Likelihood:
        Gaussian noise on spectrum:
        log P(R_obs | d) = -0.5 * \sum ((R_pred(d) - R_obs) / sigma)^2
    """
    def __init__(self, 
                 observed_spectrum: torch.Tensor, 
                 wavelengths: torch.Tensor, 
                 n_pattern: torch.Tensor,
                 sigma: float = 0.01):
        """
        Args:
            observed_spectrum: (W,) Observed reflectance.
            wavelengths: (W,) Wavelength grid.
            n_pattern: (L,) Refractive indices of the stack [Amb, L1...LN, Sub].
            sigma: Assumed noise standard deviation.
        """
        self.obs = observed_spectrum
        self.wavelengths = wavelengths
        # n_pattern includes ambient and substrate.
        # We assume n_pattern is fixed (known materials).
        # We infer thicknesses of the internal layers (indices 1 to L-2).
        self.n = n_pattern
        self.sigma = sigma
        
        self.solver = TransferMatrixMethod()
        
    def log_prob(self, params):
        """
        params: (Batch, N_films) Thicknesses in nm.
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
            
        batch_size = params.shape[0]
        device = params.device
        
        # Constrain Physical Thickness > 0
        # If parameters are unconstrained (e.g. from Gaussian proposal), 
        # we should use log-thicknesses or softplus.
        # DiffMCMC works in unconstrained space usually.
        # Let's assume the sampler provides "raw" params, we Softplus them?
        # Or just hard reject negative? Hard reject breaks gradients for HMC/Flow?
        # Flow training sees valid positive samples. 
        # RWMH might propose negative.
        # Let's use Softplus transform inside log_prob to ensure physical validity implicitly?
        # Or better: Assume input 'params' implies Log-Thicknesses?
        # Let's stick to LIN-SPACE thickness but return -inf if negative.
        # But for Flow matching, we want smooth density.
        
        # Strategy: DiffMCMC samples in Unconstrained support. 
        # Map u -> d = Softplus(u) * scale?
        # Or just handle params as d directly and penalize < 0.
        
        # Let's assume params ARE thicknesses for now, and handle <0 with penalty.
        # But RWMH random walk will drift to negative easily.
        # BETTER: Input is Log-Thickness.
        
        d_val = torch.exp(params) # (B, N_films)
        
        # Construct full d_stack (B, L)
        # Pad with 0 for ambient/sub
        d_amb = torch.zeros(batch_size, 1, device=device)
        d_sub = torch.zeros(batch_size, 1, device=device)
        d_stack = torch.cat([d_amb, d_val, d_sub], dim=1)
        
        # n_stack (B, L)
        n_stack = self.n.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Prior
        # Weak prior on thicknesses? d ~ Uniform(0, 500)?
        # LogNormal prior on d => Normal on log(d).
        # log d ~ N(log(150), 1^2)?
        mean_log_d = np.log(150.0)
        prior_lp = -0.5 * torch.sum((params - mean_log_d)**2, dim=1)
        
        # Likelihood
        # Solve TMM
        # Wavelengths need to be on device
        waves = self.wavelengths.to(device)
        
        try:
            R_pred = self.solver(n_stack, d_stack, waves) # (B, W)
        except Exception:
             return torch.ones(batch_size, device=device) * -1e9
             
        # Chi2
        obs = self.obs.to(device)
        chi2 = torch.sum(((R_pred - obs) / self.sigma)**2, dim=1)
        
        return prior_lp - 0.5 * chi2
