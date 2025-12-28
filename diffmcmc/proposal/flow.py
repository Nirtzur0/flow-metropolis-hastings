import torch
import torch.nn as nn
import hashlib
import numpy as np
from diffmcmc.proposal.nets import VelocityMLP

class FlowProposal(nn.Module):
    r"""
    Continuous Normalizing Flow Proposal defined by a velocity field v(x, t).
    
    Generative (sampling):
       z ~ N(0, I)
       Solve dx/dt = v(x, t) from t=0 to 1 -> x
       
    Density (log q(x)):
       Solve backward dx/dt = v(x, t) from t=1 to 0 -> z
       log q(x) = log p(z) - \int_0^1 div(v) dt
       
    Rigorous Exactness:
       To ensure detailed balance in MH, the density estimate q(x) must be a deterministic function of x.
       We achieve this by hashing x to seed the Hutchinson noise \epsilon.
       This implies we are targeting a slightly perturbed posterior, but the chain is strictly reversible.
    """
    def __init__(self, dim, model=None, step_size=0.1, deterministic_trace=True):
        """
        Args:
            dim: Dimension of data.
            model: Velocity network.
            step_size: Integration step size.
            deterministic_trace: If True, uses hashed noise for density estimation.
        """
        super().__init__()
        self.dim = dim
        self.step_size = step_size
        self.deterministic_trace = deterministic_trace
        if model is None:
            self.net = VelocityMLP(dim)
        else:
            self.net = model
            
    def _ode_step(self, x, t, dt):
        # RK4 step
        k1 = self.net(x, t)
        k2 = self.net(x + dt * 0.5 * k1, t + dt * 0.5)
        k3 = self.net(x + dt * 0.5 * k2, t + dt * 0.5)
        k4 = self.net(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def sample(self, num_samples):
        device = next(self.net.parameters()).device
        z = torch.randn(num_samples, self.dim, device=device)
        
        # Integrate 0 -> 1
        x = z
        t = 0.0
        steps = int(1.0 / self.step_size)
        
        with torch.no_grad():
            for _ in range(steps):
                x = self._ode_step(x, t, self.step_size)
                t += self.step_size
                
        return x
        
    def _get_hutchinson_noise(self, x):
        """
        Generate noise epsilon. 
        If deterministic_trace is True, seed rng with hash(x).
        """
        if not self.deterministic_trace:
            return torch.randn_like(x)
            
        # Hash x to get a seed
        # Note: This is slow for large batches/dims, but ensures correctness.
        # We process item by item or try a vectorized hash?
        # For MVP, per-item hash or a simple sum-based seed for speed (but risk of collision).
        # Robust: torch.tensor -> bytes -> hash -> int -> seed
        
        # Optimization: Just use a fixed noise vector? 
        # Theorem 1 says "fixing the noise... per state".
        # If we use a global fixed noise vector for ALL states, that satisfies the condition too!
        # It just means the estimator function is \hat{q}(x) = f(x, \epsilon_{global}).
        # This is a valid density.
        # This is much faster and simpler than hashing!
        # Wait, does Hutchinson unbiasedness rely on eps being random PER EVAL?
        # E[trace] = \nabla \cdot v.
        # If we fix eps globally, \hat{div} is biased for a specific x?
        # No, \hat{div}(x) is a function. 
        # But we need \hat{q} to approximate q.
        # If we fix \epsilon, for some x, \hat{div} might be hugely off.
        # Ideally \epsilon should be "random-looking" wrt v(x).
        # Hashing provides that. Global fixed noise might correlate with structure.
        
        # Hashing Strategy:
        noise_list = []
        x_np = x.detach().cpu().numpy()
        for i in range(x.shape[0]):
            # Robust hash
            x_bytes = x_np[i].tobytes()
            hash_obj = hashlib.sha256(x_bytes)
            seed_int = int.from_bytes(hash_obj.digest()[:4], 'big')
            
            # Local RNG
            rng = np.random.RandomState(seed_int)
            eps = rng.randn(self.dim)
            noise_list.append(eps)
            
        return torch.tensor(np.array(noise_list), dtype=x.dtype, device=x.device)

    def log_prob(self, x):
        """
        Compute log q(x) using Hutchinson trace estimator.
        Integration operates backwards from 1 -> 0.
        """
        device = next(self.net.parameters()).device
        batch_size = x.shape[0]
        
        # Current state
        xt = x.clone()
        zero = torch.zeros(batch_size, 1, device=device)
        log_jac_trace = zero.clone() # Accumulate integral of div
        
        # Time steps for backward integration (1 -> 0)
        # dt is negative
        dt = -self.step_size
        steps = int(1.0 / self.step_size)
        t = 1.0
        
        # Trace estimator noise
        # Generate ONCE per density evaluation? 
        # Or once per step?
        # Standard Hutchinson is per-matrix. Here we have a trajectory.
        # If we want detailed balance, the whole \hat{L}(x) must be a fixed function of x.
        # So we must fix the noise sequence.
        # Easiest: Fix ONE noise vector and use it for all time steps?
        # Or seed the sequence based on x.
        
        # Let's use ONE noise vector for the whole trajectory (common approx).
        noise = self._get_hutchinson_noise(xt)
        
        for _ in range(steps):
            # Euler Step for Density
            
            xt.requires_grad_(True)
            v = self.net(xt, t)
            
            def func_v(inputs):
                # Fix t
                return self.net(inputs, t)
                
            # Compute div estimate: eps^T * (J*eps)
            _, jvp_val = torch.autograd.functional.jvp(func_v, xt, v=noise)
            trace_est = torch.sum(noise * jvp_val, dim=1, keepdim=True)
            
            # Update integral: \int_0^1 div(v) dt
            log_jac_trace += trace_est * self.step_size 
            
            # Update state (Euler)
            with torch.no_grad():
                xt = xt + v * dt # dt is negative
            
            t += dt
            
        # Final z is xt
        log_prob_z = -0.5 * torch.sum(xt**2, dim=1) - 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi))
        
        # Result
        return log_prob_z - log_jac_trace.squeeze(1)
