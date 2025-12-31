import torch
import numpy as np
import time
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.targets.mog import GaussianMixtureTarget

def run_benchmark():
    # Setup
    dim = 2
    # Two modes at (-4, -4) and (4, 4)
    centers = [[-4.0, -4.0], [4.0, 4.0]]
    target = GaussianMixtureTarget(dim=dim, centers=centers, scale=1.0)
    
    print(f"Target: 2D Mixture of Gaussians at {centers}")
    
    # Delayed Acceptance Flow Sampler
    # Increase mixture to 0.5 to ensure frequent global jumps for demonstration
    flow = FlowProposal(dim=dim, step_size=0.1, deterministic_trace=True, mixture_prob=0.5, broad_scale=5.0)
    kernel = RWMKernel(scale=0.5)
    
    sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=kernel,
        global_proposal=flow,
        p_global=0.2, # 20% global moves
        device="cpu"
    )
    
    num_steps = 2000
    print(f"Running DiffMCMC for {num_steps} steps...")
    
    start_t = time.time()
    chain, stats = sampler.run(torch.zeros(dim), num_steps=num_steps, warmup=0, seed=42)
    end_t = time.time()
    
    print("-" * 50)
    print("Benchmark Results:")
    print(f"Total Time: {stats['total_time_sec']:.4f} s")
    print(f"ESS / sec: {stats.get('ess_per_sec', 0):.4f}")
    print(f"Min ESS: {stats.get('ess_min', 0):.2f}")
    print(f"Global Accept Rate Stage 1: {stats['accept_global_stage1'] / max(1, stats['attempts_global']):.4f}")
    print(f"Global Accept Rate Total: {stats['accept_global'] / max(1, stats['attempts_global']):.4f}")
    print(f"Global Attempts: {stats['attempts_global']}")
    print("-" * 50)
    
    # Mode Occupancy
    # Mode 1: x < 0, Mode 2: x > 0 (roughly)
    # Check 1st dimension
    samples_d0 = chain[:, 0]
    in_mode1 = np.mean(samples_d0 < 0)
    in_mode2 = np.mean(samples_d0 > 0)
    
    print(f"Mode Occupancy: Mode 1 (<0): {in_mode1:.2%}, Mode 2 (>0): {in_mode2:.2%}")
    
    is_balanced = 0.4 < in_mode1 < 0.6
    print(f"Balanced? {is_balanced}")
    
    # Check Variance
    # True variance should be approx pvar + between-mode var
    # var = 1^2 + (4 - (-4))^2 / 4 = 1 + 16 = 17? 
    # Approx E[x^2] - E[x]^2. E[x]=0.
    # E[x^2] = 0.5(Var1 + mu1^2) + 0.5(Var2 + mu2^2) = 0.5(1+16) + 0.5(1+16) = 17.
    
    sample_var = np.var(chain, axis=0)
    print(f"Sample Variance: {sample_var}")
    print(f"Expected Variance: ~17.0")
    
    if is_balanced and stats["attempts_global"] > 0:
        print("SUCCESS: Sampler explored modes and global moves were attempted.")
    else:
        print("WARNING: Sampler might be stuck or failed to explore.")

if __name__ == "__main__":
    run_benchmark()
