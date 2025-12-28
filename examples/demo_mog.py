import torch
import numpy as np
import time
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.proposal.training import train_flow_matching
from diffmcmc.targets.mog import GaussianMixtureTarget

def run_mode_jumping_demo():
    print("--- Mode Jumping Demo (MoG) ---")
    dim = 2
    target = GaussianMixtureTarget(dim=2) # Modes at -5, +5
    
    # 1. Oracle / Multi-chain Training Data Generation
    # Simulate collecting data from both modes (e.g. running 2 chains starting at -5 and 5)
    print("Generating training data from both modes...")
    x1 = torch.randn(1000, 2) + 5.0
    x2 = torch.randn(1000, 2) - 5.0
    training_data = torch.cat([x1, x2], dim=0)
    
    # 2. Train Flow
    print("Training Flow...")
    flow = FlowProposal(dim)
    # Train robustly
    train_flow_matching(flow, training_data, epochs=100, batch_size=256, verbose=False)
    
    # 3. DiffusionMH Run (Single Chain from -5)
    print("Running DiffusionMH (Start @ -5)...")
    sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5), # Standard RWM
        global_proposal=flow,
        p_global=0.2
    )
    
    # Run for enough steps
    samples, stats = sampler.run(
        initial_x=torch.tensor([-5.0, -5.0]), 
        num_steps=2000, 
        warmup=0,
        seed=123
    )
    
    # Check coverage
    mean = np.mean(samples, axis=0)
    print(f"Mean: {mean}")
    
    # Prop of samples in +5 region (x > 0)
    prop_pos = np.mean(samples[:, 0] > 0)
    print(f"Proportion in + Mode: {prop_pos:.2f}")
    
    if 0.3 < prop_pos < 0.7:
        print("SUCCESS: Mode jumping achieved (approx 50/50 coverage)!")
    else:
        print("WARNING: Mode jumping might be poor or unbalanced.")
        
    print(f"Global Accept Rate: {stats.get('accept_global',0)/stats.get('attempts_global',1):.2f}")

    # Baseline RWMH for comparison
    print("\nRunning RWMH Baseline (Start @ -5)...")
    rwm = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5),
        p_global=0.0
    )
    samples_rwm, _ = rwm.run(torch.tensor([-5.0, -5.0]), 2000)
    prop_pos_rwm = np.mean(samples_rwm[:, 0] > 0)
    print(f"RWMH Proportion in + Mode: {prop_pos_rwm:.2f}")

if __name__ == "__main__":
    run_mode_jumping_demo()
