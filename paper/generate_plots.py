import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.proposal.training import train_flow_matching
from diffmcmc.targets.banana import BananaTarget
from diffmcmc.targets.mog import GaussianMixtureTarget

def generate_banana_plots():
    print("Generating Banana Plots...")
    dim = 2
    target = BananaTarget(dim=dim, b=0.1)
    
    # 1. RWMH Baseline
    rwm = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5),
        p_global=0.0
    )
    samples_rwm, _ = rwm.run(torch.zeros(dim), 2000, warmup=500, seed=42)
    
    # 2. DiffMCMC
    # Train
    rwm_pre = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5),
        p_global=0.0
    )
    # Collect some data to train
    pre_samples, _ = rwm_pre.run(torch.zeros(dim), 2000, warmup=1000, seed=100)
    samples_train = pre_samples[::5] # subsample
    
    flow = FlowProposal(dim)
    train_flow_matching(flow, samples_train, epochs=50, verbose=False)
    
    diff = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5), # Standard RWM
        global_proposal=flow,
        p_global=0.2
    )
    samples_diff, _ = diff.run(torch.zeros(dim), 2000, warmup=500, seed=42)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].scatter(samples_rwm[:, 0], samples_rwm[:, 1], s=5, alpha=0.5, label='RWMH')
    axes[0].set_title("RWMH (Banana)")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 4)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(samples_diff[:, 0], samples_diff[:, 1], s=5, alpha=0.5, color='orange', label='DiffMCMC')
    axes[1].set_title("DiffMCMC (Banana)")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 4)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper/figures/banana.png', dpi=300)
    plt.savefig('paper/figures/banana.pdf')
    plt.close()

def generate_mog_plots():
    print("Generating MoG Plots...")
    dim = 2
    target = GaussianMixtureTarget(dim=dim) # modes at -5, 5
    
    # Generate Oracle Training Data (simulating successful exploration)
    x1 = torch.randn(1000, 2) + 5.0
    x2 = torch.randn(1000, 2) - 5.0
    training_data = torch.cat([x1, x2], dim=0)
    
    flow = FlowProposal(dim)
    train_flow_matching(flow, training_data, epochs=80, verbose=False)
    
    # RWMH
    rwm = DiffusionMH(target.log_prob, dim, RWMKernel(0.5), p_global=0.0)
    s_rwm, _ = rwm.run(torch.tensor([-5.0, -5.0]), 2000, seed=42)
    
    # DiffMCMC
    diff = DiffusionMH(target.log_prob, dim, RWMKernel(0.5), flow, p_global=0.2)
    s_diff, _ = diff.run(torch.tensor([-5.0, -5.0]), 2000, seed=42)
    
    # Plot Trace of Dimension 0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(s_rwm[:, 0], lw=1, alpha=0.8)
    axes[0].set_title("Trace Plot (RWMH): Stuck in one mode")
    axes[0].set_ylabel("x[0]")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(s_diff[:, 0], lw=1, alpha=0.8, color='orange')
    axes[1].set_title("Trace Plot (DiffMCMC): Frequent Mode Jumping")
    axes[1].set_ylabel("x[0]")
    axes[1].set_xlabel("MCMC Step")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper/figures/mog_trace.png', dpi=300)
    plt.savefig('paper/figures/mog_trace.pdf')
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('paper/figures'):
        os.makedirs('paper/figures')
    generate_banana_plots()
    generate_mog_plots()
