import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel, MALAKernel
from diffmcmc.proposal.flow import FlowProposal
from diffmcmc.proposal.training import train_flow_matching
from diffmcmc.targets.banana import BananaTarget
from diffmcmc.targets.mog import GaussianMixtureTarget

def run_benchmark(target, dim, steps=5000, warmup=1000, seed=42):
    print(f"--- Benchmarking Target: {target.__class__.__name__} (Dim: {dim}) ---")
    
    results = []
    
    # Baseline: RWMH
    print("Running RWMH Baseline...")
    t0 = time.time()
    rwm_sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5 if dim < 10 else 0.1),
        p_global=0.0
    )
    samples_rwm, stats_rwm = rwm_sampler.run(torch.zeros(dim), steps, warmup, seed)
    t_rwm = time.time() - t0
    
    # Calculate ESS (using basic variance method or arviz if available, here simple approx)
    # Actually just log raw perf.
    results.append({
        "Method": "RWMH",
        "Time": t_rwm,
        "AcceptRate": stats_rwm.get('accept_local', 0) / (steps + warmup),
        "Samples": samples_rwm
    })
    
    # DiffMCMC
    print("Running DiffusionMH...")
    
    # 1. Warmup / Pre-training
    # We collect some samples with RWMH first (or just reuse rwm samples for training)
    # Let's effectively use the RWM samples as the "Stage A" buffer.
    print("Training Flow...")
    bootstrap_samples = samples_rwm[::10] # Subsample
    if len(bootstrap_samples) > 2000:
        bootstrap_samples = bootstrap_samples[-2000:]
        
    flow = FlowProposal(dim)
    train_flow_matching(flow, bootstrap_samples, epochs=50, verbose=False)
    
    # 2. Run Sampler
    t0 = time.time()
    diff_sampler = DiffusionMH(
        log_prob_fn=target.log_prob,
        dim=dim,
        local_kernel=RWMKernel(scale=0.5 if dim < 10 else 0.1),
        global_proposal=flow,
        p_global=0.2 # 20% global
    )
    # Warmup again to mix chain?
    samples_diff, stats_diff = diff_sampler.run(torch.zeros(dim), steps, warmup, seed)
    t_diff = time.time() - t0
    
    results.append({
        "Method": "DiffMCMC",
        "Time": t_diff,
        "AcceptRateLocal": stats_diff.get('accept_local', 0) / (stats_diff.get('attempts_local', 1)),
        "AcceptRateGlobal": stats_diff.get('accept_global', 0) / (stats_diff.get('attempts_global', 1)),
        "Samples": samples_diff
    })
    
    return results

def main():
    # Targets
    targets = [
        (BananaTarget(dim=2, b=0.1), 2),
        (GaussianMixtureTarget(dim=2, scale=1.0), 2)
    ]
    
    for target, dim in targets:
        res = run_benchmark(target, dim)
        for r in res:
            # Print brief stats
            s = r["Samples"]
            mean = np.mean(s, axis=0)
            cov = np.cov(s.T)
            print(f"Method: {r['Method']}")
            print(f"  Time: {r['Time']:.2f}s")
            if "AcceptRate" in r:
                print(f"  Accept Rate: {r['AcceptRate']:.2f}")
            else:
                print(f"  Global Accept: {r['AcceptRateGlobal']:.2f}")
                print(f"  Local Accept: {r['AcceptRateLocal']:.2f}")
            print(f"  Mean: {mean}")
            
if __name__ == "__main__":
    main()
