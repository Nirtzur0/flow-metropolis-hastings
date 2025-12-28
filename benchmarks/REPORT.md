# DiffMCMC Benchmark Report

## Overview
This report documents the performance of `diffmcmc` (Diffusion/Flow-Matching Global Proposal MCMC) against a standard Random Walk Metropolis-Hastings (RWMH) baseline.

## Results

### 1. 2D Banana Distribution
**Target**: Banana shaped distribution with high curvature.
- **RWMH Baseline**: 
    - Accept Rate: ~76%
    - ESS: Moderate (mixes slowly across the curve)
- **DiffMCMC**:
    - Global Accept Rate: ~73% (Indicates the Flow learned the density $q \approx \pi$ very well)
    - Efficiency: Can propose large jumps along the banana arc.

### 2. Multi-Modal Gaussian Mixture (MoG)
**Target**: Two Gaussian modes ($N(\mu, I)$) separated by $10\sigma$ (centers at $-5$ and $+5$).
**Test**: Start sampler at $-5$ and run for 2000 steps.

| Metric | RWMH | DiffMCMC |
| :--- | :--- | :--- |
| Jump Success | **Fail** (Stuck in mode 1) | **Success** (Modes mixed) |
| Mode Coverage | 0% in mode 2 | ~44% in mode 2 (Ideal: 50%) |
| Global Accept | N/A | 70% |

**Conclusion**: DiffMCMC successfully performs mode jumping on disconnected posteriors where local samplers fail, provided the global proposal is trained on a representative buffer (Stage A/B).

## Methodology
- **Flow Model**: A 3-layer MLP velocity field with Rectified Flow training.
- **Density Estimation**: Hutchinson Trace Estimator for $\nabla \cdot v$.
- **Training**: 100 epochs on synthetic "warmup" samples.
- **Sampling**: 20% Global Moves ($p_{global}=0.2$), 80% Local RWM ($scale=0.5$).

## Reproducibility
To reproduce these results:
```bash
python3 examples/demo_mog.py
python3 benchmarks/run_benchmarks.py
```
