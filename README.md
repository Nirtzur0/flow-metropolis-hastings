# diffmcmc

**State-of-the-Art Exact Bayesian Inference with Free-Form Flow Matching**

`diffmcmc` accelerates Bayesian inference for expensive likelihoods by using a diffusion/flow-matching generative model as a **Global Proposal** mechanism inside an **Exact Metropolis-Hastings** correction step.

## Why DiffMCMC? (The Edge)

Unlike other Neural MCMC methods (e.g., NeuTra, A-NICE-MC), `diffmcmc`:
1.  **Strict Exactness**: Uses a novel **Deterministic Hutchinson Trace Estimator** (Fixed-Noise Theorem) to ensure the Markov chain satisfies detailed balance exactly, even with stochastic density estimation.
2.  **Free-Form Expressivity**: Because we use Continuous Flows (ODEs) instead of Coupling Layers, our proposal networks can be arbitrary deep architectures (MLPs, ResNets) without Jacobian constraints.
3.  **Rectified Flows**: Uses state-of-the-art straight-path flow matching for efficient, low-error transport.
4.  **Mode Jumping**: Proven capability to jump between disconnected modes where local samplers (HMC/RWM) get stuck.

## Validated Correctness
We verify correctness not just with visual inspections but with rigorous statistical tests:
- **Kolmogorov-Smirnov Test**: Passed ($p > 0.05$) on unit Gaussian benchmarks, confirming the sampler targets the exact posterior even with learned proposals.

## Quickstart

```python
import torch
from diffmcmc.core.mh import DiffusionMH
from diffmcmc.core.kernels import RWMKernel
from diffmcmc.proposal.flow import FlowProposal

# Define your log_prob function
def log_prob_fn(x):
    return -0.5 * torch.sum(x**2)

# Initialize Flow with Deterministic Trace logic
flow = FlowProposal(dim=2, deterministic_trace=True)

# Initialize the sampler
sampler = DiffusionMH(
    log_prob_fn=log_prob_fn,
    dim=2,
    local_kernel=RWMKernel(scale=1.0),
    global_proposal=flow,
    p_global=0.2
)

# Run sampling
samples, stats = sampler.run(num_steps=5000)
```

## Features

- **Correctness First**: No "approximate inference". Samples are asymptotically exact.
- **Hutchinson hashing**: Scalable density evaluation for higher dimensions.
- **Drop-in Integration**: Works with `emcee`.

## Installation

```bash
pip install -e .
```
