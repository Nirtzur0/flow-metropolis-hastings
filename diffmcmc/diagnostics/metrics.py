import numpy as np

def compute_ess(chain: np.ndarray) -> np.ndarray:
    """
    Compute Effective Sample Size (ESS) for each dimension of the chain.
    Uses the variogram estimator or simple autocorrelation sum.
    We implement the standard Geyer's initial convex sequence estimator or similar.
    
    Args:
        chain: (N, D) numpy array
        
    Returns:
        ess: (D,) numpy array
    """
    N, D = chain.shape
    if N < 2:
        return np.ones(D)
        
    ess = np.zeros(D)
    for d in range(D):
        x = chain[:, d]
        # Compute autocorrelation
        # fft based
        n = len(x)
        # Pad for fft
        f = np.fft.fft(x - np.mean(x), n=2*n)
        acf = np.real(np.fft.ifft(f * np.conjugate(f)))[:n]
        acf = acf / acf[0]
        
        # Simple sum until negative or standard heuristic
        # Geyer's initial monotone sequence estimator is robust
        # Let's use simple cutoff where rho < 0.05 or similar for now,
        # or just sum up to lag where rho < 0?
        
        # Sum of ACF
        # tau = 1 + 2 * sum(rho[1:])
        # We truncate where rho drops below 0 presumably or noise level.
        
        tau = 1.0
        for k in range(1, n):
            if acf[k] < 0.05: # Simple heuristic cutoff
                break
            tau += 2 * acf[k]
            
        ess[d] = N / tau
        
    return ess
