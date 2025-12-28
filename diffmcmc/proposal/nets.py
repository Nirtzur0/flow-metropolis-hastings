import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))

    def forward(self, t):
        # t: (B,)
        # returns: (B, dim)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1) # (B, 1)
        
        # Simple MLP embedding usually works better for Rectified Flow than complex sinusoidal
        # But let's stick to a simple concatenation or MLP if dimension allows.
        # Actually, let's just use a Gaussian Fourier projection or simple MLP for time.
        pass

class VelocityMLP(nn.Module):
    """
    Simple MLP velocity field v(x, t).
    """
    def __init__(self, dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.dim = dim
        
        # Time embedding: simple linear or concatenation
        # We'll just concatenate t as an extra channel
        input_dim = dim + 1 
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        """
        x: (B, D)
        t: (B,) or scalar
        """
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t = torch.ones(x.shape[0], 1, device=x.device) * t
        elif t.ndim == 1:
            t = t.unsqueeze(1)
            
        # Broadcast t if necessary
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
            
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)
