import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_flow_matching(flow_proposal, samples, batch_size=128, epochs=100, lr=1e-3, verbose=True):
    """
    Train the flow proposal using Rectified Flow objective.
    
    Args:
        flow_proposal: FlowProposal instance.
        samples: (N, D) torch.Tensor or numpy array of target samples.
        batch_size: Training batch size.
        epochs: Number of epochs.
        lr: Learning rate.
    """
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).float()
    
    device = next(flow_proposal.net.parameters()).device
    samples = samples.to(device)
    
    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(flow_proposal.net.parameters(), lr=lr)
    
    flow_proposal.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for (x1_batch,) in loader:
            batch_len = x1_batch.shape[0]
            
            # x0 ~ N(0, I)
            x0_batch = torch.randn_like(x1_batch)
            
            # Sample t ~ U[0, 1]
            t_batch = torch.rand(batch_len, device=device)
            
            # Linear interpolation: x_t = t * x1 + (1 - t) * x0
            # Broadcast t: (B, 1)
            t_view = t_batch.view(-1, 1)
            xt_batch = t_view * x1_batch + (1 - t_view) * x0_batch
            
            # Target velocity: v_target = x1 - x0
            v_target = x1_batch - x0_batch
            
            # Model prediction: v_pred = model(x_t, t)
            v_pred = flow_proposal.net(xt_batch, t_batch)
            
            loss = torch.mean((v_pred - v_target)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_len
            
        avg_loss = total_loss / len(samples)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
    flow_proposal.eval()
    return flow_proposal
