import torch
import torch.nn as nn
from typing import Tuple, Optional

class TransferMatrixMethod(nn.Module):
    """
    Differentiable Transfer Matrix Method (TMM) solver for multilayer thin films.
    Calculates Reflectance (R) and Transmittance (T) for a stack of layers.
    
    Assumptions:
    - Normal incidence (can be extended to angles).
    - Non-magnetic materials (mu=1).
    - Polarization generic (normal incidence s=p).
    """
    def __init__(self):
        super().__init__()

    def forward(self, 
                n_layers: torch.Tensor, 
                d_layers: torch.Tensor, 
                wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_layers: (Batch, N_layers) Refractive indices (complex or real).
                      Typically: [n_substrate, n_1, n_2, ..., n_ambient]
            d_layers: (Batch, N_layers) Physical thicknesses in nanometers.
                      Note: Substrate and Ambient thicknesses are ignored (treated as semi-infinite).
            wavelengths: (W,) Wavelengths in nanometers.
            
        Returns:
            R: (Batch, W) Reflectance spectrum.
        """
        # Ensure complex for Fresnel calc
        if not n_layers.is_complex():
            n_layers = n_layers.to(torch.complex64)
            
        # Add singleton dims for broadcasting:
        # Batch (B), Layers (L), Wavelengths (W)
        # n_layers: (B, L, 1)
        # d_layers: (B, L, 1)
        # wavelengths: (1, 1, W)
        
        n = n_layers.unsqueeze(-1) # (B, N, 1)
        d = d_layers.unsqueeze(-1) # (B, N, 1)
        lam = wavelengths.view(1, 1, -1) # (1, 1, W)
        
        batch_size = n.shape[0]
        num_layers = n.shape[1]
        num_waves = lam.shape[2]
        
        # Characteristic Admittance (Normal incidence: Y = n)
        # In general Y_s = n * cos(theta), Y_p = n / cos(theta)
        # For normal, theta=0 => Y = n.
        
        # System Matrix M (Start as Identity)
        # M is (B, 2, 2, W)
        M = torch.eye(2, dtype=torch.complex64, device=n.device)
        M = M.view(1, 2, 2, 1).expand(batch_size, -1, -1, num_waves).clone()
        
        # Iterate through layers (excluding first and last which are infinite media)
        # The structure is: Substrate (0) | Layer 1 | ... | Layer N | Ambient (-1)
        # Or: Ambient (0) | Layer 1 ... | Substrate
        # Convention: Light comes from Ambient (Layer 0) -> Stack -> Substrate (Layer -1).
        # But standard formulation typically propagates from first interface to last.
        
        # Let's define n_layers as [n_incident, n_1, ..., n_substrate]
        # Iterate interfaces. Interface k is between layer k-1 and k.
        
        # Standard Transfer Matrix recursive formulation:
        # M_total = M_1 * M_2 * ... * M_N
        # where M_layer is phase propagation.
        # Wait, usually M relates fields.
        # M_layer = [[cos phi, -i/n sin phi], [-i n sin phi, cos phi]]
        
        # Loop over inner layers (1 to N-2) if N is total count including semi-infinite ends
        # Indices: 0=Input Medium, 1..N-2=Thin Films, N-1=Exit Medium
        
        for i in range(1, num_layers - 1):
            n_i = n[:, i, :]
            d_i = d[:, i, :]
            
            # Phase thickness phi = 2 pi n d / lambda
            k0 = 2 * torch.pi / lam # (1, 1, W)
            phi = n_i * d_i * k0 # (B, 1, W)
            
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            
            # Construct Layer Matrix M_i
            # Shape (B, 2, 2, W)
            # m00 = cos_phi
            # m01 = -1j * sin_phi / n_i
            # m10 = -1j * n_i * sin_phi
            # m11 = cos_phi
            
            # Stack manually to form matrix
            # Be careful with dimensions
            # We want (B, 2, 2, W)
            
            zeros = torch.zeros_like(phi)
            ones = torch.ones_like(phi)
            
            # Re-assemble
            # This is slightly slow in loop, specialized kernels better, but ok for demo.
            
            # Optimized matrix multiplication: M_new = M_old @ M_layer
            # M_layer elements:
            m11 = cos_phi
            m12 = -1j * sin_phi / (n_i + 1e-8) # Avoid div zero
            m21 = -1j * n_i * sin_phi
            m22 = cos_phi
            
            # M_curr (B, 2, 2, W)
            # Perform batch matmul (B, ..., 2, 2)
            # permute W to batch dim or keep as is?
            # torch.matmul broadcasts over Batch, but W is at end?
            # let's permute to (B, W, 2, 2) for matmul
            
            M = M.permute(0, 3, 1, 2) # (B, W, 2, 2)
            
            M_layer = torch.zeros_like(M)
            M_layer[:, :, 0, 0] = m11.squeeze(1)
            M_layer[:, :, 0, 1] = m12.squeeze(1)
            M_layer[:, :, 1, 0] = m21.squeeze(1)
            M_layer[:, :, 1, 1] = m22.squeeze(1)
            
            M = torch.matmul(M, M_layer)
            M = M.permute(0, 2, 3, 1) # Back to (B, 2, 2, W)
            
        # Apply Boundary Conditions to get R
        # Fields: [E0, H0] = M [E_sub, H_sub]
        # At substrate (infinite), only forward wave: E_sub = 1, H_sub = n_sub * E_sub = n_sub
        # No, that's not quite right.
        # Ref: Orfanidis "Electromagnetic Waves", Ch 4.
        # B = (M00 + M01 n_sub) + (M10 + M11 n_sub) / n_0  <-- if using transmission coeff formulation
        
        # Let's use Characteristic Matrix approach:
        # [B, C] = M * [1, n_sub] (Assuming normalized E_sub=1)
        # Admittance Y = C / B
        # Reflection coefficient r = (Y_0 - Y) / (Y_0 + Y)
        # where Y_0 = n_incident
        
        n_in = n[:, 0, :]   # Incident medium
        n_sub = n[:, -1, :] # Substrate
        
        # M is total transfer matrix of the STACK (excluding interfaces with ambient/sub)
        # Wait, classical TMM includes the interfaces?
        # The formulation above (M_layer) propagates fields across the layer.
        # The BCs at the ends incorporate the refractive indices of semi-infinite media.
        
        # Matvec: [E_in, H_in] = M @ [E_out, H_out]
        # Boundary condition at output (substrate): No backward wave.
        # E_out = 1, H_out = n_sub * E_out = n_sub
        
        # Vector (B, 2, 1, W)
        BC_sub = torch.zeros(batch_size, 2, 1, num_waves, dtype=torch.complex64, device=n.device)
        BC_sub[:, 0, 0, :] = 1.0
        # n_sub is (B, 1). Expand to (B, W).
        BC_sub[:, 1, 0, :] = n_sub.squeeze(1).unsqueeze(-1).expand(-1, num_waves)
        
        # Propagate to input
        # M: (B, 2, 2, W)
        # (B, 2, 2, W) @ (B, 2, 1, W) -> (B, 2, 1, W)
        # matmul auto-broadcasts? Dimensions must align.
        # M is (B, 2, 2, W), BC is (B, 2, 1, W). Matmul operates on last 2 dims.
        # So we need to permute W out of the way or handle it.
        
        M_run = M.permute(0, 3, 1, 2) # (B, W, 2, 2)
        BC_run = BC_sub.permute(0, 3, 1, 2) # (B, W, 2, 1)
        
        EH_in = torch.matmul(M_run, BC_run) # (B, W, 2, 1)
        
        E_in = EH_in[:, :, 0, 0] # (B, W)
        H_in = EH_in[:, :, 1, 0] # (B, W)
        
        # Input Admittance Y_in = H_in / E_in
        Y_in = H_in / (E_in + 1e-9)
        
        # Reflection Coefficient r = (n_in - Y_in) / (n_in + Y_in)
        # careful with sign convention. traditionally (Y0 - Y)/(Y0 + Y)
        n_in_sq = n_in.expand(-1, num_waves) # n_in is (B, 1). Expand to (B, W).
        
        r = (n_in_sq - Y_in) / (n_in_sq + Y_in)
        
        # Reflectance R = |r|^2
        R = torch.abs(r)**2
        
        return R
