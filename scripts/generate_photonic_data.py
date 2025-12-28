import torch
import numpy as np
import os
import argparse
from diffmcmc.physics.optics import TransferMatrixMethod
from diffmcmc.data.io import DataIngestor

def generate_photonic_data(output_path, num_samples=2000):
    print(f"Generating Photonic Data ({num_samples} samples)...")
    
    # Physics Settings
    # 5-layer system: SiO2 | Si3N4 | SiO2 | Si3N4 | SiO2 (on Si substrate)
    # Materials (approx constant index for simplicity in high-dim inference, or frequency dependent)
    # Let's use constant to focus on Thickness Inference.
    # n_ambient = 1.0
    # n_SiO2 = 1.45
    # n_Si3N4 = 2.0
    # n_Si = 3.5
    
    n_ambient = 1.0
    n_1 = 1.45
    n_2 = 2.0
    n_sub = 3.5
    
    # Alternating stack 10 layers
    # Layers: [Amb, L1, L2, L1, L2, L1, L2, L1, L2, L1, L2, Sub]
    # Total 10 thin films.
    
    n_pattern = [n_ambient] + [n_1, n_2] * 5 + [n_sub]
    num_layers_total = len(n_pattern) # 1 + 10 + 1 = 12
    
    print(f"Layer Structure: {num_layers_total} layers (including ambient/sub)")
    
    # Wavelength grid
    wavelengths = torch.linspace(400, 1000, 200).float() # Extensive features
    # (200 spectral points)
    
    # Random Thicknesses
    # d ~ Uniform(50, 300) nm
    # Batch generation
    d_films = torch.rand(num_samples, 10) * 250.0 + 50.0 # (N, 10)
    
    # Pad d for ambient/sub (value doesn't matter, set to 0)
    d_amb = torch.zeros(num_samples, 1)
    d_sub = torch.zeros(num_samples, 1)
    
    d_stack = torch.cat([d_amb, d_films, d_sub], dim=1) # (N, 12)
    
    # N indices
    n_stack = torch.tensor(n_pattern).unsqueeze(0).repeat(num_samples, 1).float() # (N, 12)
    
    # Solve TMM
    print("Solving Maxwell's Equations (TMM)...")
    solver = TransferMatrixMethod()
    
    # Batch process in chunks to avoid OOM if N is huge
    batch_size = 100
    spectra_list = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            d_batch = d_stack[i:i+batch_size]
            n_batch = n_stack[i:i+batch_size]
            
            R = solver(n_batch, d_batch, wavelengths) # (B, W)
            spectra_list.append(R)
            
    spectra = torch.cat(spectra_list, dim=0)
    
    # Add noise?
    # Real world data has noise. Add 1% Gaussian noise.
    noise = torch.randn_like(spectra) * 0.01
    spectra_noisy = torch.clamp(spectra + noise, 0.0, 1.0)
    
    print("Saving to HDF5...")
    # Convert to numpy
    spectra_np = spectra_noisy.numpy()
    params_np = d_films.numpy() # Only infer the 10 film thicknesses
    waves_np = wavelengths.numpy()
    
    DataIngestor.save_dataset(
        output_path,
        spectra_np,
        params_np,
        waves_np,
        description="10-layer SiO2/Si3N4 random stack on Si. W=400-1000nm."
    )

if __name__ == "__main__":
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    generate_photonic_data('datasets/photonic_data.h5')
