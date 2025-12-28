import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import json

@dataclass
class DatasetMetadata:
    name: str
    description: str
    num_samples: int
    wavelengths: np.ndarray # Array of wavelengths
    creation_date: str
    
    def to_json(self):
        d = asdict(self)
        d['wavelengths'] = d['wavelengths'].tolist()
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str):
        d = json.loads(json_str)
        d['wavelengths'] = np.array(d['wavelengths'])
        return cls(**d)

class PhotonicHDF5Dataset(Dataset):
    """
    Professional PyTorch Dataset for Photonic Data in HDF5.
    Supports lazy loading and metadata validation.
    """
    def __init__(self, h5_path: str, mode: str = 'r'):
        self.h5_path = h5_path
        self.mode = mode
        self.file = None
        
        # Validation on init
        with h5py.File(h5_path, 'r') as f:
            if 'metadata' not in f.attrs:
                raise ValueError("HDF5 file missing required 'metadata' attribute.")
            self.metadata = DatasetMetadata.from_json(f.attrs['metadata'])
            self.length = self.metadata.num_samples
            
    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, self.mode)
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        """
        Returns dictionary of tensors: {'spectrum': ..., 'params': ...}
        """
        self._open_file()
        
        # Assume layout: /data/spectra, /data/params
        # Slicing HDF5 is efficient
        spectrum = self.file['data']['spectra'][idx]
        params = self.file['data']['params'][idx]
        
        return {
            'spectrum': torch.from_numpy(spectrum).float(),
            'params': torch.from_numpy(params).float()
        }
        
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

class DataIngestor:
    """
    Handles robust writing of the dataset.
    """
    @staticmethod
    def save_dataset(path: str, 
                     spectra: np.ndarray, 
                     params: np.ndarray, 
                     wavelengths: np.ndarray,
                     description: str = "Photonic TMM Dataset"):
        """
        Save data with schema best practices.
        spectra: (N, W)
        params: (N, P)
        """
        assert spectra.shape[0] == params.shape[0], "Mismatch in sample count"
        assert spectra.shape[1] == len(wavelengths), "Mismatch in spectral dimension"
        
        meta = DatasetMetadata(
            name="Photonic_TMM_ThinFilm",
            description=description,
            num_samples=spectra.shape[0],
            wavelengths=wavelengths,
            creation_date=str(np.datetime64('now'))
        )
        
        with h5py.File(path, 'w') as f:
            # Metadata
            f.attrs['metadata'] = meta.to_json()
            
            # Groups
            grp = f.create_group("data")
            
            # Datasets with compression
            grp.create_dataset("spectra", data=spectra, compression="gzip", chunks=True)
            grp.create_dataset("params", data=params, compression="gzip", chunks=True)
            
        print(f"Dataset successfully saved to {path} with {meta.num_samples} samples.")
