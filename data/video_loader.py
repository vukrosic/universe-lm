import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyVideoDataset(Dataset):
    def __init__(self, num_samples, config):
        self.num_samples = num_samples
        self.config = config
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random video latent: [C, T, H, W]
        # Using config.input_size which is spatial, and num_frames
        # C is usually 4 (latent channels)
        C = self.config.in_channels
        T = self.config.num_frames
        H = self.config.input_size
        W = self.config.input_size
        
        # Random noise for now, mimicking latents
        x = torch.randn(C, T, H, W)
        
        # Random label
        y = torch.randint(0, self.config.num_classes, ()) if self.config.num_classes > 0 else torch.tensor(0)
        
        return x, y

def get_dataloaders(config):
    train_ds = DummyVideoDataset(1000, config)
    val_ds = DummyVideoDataset(100, config)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader
