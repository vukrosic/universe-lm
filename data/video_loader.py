import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticVideoDataset(Dataset):
    """ Generates a video of a moving square for testing """
    def __init__(self, num_samples, config):
        self.num_samples = num_samples
        self.config = config
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        C = self.config.in_channels
        T = self.config.num_frames
        H = self.config.input_size
        W = self.config.input_size
        
        # Create a blank video [C, T, H, W]
        x = torch.zeros(C, T, H, W)
        
        # Random start position and velocity
        size = H // 4
        start_y = torch.randint(0, H - size, ())
        start_x = torch.randint(0, W - size, ())
        vy = torch.randint(-2, 3, ())
        vx = torch.randint(-2, 3, ())
        
        for t in range(T):
            y = (start_y + vy * t) % (H - size)
            x_pos = (start_x + vx * t) % (W - size)
            x[:, t, int(y):int(y+size), int(x_pos):int(x_pos+size)] = 1.0
            
        # Add a bit of noise
        x = x + torch.randn_like(x) * 0.05
        
        # Random label
        y_label = torch.randint(0, self.config.num_classes, ()) if self.config.num_classes > 0 else torch.tensor(0)
        
        return x, y_label

def get_dataloaders(config):
    train_ds = SyntheticVideoDataset(1000, config)
    val_ds = SyntheticVideoDataset(100, config)
    
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
