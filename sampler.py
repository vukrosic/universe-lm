
import torch

@torch.no_grad()
def sample_euler(model, config, device, steps=50, cfg_scale=1.0, num_samples=4):
    model.eval()
    # 1. Start with pure noise x_0
    # Shape: [B, C, T, H, W]
    x = torch.randn(num_samples, config.in_channels, config.num_frames, config.input_size, config.input_size, device=device)
    
    dt = 1.0 / steps
    
    # Dummy labels if needed
    y = torch.zeros(num_samples, dtype=torch.long, device=device)
    
    for i in range(steps):
        # Current time t from 0 to 1
        t = torch.ones(num_samples, device=device) * (i / steps)
        
        # Predict velocity v = dx/dt
        # Note: model should be wrapped in autocast if trained with it
        with torch.amp.autocast('cuda'):
            v = model(x, t, y)
            if v.shape[1] != x.shape[1] and v.shape[1] == x.shape[1] * 2:
                v, _ = torch.split(v, x.shape[1], dim=1)
            
        # Update: x_{t+dt} = x_t + v * dt
        x = x + v * dt

        
    model.train()
    return x
