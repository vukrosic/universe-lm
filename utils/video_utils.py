
import torch
import numpy as np
from PIL import Image
import os

def save_video_as_gif(video, filename, fps=10):
    """
    video: [C, T, H, W] tensor, range [0, 1]
    filename: output gif path
    """
    # Convert to [T, H, W, C] and [0, 255]
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    video = (video * 255).clip(0, 255).astype(np.uint8)
    
    # If C=1, repeat to RGB
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    elif video.shape[-1] == 4:
        # If it's latent (C=4), just take first 3 for visualization or grayscale
        video = video[..., :3]
        
    frames = [Image.fromarray(frame) for frame in video]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)

def save_video_grid(videos, filename, rows=None, fps=10):
    """
    videos: [B, C, T, H, W] tensor
    """
    B, C, T, H, W = videos.shape
    if rows is None:
        rows = int(np.sqrt(B))
    cols = B // rows
    
    # Simple grid concatenation
    # videos: [T, B, C, H, W]
    videos = videos.permute(2, 0, 1, 3, 4)
    
    combined_frames = []
    for t in range(T):
        frame_t = videos[t] # [B, C, H, W]
        # Reshape to grid
        # frame_t: [rows, cols, C, H, W]
        grid = frame_t.view(rows, cols, C, H, W)
        grid = grid.permute(0, 3, 1, 4, 2) # [rows, H, cols, W, C]
        grid = grid.reshape(rows * H, cols * W, C)
        
        # Convert to RGB
        grid_np = grid.cpu().numpy()
        grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
        if C == 1:
            grid_np = np.repeat(grid_np, 3, axis=-1)
        elif C == 4:
            grid_np = grid_np[..., :3]
            
        combined_frames.append(Image.fromarray(grid_np))
        
    combined_frames[0].save(filename, save_all=True, append_images=combined_frames[1:], duration=1000//fps, loop=0)
