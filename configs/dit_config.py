from dataclasses import dataclass

@dataclass
class DiTConfig:
    # Model Architecture
    in_channels: int = 4  # Latent channels (e.g. from VAE)
    input_size: int = 32  # Spatial size (H, W)
    num_frames: int = 16  # Temporal size (T)
    patch_size: int = 2   # 2x2x2 patches
    hidden_size: int = 1152 # d_model
    depth: int = 28         # n_layers
    num_heads: int = 16     # n_heads
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = True
    
    # Training
    train_steps: int = 10000
    batch_size: int = 16 # Global batch size
    micro_batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    
    # Optimization
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    
    # Checkpointing
    save_every: int = 1000
    log_every: int = 10
    
    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_heads(self):
        return self.num_heads
    
    @property
    def n_layers(self):
        return self.depth
