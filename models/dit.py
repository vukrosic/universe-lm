import torch
import torch.nn as nn
import numpy as np
from models.layers import TimestepEmbedder, DiTBlock, FinalLayer
from configs.dit_config import DiTConfig

class PatchEmbed(nn.Module):
    """ Video to Patch Embedding """
    def __init__(self, patch_size=2, in_channels=4, hidden_size=1152):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, hidden_size, 
            kernel_size=(patch_size, patch_size, patch_size), 
            stride=(patch_size, patch_size, patch_size)
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)
        # x: [B, D, T/p, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)
        # x: [B, L, D] where L = T/p * H/p * W/p
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer for Video Generation.
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads

        self.x_embedder = PatchEmbed(
            patch_size=config.patch_size, 
            in_channels=config.in_channels, 
            hidden_size=config.hidden_size
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        
        # Optional: Class/Text conditioning
        if config.num_classes > 0:
            self.y_embedder = nn.Embedding(config.num_classes, config.hidden_size)
            self.num_classes = config.num_classes
        else:
            self.y_embedder = None

        # Positional embedding
        # Calculate number of patches
        t_patches = config.num_frames // config.patch_size
        h_patches = config.input_size // config.patch_size
        w_patches = config.input_size // config.patch_size
        num_patches = t_patches * h_patches * w_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, config, mlp_ratio=config.mlp_ratio)
            for _ in range(config.depth)
        ])
        
        self.final_layer = FinalLayer(config.hidden_size, config.patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = self.get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            self.config.num_frames // self.patch_size,
            self.config.input_size // self.patch_size, 
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaln modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_3d_sincos_pos_embed(self, embed_dim, t_size, grid_size):
        """
        grid_size: int of the spatial grid
        t_size: int of the temporal grid
        return:
        pos_embed: [t*grid*grid, embed_dim]
        """
        assert embed_dim % 2 == 0
        w_size = grid_size
        h_size = grid_size
        
        # Helper for 1D
        def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
            assert embed_dim % 2 == 0
            omega = np.arange(embed_dim // 2, dtype=np.float32)
            omega /= embed_dim / 2.
            omega = 1. / 10000**omega

            pos = pos.reshape(-1)
            out = np.einsum('m,d->md', pos, omega)

            emb_sin = np.sin(out)
            emb_cos = np.cos(out)

            emb = np.concatenate([emb_sin, emb_cos], axis=1)
            return emb

        d_t = embed_dim // 3
        d_h = embed_dim // 3
        d_w = embed_dim - d_t - d_h

        grid_t = np.arange(t_size, dtype=np.float32)
        grid_h = np.arange(h_size, dtype=np.float32)
        grid_w = np.arange(w_size, dtype=np.float32)
        
        # meshgrid with indexing='ij' gives (T, H, W) order
        # flatten will give T major, then H, then W which matches token order
        out_t = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')[0].reshape(-1)
        out_h = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')[1].reshape(-1)
        out_w = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')[2].reshape(-1)

        pos_embed_t = get_1d_sincos_pos_embed_from_grid(d_t, out_t)
        pos_embed_h = get_1d_sincos_pos_embed_from_grid(d_h, out_h)
        pos_embed_w = get_1d_sincos_pos_embed_from_grid(d_w, out_w)
        
        pos_embed = np.concatenate([pos_embed_t, pos_embed_h, pos_embed_w], axis=1)
        return pos_embed

    def forward(self, x, t, y=None):
        """
        x: (N, C, T, H, W)
        t: (N,)
        y: (N,)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, L, D)
        t = self.t_embedder(t)                   # (N, D)
        
        if self.y_embedder is not None and y is not None:
            y_emb = self.y_embedder(y)
            c = t + y_emb
        else:
            c = t
            
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)                # (N, L, patch_size**3 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, T, H, W)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 * out_channels)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.config.input_size // p
        t = self.config.num_frames // p
        
        x = x.reshape(shape=(x.shape[0], t, h, w, p, p, p, c))
        # permute to (N, c, t, p_t, h, p_h, w, p_w)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.reshape(shape=(x.shape[0], c, t * p, h * p, w * p))
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        # Simplified for now
        return self.forward(x, t, y)
