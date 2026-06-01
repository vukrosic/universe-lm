import torch
import torch.nn as nn
import math
from typing import Optional
from configs.llm_config import LLMConfig
from models.layers import TransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                    qk_gain_init=config.qk_gain_init,
                    ffn_variant=config.ffn_variant,
                    residual_scale_init=config.residual_scale_init,
                    embedding_residual_scale_init=config.embedding_residual_scale_init,
                )
                for i in range(config.n_layers)
            ]
        )

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        if getattr(config, "zero_init_output_projections", False):
            self._zero_init_output_projections()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _zero_init_output_projections(self) -> None:
        """Zero the residual-producing projections for an identity-start lever."""
        for block in self.transformer_blocks:
            attention = block.attention
            with torch.no_grad():
                attention.qkvo_proj[attention.qkv_size:].zero_()
                block.feed_forward.down_proj.weight.zero_()

    def forward(self, x):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        x0 = x

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, x0=x0)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        if getattr(self.config, "logit_softcap", None):
            cap = float(self.config.logit_softcap)
            logits = cap * torch.tanh(logits / cap)

        return logits
