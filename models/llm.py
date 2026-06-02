import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from configs.llm_config import LLMConfig
from models.layers import TransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings.
        # emb_rank is None -> full (vocab x d_model) table (default).
        # emb_rank=r -> low-rank factorization: (vocab x r) @ (r x d_model).
        self.emb_rank = getattr(config, "emb_rank", None)
        if self.emb_rank is None:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.emb_proj = None
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, self.emb_rank)
            self.emb_proj = nn.Linear(self.emb_rank, config.d_model, bias=False)
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
                    ffn_variant=config.ffn_variant,
                )
                for i in range(config.n_layers)
            ]
        )

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings).
        # Full case: standard tied Linear. Factorized case: lm_head is computed
        # functionally in forward() through the SAME two matrices, so input and
        # output embeddings stay tied with zero extra params.
        if self.emb_rank is None:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Token embeddings
        if self.emb_rank is None:
            x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        else:
            x = self.emb_proj(self.token_embedding(x)) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        if self.emb_rank is None:
            logits = self.lm_head(x)
        else:
            # Tied factorized head: d_model -> r (via emb_proj^T) -> vocab (via
            # token_embedding^T). Reuses the exact embedding matrices.
            z = F.linear(x, self.emb_proj.weight.t())          # (..., r)
            logits = F.linear(z, self.token_embedding.weight)  # (..., vocab)

        return logits
