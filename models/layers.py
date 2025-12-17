import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import MixtureOfExperts, SwiGLUFeedForward


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=10000
        )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, n_kv_heads: int | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Calculate queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) # [B, H, T, D]
        k = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2) # [B, KV_H, T, D]
        v = self.v_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2) # [B, KV_H, T, D]
        
        Q, K, V = q, k, v

        # Apply RoPE
        Q = self.rotary(self.q_norm(Q.transpose(1, 2))).transpose(1, 2)
        K = self.rotary(self.k_norm(K.transpose(1, 2))).transpose(1, 2)
        
        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=1)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        qk_rope_dim: int | None,
        qk_nope_dim: int | None,
        kv_lora_rank: int | None,
        v_dim: int | None,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_moe: bool = True,
        n_kv_heads: int | None = None,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, n_kv_heads)

        # Feed-forward layer (MoE or Dense)
        self.use_moe = use_moe
        if use_moe:
            self.feed_forward = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        else:
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward
        if self.use_moe:
            ff_out, aux_loss = self.feed_forward(self.norm2(x))
        else:
            ff_out = self.feed_forward(self.norm2(x))
            aux_loss = None
            
        x = x + self.dropout(ff_out)
        return x, aux_loss
