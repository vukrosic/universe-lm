import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SquaredReLUFeedForward, SwiGLUFeedForward


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
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        use_attn_output_gate: bool = False,
        use_value_embed: bool = False,
        value_embed_rank: int | None = None,
        use_query_embed: bool = False,
        use_key_embed: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        
        # ============ MERGED QKVO PROJECTION ============
        # Instead of 4 separate Linear layers, use single merged projection
        q_size = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size = d_model
        
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = q_size + 2 * kv_size  # Q + K + V sizes
        
        # Single parameter tensor for all projections
        # Shape: [Q_size + K_size + V_size + O_size, d_model]
        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, d_model)
        )
        
        # Initialize all weights with std=0.02
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
        # ================================================
        
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)

        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout
        self.use_attn_output_gate = use_attn_output_gate
        if self.use_attn_output_gate:
            self.attn_output_gate = nn.Parameter(torch.zeros(n_heads))

        # #29 value embeddings: project the (factorized) token embedding into the
        # V subspace and add it to the values. Raw zero-init weight (not nn.Linear)
        # so it (a) starts as an exact baseline, (b) trains immediately via Muon,
        # and (c) draws no RNG at init — keeping every other weight bit-identical to
        # the control run, so the screen isolates the mechanism, not a re-seed.
        self.use_value_embed = use_value_embed
        if self.use_value_embed:
            assert value_embed_rank is not None, "value_embed_rank required"
            self.value_embed_proj = nn.Parameter(torch.zeros(self.kv_size, value_embed_rank))
        # #30 query embeddings: same trick on Q. Tests whether V's win is
        # V-specific or generalizes to "token identity straight into attention."
        self.use_query_embed = use_query_embed
        if self.use_query_embed:
            assert value_embed_rank is not None, "value_embed_rank required (used for Q too)"
            self.query_embed_proj = nn.Parameter(torch.zeros(self.q_size, value_embed_rank))
        # #31 key embeddings: same trick on K. K goes through RoPE after the
        # injection, so the projection's term gets positionally rotated — a
        # different operating point from V (no RoPE) or Q (also RoPE'd).
        self.use_key_embed = use_key_embed
        if self.use_key_embed:
            assert value_embed_rank is not None, "value_embed_rank required (used for K too)"
            self.key_embed_proj = nn.Parameter(torch.zeros(self.kv_size, value_embed_rank))

    def forward(self, x, ve=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # ============ MERGED QKV PROJECTION ============
        # Single matmul instead of 3 separate projections
        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])
        
        # Split the result into Q, K, V
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ================================================

        # #29 value embeddings: add the projected token embedding to the values
        # (before head reshape). Zero-inited projection => exact baseline at step 0.
        if self.use_value_embed and ve is not None:
            V = V + F.linear(ve, self.value_embed_proj)
        # #30 query embeddings: same trick, on Q.
        if self.use_query_embed and ve is not None:
            Q = Q + F.linear(ve, self.query_embed_proj)
        # #31 key embeddings: same trick, on K. (K then goes through RoPE
        # downstream, so this term is positionally rotated — different
        # operating point from V.)
        if self.use_key_embed and ve is not None:
            K = K + F.linear(ve, self.key_embed_proj)
        
        # Reshape to multi-head format
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply RoPE
        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))
        
        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)
        
        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        if self.use_attn_output_gate:
            gate = 1.0 + self.attn_output_gate.view(1, self.n_heads, 1, 1)
            attn_output = attn_output * gate
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # ============ MERGED O PROJECTION ============
        # Use the last part of qkvo_proj for output projection
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size:])


class TransformerBlock(nn.Module):
    """Standard transformer block with dense feed-forward"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        ffn_variant: str = "squared_relu",
        use_embed_residual: bool = False,
        use_attn_output_gate: bool = False,
        use_layerscale: bool = False,
        use_value_embed: bool = False,
        value_embed_rank: int | None = None,
        use_query_embed: bool = False,
        use_key_embed: bool = False,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            n_heads,
            max_seq_len,
            dropout,
            n_kv_heads,
            use_attn_output_gate=use_attn_output_gate,
            use_value_embed=use_value_embed,
            use_query_embed=use_query_embed,
            use_key_embed=use_key_embed,
            value_embed_rank=value_embed_rank,
        )
        if ffn_variant == "squared_relu":
            self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)
        elif ffn_variant == "swiglu":
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        else:
            raise ValueError(f"Unknown ffn_variant: {ffn_variant}")

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # #20 embedding residual: per-dim mix with the original token embedding x0,
        # init [m0=1, m1=0] so it starts exactly at baseline.
        self.use_embed_residual = use_embed_residual
        if use_embed_residual:
            self.resid_m0 = nn.Parameter(torch.ones(d_model))
            self.resid_m1 = nn.Parameter(torch.zeros(d_model))
        self.use_layerscale = use_layerscale
        if self.use_layerscale:
            self.attn_layerscale = nn.Parameter(torch.zeros(d_model))
            self.ffn_layerscale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, x0=None, ve=None):
        # Re-inject the original embedding before attention/MLP (#20)
        if self.use_embed_residual:
            x = self.resid_m0 * x + self.resid_m1 * x0

        # Self-attention
        attn_out = self.attention(self.norm1(x), ve)
        if self.use_layerscale:
            attn_out = attn_out * (1.0 + self.attn_layerscale)
        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        if self.use_layerscale:
            ff_out = ff_out * (1.0 + self.ffn_layerscale)
        x = x + self.dropout(ff_out)
        return x
