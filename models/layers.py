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
        use_output_embed: bool = False,
        use_q_gain: bool = False,
        use_k_gain: bool = False,
        use_deep_value_embed: bool = False,
        deep_value_embed_hidden: int | None = None,
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
        # #45 deep value embeddings: 2-layer non-linear V projection.
        # V += GELU(ve @ W1) @ W2.
        # F.linear(ve, W1) computes ve @ W1.T, so W1 is stored as
        # [hidden, emb_rank] (like Linear's weight convention).
        # Same for W2: stored as [kv_size, hidden].
        # Both zero-init so step 0 = exact baseline.
        # Cost per layer: hidden × emb_rank + kv_size × hidden.
        # For Screen10M20M (emb_rank=48, hidden=96, kv_size=48): 9,216.
        # Total: 24 × 9,216 = 221,184 params (+2.9%).
        self.use_deep_value_embed = use_deep_value_embed
        if self.use_deep_value_embed:
            assert value_embed_rank is not None, "value_embed_rank required for deep V-embed"
            assert deep_value_embed_hidden is not None, "deep_value_embed_hidden required"
            self.deep_value_embed_W1 = nn.Parameter(
                torch.zeros(deep_value_embed_hidden, value_embed_rank)
            )
            self.deep_value_embed_W2 = nn.Parameter(
                torch.zeros(self.kv_size, deep_value_embed_hidden)
            )
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
        # #33 output embeddings: same trick, applied AFTER the O projection
        # (output side of attention, not input side). This is the
        # modded-nanogpt speedrun "value embeddings" position — the token
        # identity bypasses the attention computation entirely and lands
        # straight in the residual. Tests "is V-embed winning because V is
        # a unique position, or because any token-signal-to-residual helps?"
        # Shape: [d_model, emb_rank] (one d_model, not kv_size — the output
        # is full d_model, not per-head).
        self.use_output_embed = use_output_embed
        if self.use_output_embed:
            assert value_embed_rank is not None, "value_embed_rank required (used for O too)"
            self.output_embed_proj = nn.Parameter(torch.zeros(self.d_model, value_embed_rank))
        # #37 per-head Q-gain: a learnable per-head scalar that multiplies
        # the Q vector after norm+RoPE. Zero-init so the model starts as
        # exact baseline (1 + 0 = 1). Equivalent to a per-head
        # temperature on the attention scores. Known modded-nanogpt
        # speedrun trick (q_gain in the parameter-golf baseline). Cost:
        # n_heads scalars per layer = 6 × 24 = 144 total extra params.
        # Non-embed lever: changes the attention math, not the inputs.
        self.use_q_gain = use_q_gain
        if self.use_q_gain:
            self.q_gain = nn.Parameter(torch.zeros(self.n_heads))
        # #42 per-head K-gain: symmetric to q_gain, but on K. Tests
        # whether scaling K helps as much as scaling Q, and whether
        # both are additive (V+q+k_gain might beat V+q_gain).
        self.use_k_gain = use_k_gain
        if self.use_k_gain:
            self.k_gain = nn.Parameter(torch.zeros(self.n_heads))

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
        # #45 deep value embeddings: 2-layer non-linear V projection.
        # V += GELU(ve @ W1) @ W2. Both W1 and W2 are zero-init so step 0
        # is exact baseline. The GELU has a dead-zone at 0 so the
        # gradient flows through W2 first (Muon), then W1.
        if self.use_deep_value_embed and ve is not None:
            v_hidden = F.gelu(F.linear(ve, self.deep_value_embed_W1))
            V = V + F.linear(v_hidden, self.deep_value_embed_W2)
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
        # #37 per-head Q-gain: multiply Q by (1 + q_gain) per head after
        # RoPE. Zero-init, so step 0 == baseline.
        if self.use_q_gain:
            Q = Q * (1.0 + self.q_gain.view(1, 1, self.n_heads, 1))
        # #42 per-head K-gain: symmetric to Q-gain. Multiplies K after
        # RoPE. Zero-init baseline. Applied AFTER repeat_interleave so
        # the per-head scalar matches the final head count (n_heads).
        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)
        if self.use_k_gain:
            K = K * (1.0 + self.k_gain.view(1, 1, self.n_heads, 1))

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
        output = F.linear(attn_output, self.qkvo_proj[self.qkv_size:])
        # #33 output embeddings: add the projected token embedding to the
        # attention OUTPUT (post-O). Different operating point from V/Q/K
        # (which inject into attention inputs). ve is the raw token
        # embedding [B, T, emb_rank], projection is zero-init so step 0
        # matches the baseline.
        if self.use_output_embed and ve is not None:
            output = output + F.linear(ve, self.output_embed_proj)
        return output


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
        use_output_embed: bool = False,
        use_q_gain: bool = False,
        use_k_gain: bool = False,
        use_deep_value_embed: bool = False,
        deep_value_embed_hidden: int | None = None,
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
            use_output_embed=use_output_embed,
            use_q_gain=use_q_gain,
            use_k_gain=use_k_gain,
            use_deep_value_embed=use_deep_value_embed,
            deep_value_embed_hidden=deep_value_embed_hidden,
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
