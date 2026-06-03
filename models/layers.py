import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SquaredReLUFeedForward, SwiGLUFeedForward, GELUFeedForward


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=base
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
        use_qk_norm_post_rope: bool = False,
        use_sliding_window: bool = False,
        sliding_window_size: int = 512,
        use_nope: bool = False,
        rope_base: int = 10000,
        use_tied_qk: bool = False,
        use_mla: bool = False,
        mla_latent_dim: int | None = None,
        attention_dilation: int = 1,
        use_post_norm: bool = False,
    ):
        super().__init__()
        # #75 Post-norm: when set, the norm is applied AFTER the
        # residual addition instead of before. Implementation:
        # compute (norm, residual) inside the function but apply
        # the norm to (x + sublayer_out) before returning.
        self.use_post_norm = use_post_norm
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
        # #72 Tied QK: when set, Q and K share a single W (PaLM-style).
        # We allocate a SEPARATE parameter and don't use the Q/K slices
        # of the merged qkvo_proj in that case.
        self.use_tied_qk = use_tied_qk
        if use_tied_qk:
            self.qk_proj = nn.Parameter(torch.empty(q_size + kv_size, d_model))
        # #73 MLA: latent K,V with down/up projections. Separate
        # parameters; the standard Q/K/V slices of qkvo_proj are unused
        # when MLA is on.
        self.use_mla = use_mla
        self.mla_latent_dim = mla_latent_dim if mla_latent_dim is not None else max(8, d_model // 4)
        if use_mla:
            self.mla_dkv = nn.Parameter(torch.empty(self.mla_latent_dim, d_model))
            self.mla_uk  = nn.Parameter(torch.empty(kv_size, self.mla_latent_dim))
            self.mla_uv  = nn.Parameter(torch.empty(kv_size, self.mla_latent_dim))
        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, d_model)
        )
        
        # Initialize all weights with std=0.02
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
            if use_tied_qk:
                torch.nn.init.normal_(self.qk_proj, mean=0.0, std=0.02)
            if use_mla:
                torch.nn.init.normal_(self.mla_dkv, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.mla_uk,  mean=0.0, std=0.02)
                torch.nn.init.normal_(self.mla_uv,  mean=0.0, std=0.02)
        # ================================================
        
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)

        self.rotary = Rotary(self.d_k, max_seq_len, base=rope_base)
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
        # #49 QK-norm-post-RoPE: apply RMSNorm to Q,K AFTER RoPE (modded-
        # nanogpt trick) instead of the default BEFORE RoPE. Different
        # mathematical operating point. Flag-only, no extra params.
        self.use_qk_norm_post_rope = use_qk_norm_post_rope
        # #53 NoPE: skip the rotary positional embedding entirely. The
        # Q,K tensors still go through RMSNorm (norm is the Q/K
        # magnitude stabilizer, separate concern from position), but
        # the rotary is bypassed.
        self.use_nope = use_nope
        # #63 RoPE base: control the wavelength of the rotary. The
        # default base=10000 is GPT-Neo style; Llama uses 500000 which
        # extends the useful positional range. Tests whether the
        # default decay is hurting at our seq_len=2048.
        self.rope_base = rope_base
        # #51 sliding-window attention: build a [T, T] causal-local
        # boolean mask once at init and reuse. True = attend,
        # False = mask out. Built for max_seq_len; the SDPA call slices
        # the upper-left [seq_len, seq_len] submatrix. causal AND
        # local-window — both are required, otherwise the mask lets
        # each position attend to its future.
        # #74 dilated attention: same as SWA but the window consists
        # of every `attention_dilation`-th position (dilation=1 is
        # the contiguous SWA case; dilation=2 takes every other
        # position in the window range; etc.).
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.attention_dilation = max(1, int(attention_dilation))
        if self.use_sliding_window:
            idx = torch.arange(max_seq_len)
            diff = idx[:, None] - idx[None, :]
            if self.attention_dilation == 1:
                # contiguous SWA — original mask
                self.register_buffer(
                    "_sliding_window_mask",
                    (diff >= 0) & (diff < sliding_window_size),
                    persistent=False,
                )
            else:
                # dilated: keep positions j where diff is a multiple
                # of dilation AND within the window
                self.register_buffer(
                    "_sliding_window_mask",
                    (diff >= 0)
                    & (diff < sliding_window_size)
                    & ((diff % self.attention_dilation) == 0),
                    persistent=False,
                )

    def forward(self, x, ve=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # ============ MERGED QKV PROJECTION ============
        # Single matmul instead of 3 separate projections
        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])
        
        # Split the result into Q, K, V
        # #72 Tied QK (PaLM): Q and K share the same W matrix. Use a
        # separate qk_proj parameter; the Q/K slices of qkvo_proj are
        # unused in this mode. V is still from its qkvo_proj slice.
        # #73 MLA: K, V come from a low-rank latent. The latent is
        # computed once per layer (down-project input), then
        # up-projected per head to K, V.
        if self.use_tied_qk:
            qk = F.linear(x, self.qk_proj)
            Q, K = qk.split([self.q_size, self.kv_size], dim=-1)
            V = F.linear(x, self.qkvo_proj[self.qkv_size - self.kv_size:self.qkv_size])
        elif self.use_mla:
            latent = F.linear(x, self.mla_dkv)  # [B, T, mla_latent_dim]
            K = F.linear(latent, self.mla_uk)    # [B, T, kv_size]
            V = F.linear(latent, self.mla_uv)    # [B, T, kv_size]
            Q = F.linear(x, self.qkvo_proj[:self.q_size])
        else:
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
        # #49 QK-norm-post-RoPE: by default we apply RMSNorm to Q,K BEFORE
        # RoPE (the pre-RoPE norm). The modded-nanogpt variant applies the
        # norm AFTER RoPE. The two are mathematically different — post-RoPE
        # norm constrains the post-RoPE Q,K magnitudes per head, which can
        # help with attention score stability at scale.
        # #53 NoPE: when use_nope is set, skip the rotary call entirely.
        # RMSNorm still runs (it's a Q/K magnitude stabilizer, separate
        # from position), but the rotation is bypassed.
        if self.use_nope:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        elif self.use_qk_norm_post_rope:
            Q = self.q_norm(self.rotary(Q))
            K = self.k_norm(self.rotary(K))
        else:
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
        # #51 sliding-window: when enabled, use a [T, T] causal-local
        # boolean mask instead of SDPA's `is_causal=True` fast path.
        # The mask is broadcast across batch and head dims.
        if self.use_sliding_window:
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=self._sliding_window_mask[:seq_len, :seq_len],
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
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
        use_ffn_embed: bool = False,
        use_qk_norm_post_rope: bool = False,
        use_sliding_window: bool = False,
        sliding_window_size: int = 512,
        use_nope: bool = False,
        rope_base: int = 10000,
        use_tied_qk: bool = False,
        use_mla: bool = False,
        mla_latent_dim: int | None = None,
        attention_dilation: int = 1,
        use_post_norm: bool = False,
    ):
        super().__init__()
        # #75 Post-norm: when set, the norm is applied AFTER the
        # residual addition instead of before. Implementation:
        # compute (norm, residual) inside the function but apply
        # the norm to (x + sublayer_out) before returning.
        self.use_post_norm = use_post_norm

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
            use_qk_norm_post_rope=use_qk_norm_post_rope,
            use_sliding_window=use_sliding_window,
            sliding_window_size=sliding_window_size,
            use_nope=use_nope,
            rope_base=rope_base,
            use_tied_qk=use_tied_qk,
            use_mla=use_mla,
            mla_latent_dim=mla_latent_dim,
            attention_dilation=attention_dilation,
            value_embed_rank=value_embed_rank,
        )
        if ffn_variant == "squared_relu":
            self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)
        elif ffn_variant == "swiglu":
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        elif ffn_variant == "gelu":
            self.feed_forward = GELUFeedForward(d_model, d_ff, dropout)
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
        # #47 FFN embeddings: add a learned projection of the factorized
        # token embedding to the FFN input. Different position from
        # V-embed (#29, inside attention) and O-embed (#33, post-O).
        # Tests whether the V-embed win is about attention content or
        # about residual content. Zero-init, so step 0 = exact baseline.
        # Cost: 24 × (d_model 144 × emb_rank 48) = 165,888 extra params.
        self.use_ffn_embed = use_ffn_embed
        if self.use_ffn_embed:
            assert value_embed_rank is not None, "value_embed_rank required for FFN-embed"
            self.ffn_embed_proj = nn.Parameter(
                torch.zeros(d_model, value_embed_rank)
            )

    def forward(self, x, x0=None, ve=None):
        # Re-inject the original embedding before attention/MLP (#20)
        if self.use_embed_residual:
            x = self.resid_m0 * x + self.resid_m1 * x0

        if self.use_post_norm:
            # #75 Post-norm: apply norm AFTER the residual addition.
            # x = norm(x + sublayer(x_or_norm_x))
            # We use the un-normalized x as the sublayer input (the
            # original Transformer design — sometimes called "post-norm").
            attn_out = self.attention(x, ve)
            if self.use_layerscale:
                attn_out = attn_out * (1.0 + self.attn_layerscale)
            x = self.norm1(x + self.dropout(attn_out))

            ffn_in = x
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            x = self.norm2(x + self.dropout(ff_out))
        else:
            # Pre-norm (default): norm before sublayer, residual after.
            attn_out = self.attention(self.norm1(x), ve)
            if self.use_layerscale:
                attn_out = attn_out * (1.0 + self.attn_layerscale)
            x = x + self.dropout(attn_out)

            ffn_in = self.norm2(x)
            if self.use_ffn_embed and ve is not None:
                ffn_in = ffn_in + F.linear(ve, self.ffn_embed_proj)
            ff_out = self.feed_forward(ffn_in)
            if self.use_layerscale:
                ff_out = ff_out * (1.0 + self.ffn_layerscale)
            x = x + self.dropout(ff_out)
        return x
