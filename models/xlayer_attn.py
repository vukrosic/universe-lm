"""150 — Cross-Layer Feedback Attention (Holtzman et al. 2020, Feedback
Transformer; Fan, Lavie, et al. 2020, "Reducing Transformer Depth on
Demand with Structured Dropout").

Each transformer block reads from a small cache of the previous K
layers' pre-FFN residual states via a tiny cross-attention head, and
adds the result as a *gated* residual branch. The gate is per-block
`xlayer_gate = nn.Parameter(torch.zeros(1))` (set in
`TransformerBlock`), so at step 0 the cross-attn contribution is
multiplied by 0 and the forward is bit-identical to the no-feedback
baseline. The cross-attn head is intentionally small (1 head, head_dim
16) to keep the per-block param overhead at ~8 KB at tiny1m3m
(d_model=64): Q/K projections are d_model × head_dim, V is d_model ×
d_model, out is d_model × d_model — total = 2·64·16 + 64² + 64² ≈
10.2K params/block. Cost at tiny1m3m (12 blocks): ~120K params, ≈ 13%
of the 0.94M model. The small head keeps the attention map compact
(K·T ≤ 4096 for K=2, T=2048) and avoids the cost of multi-head cross-
attn over a 2x-4x longer sequence.

Identity at step 0: the block's `xlayer_gate` is 0, so the cross-attn
output is multiplied by 0 regardless of the Q/K/V projection values.
This guarantees step-0 ≡ baseline.

Causality: the K, V tensor comes from previous blocks' pre-FFN states
at the SAME token positions, so cross-attn does NOT need a causal
mask on the K axis. A position can attend to all K·T positions in the
mem — they're all "past" in the layer dimension.

Distinct from value-residual (021): value-residual blends a single
V tensor from layer 0 into the post-W_V stream of every later layer
(`V_l ← (1-λ)·V_l + λ·V_0`). The lever is "share V across the
depth axis." Cross-Layer Feedback is "attend over a small window of
hidden states from the last K layers." Different operating point:
cross-attn with a *gate* (selection) vs linear V blending (mixing).
The closest null in the closed list is 116-hyper-connections (mHC)
— mHC mixes adjacent-layer outputs linearly (no attention, no
selection). Cross-Layer Feedback mixes via attention, with a learnable
selection mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class XLayerCrossAttn(nn.Module):
    """Cross-layer attention: Q from current x, K/V from a mem tensor
    (a list of K previous layer pre-FFN states concatenated along the
    time axis).

    Single-head cross-attention with a small head_dim to keep params
    small. The block-level gate (set in `TransformerBlock`) is the
    identity lever: at step 0 the gate is 0 so the cross-attn
    contribution vanishes exactly.

    Args:
        d_model: channel dim of the residual stream (B, T, d_model).
        k_window: number of previous layers to read from (K in the
            idea doc). Default 2 (the idea's spec pin).
        n_heads: number of cross-attn heads. Default 1 (the idea's
            spec pin — small head keeps params/attn-map compact).
        head_dim: per-head dim. Default 16. The output dim of cross-
            attn is `n_heads * head_dim`, then projected back to
            `d_model` via `out_proj`. If `n_heads * head_dim != d_model`,
            `out_proj` still maps `n_heads * head_dim → d_model`
            (no extra reshape is needed for the simple n_heads=1 case).

    Forward:
        x: [B, T, d_model] — current block's pre-FFN x (queries).
        mem: Optional[List[Tensor]] — list of up to K previous
            blocks' pre-FFN states, each [B, T, d_model]. If `mem` is
            `None` or empty, returns a zero tensor of shape `x` (the
            block's gate will multiply by 0 anyway, so the result
            is a no-op in the residual stream).

        Returns: [B, T, d_model] = xlayer_attn(x, mem).
    """

    def __init__(self, d_model: int, k_window: int = 2,
                 n_heads: int = 1, head_dim: int = 16):
        super().__init__()
        self.d_model = int(d_model)
        self.k_window = max(1, int(k_window))
        self.n_heads = max(1, int(n_heads))
        self.head_dim = max(1, int(head_dim))
        # Total Q/K/V dim per head = head_dim. Total across heads =
        # n_heads * head_dim. Q, K, V are all small projections to
        # this same dim; the output proj maps it back to d_model.
        # For the spec pin n_heads=1, head_dim=16 at d_model=64: this
        # is 64→16 for Q/K/V (much smaller than d_model→d_model)
        # plus a 16→64 out projection.
        qk_dim = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(self.d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, qk_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, qk_dim, bias=False)
        # Out: qk_dim → d_model. With n_heads=1, head_dim=16 at
        # d_model=64, this is a small 16→64 projection.
        self.out_proj = nn.Linear(qk_dim, self.d_model, bias=False)
        # Standard init: normal std=0.02 (matches the rest of the
        # codebase's `_init_weights`). The block-level gate is 0, so
        # the random init doesn't matter for step 0.
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor,
                mem: list | None) -> torch.Tensor:
        # x: [B, T, d_model]; mem: list of [B, T, d_model] (up to K).
        if mem is None or len(mem) == 0:
            # First blocks in the stack have no mem yet. Return zeros —
            # the block's gate is 0 at step 0 so the contribution is
            # already a no-op, and as mem fills in the cross-attn will
            # start producing real outputs that the gate scales.
            return torch.zeros_like(x)
        # Concatenate the previous K layers' pre-FFN states along T.
        # M: [B, K*T, d_model]. K is at most k_window (the model loop
        # truncates to the last K).
        M = torch.cat(mem, dim=1)
        # Q from x: [B, T, qk_dim]. K, V from M: [B, K*T, qk_dim].
        Q = self.q_proj(x)
        K = self.k_proj(M)
        V = self.v_proj(M)
        # Reshape to multi-head layout. [B, n_heads, T, head_dim].
        B, T, _ = Q.shape
        Q_h = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K_h = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V_h = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # Attention: softmax(QKᵀ / √head_dim) · V.
        # No causal mask — mem is from previous blocks at the same
        # positions, all valid.
        scores = (Q_h @ K_h.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # attn @ V_h -> [B, n_heads, T, head_dim]. Merge heads back to
        # [B, T, n_heads * head_dim] = [B, T, qk_dim].
        out_h = attn @ V_h
        out = out_h.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)
