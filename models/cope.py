"""CoPE — Content-aware Positional Encoding (Golovneva et al. 2024,
arXiv:2405.18719, Meta).

A drop-in alternative to RoPE. Position offset between query i and
key j is the count of "important" tokens (those with high content-
dot-product to a learned per-head probe) in [j, i], not the literal
index distance. This makes the position signal content-conditional —
a different inductive bias from RoPE's absolute-position decay.

Mechanism (soft sigmoid variant, τ=0 pinned per plan):
    score[b,t,h] = x[b,t,:] · p[h,:]                    # [B, T, H]
    g[b,t,h]     = sigmoid(score[b,t,h] - τ)            # gate
    cum_g[b,t,h] = cumsum_t g[b,t,h]                    # prefix sum
    offset[b,h,i,j] = cum_g[b,i,h] - cum_g[b,j-1,h]     # count in [j, i]
                                                      # (cum_g[:, -1, :] := 0)
    attention_scores += offset

The offset is added to attention logits just like FIRE or ALiBi.

Identity at step 0: with probe init N(0, 0.02), score ~ N(0, 0.16) for
d_model=64 (tiny1m3m), so g ≈ 0.5 ± 0.04 — the offset is roughly
(i - j + 1) * 0.5 (linear in distance, RoPE-like). NOT bit-identical
to no-CoPE; the A/B is a real lever (matches the paper).

Default off (`use_cope=False` in `LLMConfig`) — the module is built
unconditionally when the flag is on, but its forward is never called
when the flag is off.
"""
import torch
import torch.nn as nn


class CoPE(nn.Module):
    """CoPE — content-aware positional bias, [B, H, T, T].

    Parameters
    ----------
    d_model : int
        Input embedding dim.
    n_heads : int
        Number of attention heads; the bias is per-head.
    max_seq_len : int
        Kept for API consistency with FIRE/Rotary (not used in forward —
        the cumsum is bounded by the runtime T).
    threshold : float
        Sigmoid threshold τ (default 0.0 = midpoint, pinned per plan.md).

    Forward
    -------
    x : [B, T, d_model]
    Returns bias : [B, H, T, T] — additive on attention logits.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 threshold: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.threshold = float(threshold)
        # Per-head probe p: shape [n_heads, d_model]. Init N(0, 0.02) —
        # mirrors FIRE's per-head content projection init in
        # `models/fire_pe.py:60` (`nn.init.normal_(self.phi, mean=0.0,
        # std=0.02)`). Probe is the *only* learned param; threshold τ
        # is pinned at 0.0 (one-seed-only rule forbids the τ sweep).
        self.probe = nn.Parameter(torch.empty(n_heads, d_model))
        nn.init.normal_(self.probe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]. Returns [B, H, T, T].
        B, T, _ = x.shape
        H = self.n_heads
        # Per-head importance score: [B, T, H].
        # einsum over (b,t,d) × (h,d) → (b,t,h).
        score = torch.einsum("btd,hd->bth", x, self.probe)
        # Soft sigmoid gate (τ pinned at 0.0).
        g = torch.sigmoid(score - self.threshold)  # [B, T, H]
        # Cumulative sum over the position axis: [B, T, H].
        cum_g = torch.cumsum(g, dim=1)
        # Prepend a zero so that offset[i, 0] = cum_g[i] - 0 = cum_g[i].
        zero = torch.zeros(B, 1, H, device=x.device, dtype=x.dtype)
        cum_g_pad = torch.cat([zero, cum_g], dim=1)  # [B, T+1, H]
        # offset[b, h, i, j] = cum_g_pad[b, i+1, h] - cum_g_pad[b, j, h]
        # = cum_g[b, i, h] - cum_g[b, j-1, h] (count in [j, i]).
        right = cum_g_pad[:, 1:, :].unsqueeze(2)  # [B, T, 1, H]
        left = cum_g_pad[:, :-1, :].unsqueeze(1)  # [B, 1, T, H]
        offset = (right - left).permute(0, 3, 1, 2)  # [B, H, T, T]
        return offset
