"""FIRE positional encoding (Li et al., NeurIPS 2023, arXiv:2306.02613).

Adds a content-aware position bias to attention logits:
    bias(t, s) = γ(|t-s|) · f([φ(x_t); φ(x_s)])

where
  - γ is a fixed Lp-norm kernel, monotone decay in relative distance,
  - φ is a per-head linear projection of the input (small-init, std=0.02),
  - f is a per-head linear from concat(φ_t, φ_s) → scalar (zero-init).

Identity at step 0: φ has small init (non-zero), but f is zero-init, so
the bias is exactly 0 → softmax unchanged from the no-bias baseline.
Default off (`use_fire_pe=False` in `LLMConfig`) — the module is built
unconditionally but its forward is never called.

v1 uses the score-only formulation to avoid materializing the
[B, H, T, T, 2·d_phi] pair tensor (which would be ~3.2 GB at T=2048).
Reformulation: f([φ_t; φ_s]) = W_t·φ_t + W_s·φ_s (linear, no hidden
layer). The bias is `γ · (W_t·φ_t + W_s·φ_s)` — computed as
`(W_t·φ).unsqueeze(3) + (W_s·φ).unsqueeze(2)`, then multiplied by γ.
Same math, O(B·H·T·d_phi) memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FIREBias(nn.Module):
    """FIRE positional bias — additive logit bias, [B, H, T, T].

    Parameters
    ----------
    d_model : int
        Input embedding dim.
    n_heads : int
        Number of attention heads; the bias is per-head.
    max_seq_len : int
        Max T — fixes the kernel buffer length.
    d_phi : int
        Dim of the per-head content projection (default 4).
    p : float
        Exponent on the Lp-norm kernel (default 1.0 → linear decay).

    Forward
    -------
    x : [B, T, d_model]
    Returns bias : [B, H, T, T] — additive on attention logits.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 d_phi: int = 4, p: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_phi = d_phi
        # Per-head content projection φ: d_model → d_phi. Small-init
        # (std=0.02) so the init distribution is similar to a Linear
        # in this codebase.
        self.phi = nn.Parameter(torch.empty(n_heads, d_phi, d_model))
        nn.init.normal_(self.phi, mean=0.0, std=0.02)
        # Per-head "MLP" f: 2·d_phi → 1. Stored as two slices
        # (W_t, W_s) of length d_phi each. ZERO-INIT so step-0 bias
        # is exactly 0 regardless of φ (identity-safe).
        self.f_w_t = nn.Parameter(torch.zeros(n_heads, d_phi))
        self.f_w_s = nn.Parameter(torch.zeros(n_heads, d_phi))
        # Fixed kernel γ: γ[d] = (1 - d/d_max)^p, clamped at 0.
        idx = torch.arange(max_seq_len, dtype=torch.float32)
        d_max = float(max_seq_len)
        gamma = (1.0 - idx / d_max).clamp_min(0.0).pow(p)
        self.register_buffer("gamma", gamma, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]. Returns [B, H, T, T].
        B, T, _ = x.shape
        H, D_phi, D_in = self.phi.shape
        # Per-head content projection: [B, T, H, d_phi]
        phi_x = torch.einsum('btd,hed->bthe', x, self.phi)
        # Split "MLP" into two halves: bias = W_t·φ_t + W_s·φ_s.
        #   W_t·φ_t: [B, T, H] = einsum('bthe,he->bth', phi_x, f_w_t)
        #   W_s·φ_s: [B, T, H] = einsum('bthe,he->bth', phi_x, f_w_s)
        term_t = torch.einsum('bthe,he->bth', phi_x, self.f_w_t)  # [B, T, H]
        term_s = torch.einsum('bthe,he->bth', phi_x, self.f_w_s)  # [B, T, H]
        # Combine: [B, H, T, T] = term_t[:, t, h, None] + term_s[:, s, h, None]
        # (broadcast over the other axis)
        bias = term_t.permute(0, 2, 1).unsqueeze(-1) + term_s.permute(0, 2, 1).unsqueeze(2)
        # bias: [B, H, T, T] (term_t broadcasts over s, term_s over t)
        # Multiply by γ(|t-s|).
        idx = torch.arange(T, device=x.device)
        diff = (idx[:, None] - idx[None, :]).abs()  # [T, T], values in [0, T-1]
        gamma = self.gamma[diff]  # [T, T]
        return bias * gamma[None, None]  # broadcast over B, H
