"""Forgetting Transformer (FoX) — softmax attention with a forget gate
(Lin, Wang, Yang et al., March 2025, arXiv:2503.02130).

A *conservative extension* of softmax attention. Per-head, per-token
sigmoid forget gate contributes a causal cumulative log-decay to the
attention logits, so the softmax sees `QK^T/√d + log_D`. This is the
paper's logit-add formulation — mathematically equivalent to the
multiply-after-softmax notation in the original paper sketch, and
numerically stable.

Mechanism (logit-add form, as implemented)
------------------------------------------
    f_h,t   = sigmoid(W_f^h · x_t + b_f^h)         # ∈ (0, 1)
    log_f_h = logsigmoid(z)                         # [B, T, H]  ≤ 0
    cum_h   = cumsum(log_f_h, dim=T)                # [B, T, H]
    log_D[b,h,i,j] = cum_h[b,h,i] - cum_h[b,h,j-1]  for j ≥ 1
    log_D[b,h,i,0] = cum_h[b,h,i]                   boundary case
    A_h     = softmax(Q_h K_h^T / √d + log_D_h)
    y_h     = A_h V_h

For j ≤ i:  log_D[i,j] = sum_{k=j..i} log_f[k] ≤ 0  (the standard
            cumulative-product decay over the kept window).
For j > i:  the right-minus-left difference is ≥ 0 (wrong sign), so we
            mask the upper-tri to 0 here. The caller's causal mask
            (-1e9) dominates regardless, but a hygienic 0 in the upper
            triangle keeps callers free to inspect the tensor.

Identity init
-------------
b_f^h = +10 → f = sigmoid(10) ≈ 0.99995 → log f ≈ −4.54e-5. Over
max_seq_len = 2048, log_D[0, 2047] = 2047 · log f ≈ −0.093, so
softmax(scores + log_D) is within ≈ 9% of softmax(scores) at the
worst-case (query 0, key 2047) row position. W_f^h = 0 means the gate
has no input-dependence at init. After softmax on a roughly-uniform
attention map, attention output is within ~1e-2 of the no-FoX baseline.
The model still has to *learn* to forget from scratch; gates start
nearly 1.0 and can only go down.

Why log-add (recode r2)
-----------------------
The r1 implementation did `attn_w = softmax(scores); attn_w *= D;
attn_w /= sum`. At step ~400 in tiny1m3m training, the gate had moved
off its `b_f=+10` init enough for `D[i, j]` to underflow at j ≪ i
(e.g. once `log f ≈ −0.01`, `D[0, 2047] = exp(−20) ≈ 2e-9`). The
row-renorm then divided ≈0 / ≈0 → NaN. Replacing the post-softmax
multiply with a pre-softmax `scores += log_D` is exactly equivalent
(softmax(s) ⊙ d / sum = softmax(s + log d)) and bypasses underflow
entirely — softmax's max-subtraction handles arbitrarily negative
log_D. See `autoresearch/ideas/020-forgetting-attn/evidence.md` for
the NaN diagnosis.

Default off (`use_fox=False` in `LLMConfig`) — the module is built
unconditionally when the flag is on; its forward is never called
otherwise.
"""
import torch
import torch.nn as nn


# Init bias for the per-head forget gate. +10 → f = sigmoid(10) ≈ 0.99995
# → log f ≈ −4.54e-5 → log_D[0, 2047] = −0.093 at T=2048 (≤ 9% worst-case
# decay on the softmax output over the full context). Pinned by the
# plan; do not change without re-deriving the identity-init math in
# plan.md §(e).
FOX_BF_INIT: float = 10.0


class FoX(nn.Module):
    """FoX — per-head learnable forget-gate kernel.

    Returns the *log* of the decay matrix so the caller can add it to
    attention scores before softmax (numerically stable). The pre-softmax
    add `scores += log_D` is mathematically equivalent to the
    multiply-after-softmax `attn = (softmax(scores) ⊙ D) / row_sum`
    form in the original paper sketch.

    Parameters
    ----------
    d_model : int
        Input embedding dim (the per-head gate is a Linear(d_model → H)).
    n_heads : int
        Number of attention heads.
    b_f : float
        Init bias for the per-head gate (default `FOX_BF_INIT` = +10).
        W_f is zero-init regardless.

    Forward
    -------
    x : [B, T, d_model]
    Returns log_D : [B, H, T, T] — pre-softmax additive bias on logits.
    Upper triangle (j > i) is masked to 0 for hygiene; the caller's
    causal mask sets those positions to -1e9 before softmax anyway.
    Lower triangle is ≤ 0 (the standard cumulative log-decay).
    """

    def __init__(self, d_model: int, n_heads: int, b_f: float = FOX_BF_INIT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.b_f = float(b_f)
        # Per-head gate projection: d_model → H. W is zero-init so the
        # input-dependent term is 0 at step 0; b_f carries the entire
        # gate value at init (a constant per head → constant cum_log_f
        # → smoothly-decaying log_D). Stored as a raw parameter (not
        # nn.Linear) so the call site is a single einsum with no bias
        # handling: gate_logits = einsum("btd,hd->bth", x, W).
        self.gate_w = nn.Parameter(torch.zeros(n_heads, d_model))
        # Per-head bias buffer (n_heads,). b_f = +10 → f ≈ 0.99995 at
        # init, log_D ≈ 0 within ~0.09 over the full context. Buffer
        # (not Parameter) because it's a fixed init constant; the only
        # learnable contribution is the per-token input-dependent term.
        self.register_buffer("gate_b", torch.full((n_heads,), self.b_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]. Returns log_D: [B, H, T, T] (lower-tri ≤ 0,
        # upper-tri = 0).
        B, T, _ = x.shape
        H = self.n_heads
        # Per-head gate logit: z[b,t,h] = x[b,t,:] · W[h,:] + b[h].
        # W is zero-init, so the x·W term is 0; the bias dominates.
        z = torch.einsum("btd,hd->bth", x, self.gate_w) + self.gate_b.view(1, 1, H)
        # f ∈ (0, 1); log_f = logsigmoid(z) ≤ 0 (numerically stable for
        # any z, unlike `log(sigmoid(z))`).
        log_f = torch.nn.functional.logsigmoid(z)  # [B, T, H]
        # Cumulative sum over the position axis: cum[b, t, h] =
        # sum_{s=0..t} log_f[b, s, h]. Bounded by T * log f_min (≤ 0).
        cum = torch.cumsum(log_f, dim=1)  # [B, T, H]
        # Prepend a zero along T so log_D[i, 0] = cum[i] - 0 = cum[i].
        # Shape [B, T+1, H].
        zero = torch.zeros(B, 1, H, device=x.device, dtype=cum.dtype)
        cum_pad = torch.cat([zero, cum], dim=1)
        # log_D[b, h, i, j] = cum_pad[b, i+1, h] - cum_pad[b, j, h]
        #   = cum[b, i, h] - cum[b, j-1, h]   for j ≥ 1
        #   = cum[b, i, h]                     for j = 0
        # Build via two slices on the T+1 axis (so j runs 0..T-1).
        right = cum_pad[:, 1:, :].unsqueeze(2)  # [B, T, 1, H]
        left = cum_pad[:, :-1, :].unsqueeze(1)  # [B, 1, T, H]
        log_d = right - left  # [B, T, T, H]
        # Move heads to dim 1: [B, H, T, T].
        log_d = log_d.permute(0, 3, 1, 2)
        # Mask upper-tri to 0 (hygiene — the right-minus-left difference
        # is ≥ 0 there, which is the wrong sign and would corrupt the
        # decay if anyone added it to unmasked logits). The caller's
        # causal mask sets those positions to -1e9 before softmax, so
        # the value of log_d in the upper triangle is moot for our use
        # site; we zero it so the tensor is safe to inspect / reuse.
        ar = torch.arange(T, device=x.device)
        causal = (ar[None, :] <= ar[:, None]).view(1, 1, T, T)
        log_d = log_d.masked_fill(~causal, 0.0)
        return log_d
