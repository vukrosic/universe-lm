"""Forgetting Transformer (FoX) — softmax attention with a forget gate
(Lin, Wang, Yang et al., March 2025, arXiv:2503.02130).

A *conservative extension* of softmax attention. Per-head, per-token
sigmoid forget gate multiplies the attention matrix element-wise, then
the rows are renormalized. The softmax itself is unchanged.

Mechanism
---------
    f_h,t   = sigmoid(W_f^h · x_t + b_f^h)       # ∈ (0, 1) per head, per token
    log_f_h = log(f_h,t)                          # [B, H, T]  (always ≤ 0)
    cum_h   = cumsum(log_f_h, dim=T)              # [B, H, T]   monotone non-incr
    D[b,h,i,j] = exp(cum_h[b,h,i] - cum_h[b,h,j-1])   for j ≥ 1
    D[b,h,i,0] = exp(cum_h[b,h,i])                     boundary case
    A_h  = softmax(Q_h K_h^T / √d) ⊙ D_h
    A_h' = A_h / A_h.sum(-1, keepdim=True)        # row-renorm
    y_h  = A_h' V_h

`D[i, j]` is a *learned* causal decay between query position `i` and key
position `j` — the model can choose to forget far tokens fast (small f_h)
or keep them (f_h → 1).

Identity init
-------------
b_f^h = +10 → f = sigmoid(10) ≈ 0.99995 → log f ≈ −4.54e-5. Over
max_seq_len = 2048, D[0, 2047] = exp(−0.093) ≈ 0.911 (≤ 9% decay over
the full context). W_f^h = 0 means the gate has no input-dependence at
init. After softmax + row-renorm on a roughly-uniform attention map,
attention output is within ~1e-2 of the no-FoX baseline (see plan.md
test (e)). The model still has to *learn* to forget from scratch; gates
start nearly 1.0 and can only go down.

NOTE: b_f = +5 (the r1 value in the closed-loop draft) is WRONG at
T=2048 — it gives D[0, 2047] = exp(−14) ≈ 1e-6, killing long-range
attention at init. b_f = +10 is the corrected value (see
`autoresearch/ideas/020-forgetting-attn/review.md` r1 BLOCKING-1).

Default off (`use_fox=False` in `LLMConfig`) — the module is built
unconditionally when the flag is on; its forward is never called
otherwise.
"""
import torch
import torch.nn as nn


# Init bias for the per-head forget gate. +10 → f = sigmoid(10) ≈ 0.99995
# → log f ≈ −4.54e-5 → D[0, 2047] = exp(−0.093) ≈ 0.91 at T=2048 (≤ 9%
# worst-case decay over the full context). Pinned by the plan; do not
# change without re-deriving the identity-init math in plan.md §(e).
FOX_BF_INIT: float = 10.0


class FoX(nn.Module):
    """FoX — per-head learnable forget-gate kernel, [B, H, T, T].

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
    Returns D : [B, H, T, T] — multiplicative on attention probabilities
    (post-softmax). The caller multiplies D into attn_w and re-normalizes
    rows.

    Causal convention: D[i, j] = 0 for j > i (i.e. the kernel is strictly
    lower-triangular). The cumulative log-gate is built from t=0..T-1;
    for j=0, the "prepend zero" trick gives D[i, 0] = exp(cum[i]) — the
    contribution of tokens 0..i to the running product, which is exactly
    what the cumulative decay should weigh.
    """

    def __init__(self, d_model: int, n_heads: int, b_f: float = FOX_BF_INIT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.b_f = float(b_f)
        # Per-head gate projection: d_model → H. W is zero-init so the
        # input-dependent term is 0 at step 0; b_f carries the entire
        # gate value at init (a constant per head → constant cum_log_f
        # → smoothly-decaying D). Stored as a raw parameter (not
        # nn.Linear) so the call site is a single einsum with no bias
        # handling: gate_logits = einsum("btd,hd->bth", x, W).
        self.gate_w = nn.Parameter(torch.zeros(n_heads, d_model))
        # Per-head bias buffer (n_heads,). b_f = +10 → f ≈ 0.99995 at
        # init, D ≈ 1 within 9% over the full context. Buffer (not
        # Parameter) because it's a fixed init constant; the only
        # learnable contribution is the per-token input-dependent term.
        self.register_buffer("gate_b", torch.full((n_heads,), self.b_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]. Returns D: [B, H, T, T] (causal lower-tri).
        B, T, _ = x.shape
        H = self.n_heads
        # Per-head gate logit: z[b,t,h] = x[b,t,:] · W[h,:] + b[h].
        # W is zero-init, so the x·W term is 0; the bias dominates.
        z = torch.einsum("btd,hd->bth", x, self.gate_w) + self.gate_b.view(1, 1, H)
        # f ∈ (0, 1); log_f ≤ 0 (causal-monotone product won't grow).
        log_f = torch.nn.functional.logsigmoid(z)  # [B, T, H]
        # Cumulative sum over the position axis: cum[b, t, h] =
        # sum_{s=0..t} log_f[b, s, h]. Bounded by T * log f_min (≤ 0).
        cum = torch.cumsum(log_f, dim=1)  # [B, T, H]
        # Prepend a zero along T so D[i, 0] = exp(cum[i] - 0) = exp(cum[i]).
        # Shape [B, T+1, H].
        zero = torch.zeros(B, 1, H, device=x.device, dtype=x.dtype)
        cum_pad = torch.cat([zero, cum], dim=1)
        # D[b, h, i, j] = exp(cum_pad[b, i+1, h] - cum_pad[b, j, h])
        #   = exp(cum[b, i, h] - cum[b, j-1, h])   for j ≥ 1
        #   = exp(cum[b, i, h])                     for j = 0
        # Build via two slices on the T+1 axis (so j runs 0..T-1).
        right = cum_pad[:, 1:, :].unsqueeze(2)  # [B, T, 1, H]
        left = cum_pad[:, :-1, :].unsqueeze(1)  # [B, 1, T, H]
        log_d = right - left  # [B, T, T, H]
        # Move heads to the second dim and exp; for j > i the log is
        # log_f[j-1] + ... + log_f[i] ≤ 0 so exp ≤ 1, but we explicitly
        # mask to 0 (the attention mask in MHA will already zero the
        # post-softmax A in the upper triangle, so the value of D there
        # is moot — but zero keeps the kernel strictly causal and
        # avoids any fp issue).
        d = log_d.permute(0, 3, 1, 2).exp()  # [B, H, T, T]
        ar = torch.arange(T, device=x.device)
        causal = ar[None, :] <= ar[:, None]  # [T, T]
        d = d * causal.view(1, 1, T, T).to(d.dtype)
        return d
