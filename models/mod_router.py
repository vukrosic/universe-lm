"""118 — Mixture-of-Depths (MoD) per-token router.

Raposo, Ritter, Richards, Lajoie, Sharma, "Mixture-of-Depths:
Dynamically allocating compute in transformer inference"
(arXiv:2404.02258, April 2024).
https://arxiv.org/abs/2404.02258

Mechanism (paper §2):
  For each transformer block, a small per-token MLP scores whether the
  token should go through the block. The top-k tokens (by router score)
  are routed through the block; the rest skip the block via an identity
  residual `x_t ← x_t`. The block's residual contribution is rescaled
  by `c = k/T` so the expected per-token residual magnitude matches the
  dense baseline.

Concretely at each block `l`:
    scores = σ(W_2 · σ(W_1 · x))     # [B, T]
    top_k_idx = topk(scores, k = cap · T)
    delta = block(x)[top_k_idx] - x[top_k_idx]
    x[top_k_idx]  ← x[top_k_idx]  + c · delta
    x[~top_k_idx] ← x[~top_k_idx]   # skip for the rest

Identity at step 0: W_1, W_2 zero-init ⇒ σ(0) = 0.5 for every token.
Top-k on a uniform 0.5 score selects an arbitrary k-token subset — no
signal, no preference. The block is applied to the whole stream (no
real compute savings at the implementation level — the lever is the
*quality* effect of per-token routing, not the FLOP count), and the
residual update is gated to the random top-k. The expected residual
contribution per token is `(k/T) · E[block(x)] = 0.5 · E[block(x)]`,
matching the average residual contribution but with per-token variance.

NOT byte-identical to the dense baseline at step 0 (the baseline applies
the block to *every* token), but the deviation is bounded and explicit:
see `autoresearch/ideas/118-mixture-of-depths/idea.md`. With
`use_mod=False` (default) the router module is never built and the
baseline forward graph is bit-identical to the pre-norm residual path.

Cost: ~d_model · hidden + hidden · 1 params per block. With
mod_router_hidden=64 at tiny1m3m (d_model=64) → 64·64+64·1 = 4,160
params/block × 12 = ~50k (~5% of the 0.94M budget; tiny for a screen).
"""

import torch
import torch.nn as nn


class MoDRouter(nn.Module):
    """Per-token router for MoD: scores = σ(W_2 · σ(W_1 · x)).

    Args:
        d_model: input/output dim of the residual stream.
        hidden:  router MLP hidden width.

    Returns:
        scores: tensor of shape `[B, T]` in `[0, 1]`.
    """

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        # 2-layer MLP. Zero-init both layers ⇒ σ(0) = 0.5 for every
        # token at step 0 ⇒ top-k selects an arbitrary subset.
        self.W1 = nn.Linear(d_model, hidden, bias=False)
        self.W2 = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.W1.weight)
        nn.init.zeros_(self.W2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → scores: [B, T] in [0, 1].
        h = torch.sigmoid(self.W1(x))            # [B, T, hidden]
        return torch.sigmoid(self.W2(h)).squeeze(-1)  # [B, T]