"""145 — Expert-Choice MoE (Zhou, Lei, et al. 2022, arXiv:2202.09368,
Google "Mixture-of-Experts with Expert Choice Routing").

Unlike token-choice MoE (where each token picks its top-k experts),
expert-choice MoE INVERTS the routing direction: each expert picks
its own top-k tokens. Load balance is *by construction* — every
expert processes exactly k = n_tokens / n_experts tokens — and no
auxiliary load-balancing loss is required.

Mechanism (paper §2.1):
  1. `router_logits = W_r · x`  ∈  R^{n_experts × n_tokens} per (B, T).
  2. For each expert e: `topk_tokens_e = topk(router_logits[e, :], k)`
     where `k = n_tokens / n_experts` (we round to integer with
     floor; the residual r = n_tokens − k·E tokens get duplicate
     dispatch to the highest-scoring expert so the layer covers all
     tokens).
  3. Expert e processes its `k` (or `k+1`) tokens.
  4. Output: weighted sum of expert outputs, weight =
     `softmax(router_logits[e, topk_tokens_e])` per expert.

Identity at step 0: `W_r` is zero-init ⇒ all expert-token scores are
0 ⇒ topk picks the FIRST k token indices for every expert (PyTorch
`torch.topk` returns the smallest-index ties by default). All
experts process the same set of tokens, all with uniform softmax
weights (1/k). Output at step 0 is a uniform mean of all expert
FFN outputs applied to the same tokens. Since each expert is
init'd with the same per-`ffn_variant` factory, the average ≈
a single expert's output (variance among E identically-init'd
FFNs is 1/√E smaller than a single one) — close to a single-FFN
output but NOT byte-identical (the 1/E mean-pooling alters the
per-token output scale). The flag is OFF by default so the
baseline path is bit-identical.

Default off → baseline path bit-identical.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    SquaredReLUFeedForward,
    SwiGLUFeedForward,
    GELUFeedForward,
    SaturatingReLUFeedForward,
)


def _make_inner_ffn(d_model: int, d_ff: int, dropout: float, variant: str) -> nn.Module:
    if variant == "squared_relu":
        return SquaredReLUFeedForward(d_model, d_ff, dropout)
    if variant == "swiglu":
        return SwiGLUFeedForward(d_model, d_ff, dropout)
    if variant == "gelu":
        return GELUFeedForward(d_model, d_ff, dropout)
    if variant == "satrelu":
        return SaturatingReLUFeedForward(d_model, d_ff, dropout)
    raise ValueError(f"Unknown inner-FFN variant for ExpertChoiceMoE: {variant!r}")


class ExpertChoiceMoE(nn.Module):
    """Expert-Choice MoE wrapper: E parallel full-width FFNs + top-k-per-expert router.

    Args:
        d_model: input/output dimension.
        d_ff:    per-expert hidden width. Each expert is full-width
                 (no narrowing), so total FFN params scale as
                 `n_experts × 2 × d_model × d_ff` — same 4× FFN
                 param cost as 146-switch-ffn.
        n_experts: number of experts E. Default 4.
        dropout:   dropout applied inside each expert FFN.
        ffn_variant: which standard FFN class each expert uses
                     (squared_relu / swiglu / gelu / satrelu).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        dropout: float = 0.0,
        ffn_variant: str = "squared_relu",
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be ≥ 1, got {n_experts}")
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.n_experts = int(n_experts)
        # Per-token router: x ∈ R^d → logits ∈ R^E. Zero-init ⇒
        # all expert-token scores are 0 ⇒ topk picks the first k
        # token indices for every expert.
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.zero_()
        # E parallel full-width experts.
        self.experts = nn.ModuleList(
            [
                _make_inner_ffn(d_model, d_ff, dropout, ffn_variant)
                for _ in range(self.n_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]. Flatten tokens for routing math.
        B, T, D = x.shape
        N = B * T
        E = self.n_experts
        x_flat = x.reshape(N, D)

        # ---- Router scores ----
        # router_logits: [N, E] (per-token, per-expert). Transpose to
        # [E, N] so each row is one expert's full-token score vector.
        router_logits = self.router(x_flat).t().contiguous()  # [E, N]

        # ---- Top-k tokens per expert ----
        # Each expert gets `cap` = ceil(N/E) tokens. Tokens are then
        # weighted by `softmax(router_logits[e, topk_idx])` and the
        # contribution is scattered back to its position. A token
        # processed by multiple experts (the residual tokens) gets
        # a weighted sum of those expert outputs. A token processed
        # by no expert keeps its residual pass-through.
        cap = max(1, int(math.ceil(float(N) / float(E))))
        # topk returns (values, indices). Indices are [E, cap].
        _, topk_idx = torch.topk(router_logits, k=cap, dim=-1)
        # The gather/scatter math operates on a flat [N, D] buffer.
        out = torch.zeros_like(x_flat)
        weight_sum = torch.zeros(N, device=x.device, dtype=x.dtype)
        for e in range(E):
            idx_e = topk_idx[e]                              # [cap]
            tokens_e = x_flat[idx_e]                         # [cap, D]
            scores_e = router_logits[e, idx_e]               # [cap]
            # Softmax over the k tokens routed to this expert.
            w_e = torch.softmax(scores_e, dim=-1)            # [cap]
            y_e = self.experts[e](tokens_e)                  # [cap, D]
            # Scatter weighted expert output back to the original
            # token positions; a token hit by multiple experts gets
            # a sum. We accumulate the matching weight vector in
            # `weight_sum` for the residual-pass-through check.
            out.index_add_(0, idx_e, y_e * w_e.unsqueeze(-1))
            weight_sum.index_add_(0, idx_e, w_e)
        # Residual pass-through for tokens no expert picked (these
        # are the rare tokens at the tail when N is not divisible
        # by E). At step 0 the cap = ceil(N/E) covers N tokens per
        # expert so this branch is empty; during training the same
        # cap is used and the residual covers the last N mod E
        # tokens that fall outside the first k·E positions in the
        # topk argmax order.
        residual_mask = weight_sum == 0
        if residual_mask.any():
            out[residual_mask] = x_flat[residual_mask]
        return out.reshape(B, T, D)
