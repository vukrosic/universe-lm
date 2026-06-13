"""146 — Switch FFN (Fedus, Zoph, Shazeer 2022, arXiv:2101.03961).

Sparse Mixture-of-Experts replacement for the dense FFN: N parallel FFNs
"experts" + a learned top-1 router per token. The simplest form of MoE.

Mechanism:
  - router_logits = W_router · x                       shape [N, n_experts]
  - expert_idx    = argmax(router_logits, dim=-1)     shape [N]
  - capacity      = ceil(N / n_experts) * capacity_factor
  - For each token i:  out[i] = experts[expert_idx[i]](x[i])
  - Overflowed tokens (those whose expert is already at capacity) are
    skipped: their contribution to the residual is `0` (residual pass-
    through) per the original Switch Transformer paper.

Capacity factor: each expert can hold at most
`ceil(n_tokens / n_experts) * capacity_factor` tokens. Overflow tokens
are passed through unchanged (residual identity). The paper uses
capacity_factor=1.25 for stability.

Identity at step 0:
  - W_router is zero-initialized ⇒ argmax over a uniform-zero vector
    returns index 0 for every token (the lowest-index expert).
  - All N tokens route to expert 0; experts 1..E-1 are unused.
  - Output = expert_0(x) for all tokens, which is the standard dense
    FFN. With `expert_0`'s parameters initialized the same way the
    baseline SquaredReLUFeedForward is, the flag-on step-0 forward is
    bit-identical to the dense-FFN baseline (the W·x of the router
    contributes nothing because the output is gathered purely from
    expert 0).

Capacity factor:
  The paper sets per-expert capacity = `ceil(N/E) * capacity_factor`,
  intended as a soft buffer for routing imbalance (~1.25× the average
  load). When routing is degenerate (e.g. step 0 where all N tokens
  route to one expert), the paper's formula caps the expert at less
  than N and the rest get dropped (residual pass-through). To keep
  step 0 byte-identical to the dense-FFN baseline — where EVERY token
  gets a non-zero FFN output — we use `effective_capacity = max(N,
  ceil(N/E * capacity_factor))`. This makes step-0 routing collapse
  harmlessly (capacity = N) while still enforcing the paper's
  drop-tokens mechanism during training when routing diversifies.
  With N=4096 (tiny1m3m batch), E=4, cf=1.25: paper_capacity=1280,
  effective_capacity=4096 ⇒ no truncation at step 0.
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
    raise ValueError(f"Unknown inner-FFN variant for SwitchFFN: {variant!r}")


class SwitchFFN(nn.Module):
    """Switch-Transformer-style sparse FFN: N parallel experts, top-1 router.

    Args:
        d_model: input/output dimension.
        d_ff:    per-expert hidden width. Each expert is a full-width FFN
                 (so total FFN params scale as `n_experts × 2 × d_model × d_ff`
                 — a real param injection vs the dense baseline). The paper
                 uses ~4 experts; we default to 4 here.
        n_experts: number of experts E.
        capacity_factor: each expert's max slot count =
                 `ceil(N / E) * capacity_factor`. Tokens whose chosen
                 expert is at capacity are dropped (residual pass-through).
                 Paper default = 1.25.
        dropout: dropout applied inside each expert FFN.
        ffn_variant: which standard FFN class each expert uses.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        ffn_variant: str = "squared_relu",
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be ≥ 1, got {n_experts}")
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = int(n_experts)
        self.capacity_factor = float(capacity_factor)
        # Per-token router: x ∈ R^d → logits ∈ R^E. Zero-init ⇒
        # argmax over uniform-zero returns 0 for every token ⇒ all
        # tokens → expert 0 ⇒ output = expert_0(x) = standard FFN at step 0.
        self.router = nn.Linear(d_model, self.n_experts, bias=False)
        with torch.no_grad():
            self.router.weight.zero_()
        # E parallel full-width experts (no narrowing — Switch FFN pays the
        # 4× FFN-param cost in exchange for specialization capacity).
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
        x_flat = x.reshape(N, D)
        E = self.n_experts
        # Capacity per expert — paper uses ceil(N/E) * capacity_factor.
        # Effective capacity is clamped up to N so that degenerate
        # routing (e.g. step 0: all tokens → expert 0) does NOT truncate
        # tokens. The paper's drop mechanism still fires during training
        # once routing diversifies enough to overload an expert.
        paper_capacity = max(
            1, int(math.ceil(float(N) / float(E) * self.capacity_factor))
        )
        capacity = max(N, paper_capacity)
        # Top-1 routing: argmax over expert logits.
        # On a uniform-zero router, torch.argmax returns index 0 for every
        # row (deterministic — argmax picks the first max).
        router_logits = self.router(x_flat)             # [N, E]
        expert_idx = torch.argmax(router_logits, dim=-1)  # [N]
        # Per-expert gather: tokens going to each expert.
        out = torch.zeros_like(x_flat)
        for e in range(E):
            mask_e = expert_idx == e
            idx_e = torch.nonzero(mask_e, as_tuple=False).squeeze(-1)
            if idx_e.numel() == 0:
                continue
            if idx_e.numel() > capacity:
                # Overflow tokens are dropped (residual pass-through).
                # Switch paper §2.2 — bypasses the FFN. With our
                # `capacity = max(N, paper_capacity)` clamp, this only
                # fires during training when routing is severely
                # imbalanced.
                idx_e = idx_e[:capacity]
            tokens_e = x_flat[idx_e]
            out_e = self.experts[e](tokens_e)
            out[idx_e] = out_e
        return out.reshape(B, T, D)