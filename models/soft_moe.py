"""117 — Soft MoE (Puigcerver et al. 2024, arXiv:2406.06589).

Fully-differentiable Mixture-of-Experts replacement for the FFN.

Mechanism (paper §2):
  - `X ∈ R^{N×d}` with N tokens.
  - Learn dispatch `D ∈ R^{N × m × E}` (softmax over input-token axis per (slot, expert))
    and combine `C ∈ R^{N × m × E}` (softmax over (slot, expert) axes per output token).
  - Dispatch:  `M_e[s, :] = Σ_i D[i, s, e] · X[i, :]` → shape (m, d) per expert.
  - Each expert `f_e` is a narrower FFN (`d_ff / n_experts` wide) — total FFN
    params stay at the baseline budget.
  - Combine:  `Y[i, :] = Σ_{s,e} C[i, s, e] · f_e(M_e)[s, :]` → shape (N, d).

Identity at step 0: `W_d, W_c` are zero-initialized; softmax(0) is uniform.
- `D[i, s, e] = 1/N` (each (slot, expert) sees an equal mixture of input tokens)
- `C[i, s, e] = 1/(m·E)` (each output token uses an equal mixture of expert outputs)
All `E` experts see the same input (`mean(X)`) and are init'd with the standard
FFN scheme; their outputs are statistically equivalent — so the layer collapses
to ~a single FFN applied to the mean of the input. NOT byte-identical to the
single-FFN baseline when flag is ON (the mean-over-tokens aggregation changes
the per-token output); with `use_soft_moe=False` the module is never built
and the baseline path is bit-identical.

Default off → baseline path bit-identical.
"""

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
    raise ValueError(f"Unknown inner-FFN variant for SoftMoE: {variant!r}")


class SoftMoEFFN(nn.Module):
    """Soft MoE wrapper: E narrower experts + softmax dispatch/combine.

    Args:
        d_model: input/output dimension.
        d_ff:    baseline FFN hidden width. Per-expert width is `d_ff // E`
                 (so total FFN params stay ≈ 2 · d_model · d_ff, matching
                 the single-FFN baseline).
        n_experts: number of experts E.
        n_slots:   number of slots per token m.
        dropout:   dropout applied to the final combined output.
        ffn_variant: which standard FFN class to use for each expert
                     (squared_relu / swiglu / gelu / satrelu). The Soft
                     MoE math is variant-agnostic — only the inner expert
                     changes.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        n_slots: int = 4,
        dropout: float = 0.0,
        ffn_variant: str = "squared_relu",
    ):
        super().__init__()
        if n_experts < 1:
            raise ValueError(f"n_experts must be ≥ 1, got {n_experts}")
        if n_slots < 1:
            raise ValueError(f"n_slots must be ≥ 1, got {n_slots}")
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = int(n_experts)
        self.n_slots = int(n_slots)
        # Per-expert width — narrowed so the total FFN cost is at the budget.
        # Floor at 1 to avoid a zero-width expert in pathological configs.
        self.d_ff_e = max(1, d_ff // self.n_experts)
        # Per-token projections that produce the m·E dispatch/combine logits.
        # Zero-init ⇒ softmax(0) is uniform ⇒ step-0 collapses to a single
        # (mean-X) FFN path. NOT byte-identical to the single-FFN baseline
        # (the aggregation changes the per-token output), but the deviation
        # is bounded (see module docstring).
        self.W_d = nn.Parameter(torch.zeros(self.n_experts * self.n_slots, d_model))
        self.W_c = nn.Parameter(torch.zeros(self.n_experts * self.n_slots, d_model))
        # E parallel narrower FFNs.
        self.experts = nn.ModuleList(
            [
                _make_inner_ffn(d_model, self.d_ff_e, dropout, ffn_variant)
                for _ in range(self.n_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model). Flatten tokens for the softmax/dispatch math.
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # ---- Dispatch ----
        # Per-token linear → (N, m·E) → reshape → softmax over input-token axis.
        # `softmax(..., dim=0)` makes D[i, s, e] sum to 1 over i for each (s, e).
        d_logits = F.linear(x_flat, self.W_d)            # (N, m·E)
        d_logits = d_logits.reshape(N, self.n_slots, self.n_experts)
        D_w = F.softmax(d_logits, dim=0)                 # (N, m, E)

        # M_e[s, :] = Σ_i D[i, s, e] · X[i, :]. Output shape (m, E, D).
        # Einsum: D_w_ise · x_flat_id → out_sed.
        M = torch.einsum("nse,nd->sed", D_w, x_flat)     # (m, E, D)

        # ---- Experts ----
        # Each expert e reads its (m, D) slice and produces (m, D).
        Y_all = torch.empty(
            self.n_slots, self.n_experts, D,
            device=x.device, dtype=x.dtype,
        )
        for e_idx, expert in enumerate(self.experts):
            Y_all[:, e_idx, :] = expert(M[:, e_idx, :])

        # ---- Combine ----
        # Per-token linear → (N, m·E) → softmax over the flattened (m·E) axis.
        c_logits = F.linear(x_flat, self.W_c)            # (N, m·E)
        # Flatten (m, E) → softmax over (m·E) → reshape back to (N, m, E).
        C_w = F.softmax(c_logits.reshape(N, -1), dim=-1).reshape(
            N, self.n_slots, self.n_experts
        )
        # y[i, :] = Σ_{s,e} C[i, s, e] · Y_all[s, e, :].  → (N, D).
        Y = torch.einsum("nse,sed->nd", C_w, Y_all)
        Y = Y.reshape(B, T, D)
        return self.dropout(Y)
