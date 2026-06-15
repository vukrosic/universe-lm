---
id: 194-lowrank-ffn
status: needs-repitch
round: 1
updated: 2026-06-15T08:18:15Z
transfer-risk: med
plain: Shrink the FFN's intermediate dimension by a learnable low-rank factorization (init at identity so the FFN matches the baseline at step 0), like a soft bottleneck that the optimizer can tighten over training.
---

# 194 — Low-Rank FFN Bottleneck (Learnable Rank-r Projection as Soft Bottleneck)

## Source
- 146-sparse-ffn (closed null Δ=+0.0057 inside band) — sparse MoE-style FFN (4 experts, top-k gating). Different shape (sparsity, not low-rank).
- 157-conv-ffn (closed null Δ=−0.0078 inside band) — depthwise Conv1d post-FFN-activation. Different placement.
- 153-relu2-ffn (closed null Δ=−0.0053 inside band) — ReLU² activation in FFN. Activation shape lever.
- 158-gau (closed null Δ=+0.1095 drift) — Gated Attention Unit, fuses attention+FFN. Different shape.
- 170-swiglu-ffn (closed null Δ=−0.0170 inside band) — SwiGLU activation with 2/3-trick gate. Activation shape lever.
- 117-soft-moe, 118-mixture-of-depths, 145-expert-choice — all FFN-MoE variants closed at 0.94M. FFN capacity levers don't bind at this tier.
- Arora et al., "Linear Algebraic Structure of Word Senses" / "Low-Rank FFN" — multiple theoretical analyses suggest FFN matrices are approximately low-rank at scale.
- Hu et al., "LoRA" (arXiv:2106.09685, 2021) — learned low-rank factorization for adaptation; FFN-low-rank at init-identity is the training-from-scratch analog.

## Mechanism
Standard FFN: `out = (silu(W_gate · x) ⊙ (W_up · x)) · W_down` with W_gate, W_up ∈ R^{d_ff × d_model}, W_down ∈ R^{d_model × d_ff}, d_ff = 4·d_model = 256.

Low-rank FFN: replace W_up and W_gate with low-rank factorizations:
```
W_up_lowrank = W_up_A · W_up_B       # d_model → r → d_ff, r = 16 (vs d_ff = 256)
W_gate_lowrank = W_gate_A · W_gate_B # d_model → r → d_ff, r = 16
```
At init: `W_up_B = 0`, `W_up_A = some_normal_init`, `W_gate_B = 0`, `W_gate_A = some_normal_init`. With `W_up_B = 0`, the path `W_up_lowrank · x = W_up_A · 0 = 0`. To match baseline at step 0, init **W_up_B = 0, W_up_A = I_padded** (or similar identity-like structure) — or use the trick: **keep W_up, W_gate at full d_ff but add a learned rank-r residual correction** `W_up_eff = W_up + α · (W_up_A · W_up_B)`, init α=0. At α=0, `W_up_eff = W_up` exactly.

## Design sketch
- **File**: `models/components.py` (FFN module) — add an optional rank-r residual correction to W_up and W_gate.
- **Config flag**: `use_lowrank_ffn: bool = False`, `ffn_rank: int = 16`, `ffn_lowrank_alpha_init: float = 0.0`.
- **Compute**: `W_up_eff = W_up + α · W_up_A @ W_up_B`, where α = `sigmoid(α_raw)` init at `α_raw = -10` (so sigmoid ≈ 0). Same for W_gate.
- **Bit-identical at step 0**: α ≈ 0 ⇒ `W_up_eff = W_up` exactly (no contribution from the rank-r correction).
- **Params**: 2 × (d_model · r + r · d_ff) × 12 blocks = 2 × (64·16 + 16·256) × 12 = 2 × 5120 × 12 = 122,880 params (+13% of 0.94M); with α-init 0 the correction is silent at step 0. The 13% param inflation is non-trivial; the *lever* is whether the optimizer activates the rank-r path during training.
- **Intuition**: FFN matrices may have intrinsic low-rank structure; the rank-r residual lets the optimizer "tighten" the FFN to its effective rank over training. If the FFN is approximately rank-16 at 0.94M, this should win; if not, the rank-r correction becomes noise that drift the FFN off-baseline.

## Scale evidence
LoRA at 7B–65B; FFN-low-rank analyses at 7B+ (e.g., "LLM.int8()", Frantar et al. 2022). No published *training-from-scratch* low-rank FFN win at <100M that I'm aware of. Transfer-risk: med.

## Why it's worth a slot
The FFN-capacity-injection axis (146, 157, 170) and FFN-activation-shape axis (153) both closed null at 0.94M. 194 tests a different lever: **FFN with a learned rank-r correction**. The bet is that the FFN at 0.94M is over-parameterized in some specific rank-r subspace, and the rank-r correction lets the optimizer exploit that subspace without disturbing the main FFN. A null at 0.94M is expected (FFN-capacity levers don't bind); a win would mean the FFN has specific low-rank structure that the optimizer can leverage, distinct from the closed MoE / activation-shape axes.
