---
id: 206-cross-block-ffn-share
status: reviewing
round: 1
updated: 2026-06-15T08:30:35Z
transfer-risk: med
plain: Let each FFN use a small fraction of the previous block's W_up, W_down projections (a learnable per-block scalar, init 0 so step-0 is byte-identical), like tying the FFN upward and downward projections across adjacent blocks.
---

# 206 — Cross-Block FFN Projection Sharing (Learnable Blend of W_up, W_down Across Blocks)

## Source
- 197-tied-wo-across-blocks (in-repo idea) — share W_O across all blocks (hard tying with soft α blend). 206 is the FFN analog.
- 188-cross-block-kv-share (in-repo implementing) — share K, V projections across blocks. 206 is the FFN analog (W_up, W_down).
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — share *all* parameters across blocks.
- Lan et al., "ALBERT" (arXiv:1909.11942, 2020) — share attention and FFN parameters across blocks; validated at BERT-base/large/xxlarge.
- In-repo: closed.md line "layer tying" closed the full-layer-sharing axis. 206 is narrower: only W_up, W_down (the largest FFN matrices).
- 021-value-residual (in-repo WIN Δ=−0.034) — V-side cross-block carry. Different axis.

## Mechanism
Standard FFN: each block b has its own `W_up_b, W_gate_b, W_down_b`. Total FFN params per block: 3 × d_model × d_ff = 3 × 64 × 256 = 49,152.

Cross-block FFN projection sharing: each block's W_up and W_down are blended with the previous block's:
```
W_up_eff_b = (1 − α_up_b) · W_up_b + α_up_b · W_up_{b-1}     # α init 0
W_down_eff_b = (1 − α_down_b) · W_down_b + α_down_b · W_down_{b-1}   # α init 0
```
At α=0, W_up_eff = W_up_b (bit-identical baseline). At α=1, fully shared.

W_gate is left per-block (W_gate is the smaller, gating projection; sharing W_gate is too aggressive).

## Design sketch
- **File**: `models/components.py` — modify the FFN module to optionally blend W_up, W_down with the previous block's.
- **Config flag**: `use_cross_block_ffn_share: bool = False`, `ffn_share_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: per block b, compute α_up = sigmoid(α_up_raw), α_down = sigmoid(α_down_raw). `W_up_eff = (1 − α_up) · W_up_b + α_up · W_up_{b-1}.detach()`. Same for W_down.
- **Bit-identical at step 0**: α ≈ 0 ⇒ W_up_eff = W_up_b, W_down_eff = W_down_b exactly.
- **Params**: 2 α scalars per block × 12 blocks = 24 scalars (+0.003% of 0.94M).
- **Intuition**: W_up and W_down are the largest FFN matrices (49k params each). Tying them across blocks forces the FFN to use a shared "expansion subspace" (W_up) and "compression subspace" (W_down) across depth — a strong regularizer on the FFN's *input-output mapping*. Different from 197 (which ties W_O) and 188 (which ties K, V).

## Scale evidence
ALBERT validated at BERT-base/large/xxlarge; full layer tying closed in-repo at 0.94M (axis too aggressive). 206 is a *narrower* tying (only the two largest FFN matrices). Transfer-risk: med (lever is novel; partial layer tying is not well-validated at <100M).

## Why it's worth a slot
**Pattern**: full layer tying closed null at 0.94M (too aggressive); per-block W_O tying (197) is the in-repo idea testing the narrower axis. 206 tests the FFN analog — narrower still, only W_up and W_down. The bet: at 0.94M, the FFN's expansion (W_up) and compression (W_down) subspaces have useful structure to share across depth, distinct from the closed full-layer-tying axis. A 206 WIN would mean narrow FFN tying is a missing lever; a 206 NULL would mean full-layer tying's null generalizes to all partial-tying variants at 0.94M.
