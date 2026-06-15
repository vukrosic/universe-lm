---
id: 207-wo-lowrank-bottleneck
status: tasting
round: 1
updated: 2026-06-15T08:21:04Z
transfer-risk: med
plain: Insert a tiny low-rank bottleneck in the W_O projection (init so the bottleneck is silent, byte-identical at step 0), letting the model softly bottleneck what each attention block can write back to the residual stream.
---

# 207 — W_O Low-Rank Bottleneck (Learnable Rank-r Projection as Soft W_O Bottleneck)

## Source
- 194-lowrank-ffn (in-repo idea, needs-repitch) — low-rank FFN bottleneck. Different placement (FFN, not W_O).
- 197-tied-wo-across-blocks (in-repo idea) — tie W_O across all blocks. 207 is the *low-rank* version of W_O (different axis: rank vs sharing).
- Arora et al., "Linear Algebraic Structure of Word Senses" — theoretical analyses of low-rank structure in transformer weights.
- Hu et al., "LoRA" (arXiv:2106.09685, 2021) — learned low-rank factorization for adaptation; 207 is training-from-scratch on W_O.
- 160-rms-gain-per-head (closed null) — per-head gain on attention output. Different axis (gain not rank).
- 142-layerscale (closed null) — per-channel diagonal gain. Different axis.
- 161-dyt-temp (closed null Δ=+0.0830 wrong-sign) — different lever.

## Mechanism
Standard W_O: `out = W_O @ attn_output`, shape `[B, T, d_model]` from `[B, T, d_model]`. W_O ∈ R^{d_model × d_model} = 4096 params per block × 12 = 49,152 W_O params total.

Low-rank W_O: replace W_O with a low-rank factorization:
```
W_O_lowrank = W_O_A · W_O_B         # d_model → r → d_model, r = 16
out_lowrank = W_O_A · (W_O_B · attn_output)
```
At init: `W_O_B = 0`, `W_O_A = some_normal_init`. With W_O_B = 0, `out_lowrank = 0` (no contribution). To match baseline, use the **residual formulation**:
```
W_O_eff = W_O + α · W_O_A · W_O_B     # α init 0
out = W_O_eff @ attn_output
```
At α=0, `W_O_eff = W_O` exactly (bit-identical). At α>0, the optimizer can activate the rank-r path, which adds a low-rank *correction* to W_O.

The rank-r correction is a soft bottleneck: if W_O has intrinsic rank-16 structure, the correction lets the optimizer exploit it. If not, the correction adds noise.

## Design sketch
- **File**: `models/layers.py` — modify W_O projection to optionally include a rank-r residual correction.
- **Config flag**: `use_lowrank_wo: bool = False`, `wo_rank: int = 16`, `wo_lowrank_alpha_init: float = -10.0` (sigmoid ≈ 0).
- **Compute**: per block, `α = sigmoid(α_raw)`. `W_O_eff = W_O + α · W_O_A @ W_O_B`. `out = W_O_eff @ attn_output`.
- **Bit-identical at step 0**: α ≈ 0 ⇒ `W_O_eff = W_O` exactly.
- **Params**: 2 × (d_model · r) = 2 × (64 × 16) = 2048 per block × 12 blocks = 24,576 params (+2.6% of 0.94M); plus 12 α scalars.
- **Intuition**: W_O is the *output* of attention to the residual stream. A low-rank correction lets the optimizer add a structured bottleneck to what attention can write to the residual. Different from LayerScale (per-channel diagonal gain — full-rank), per-head gain (160), and W_O tying (197).

## Scale evidence
LoRA at 7B-65B; FFN-low-rank at 7B+; 194 (FFN low-rank, needs-repitch) tested at 0.94M. No published "W_O low-rank" win for LMs that I'm aware of. Transfer-risk: med (lever is well-defined; novel placement of low-rank factorization).

## Why it's worth a slot
**Pattern**: 194 (FFN low-rank) needs-repitch; 197 (W_O tying) is the in-repo sharing version of W_O; 199 (spectral-norm W_O regularization) tests Lipschitz control on W_O. 207 is the *low-rank* axis on W_O. The bet: at 0.94M, W_O has low intrinsic rank (it's d_model × d_model but the effective rank may be much smaller), and a learned rank-r correction exploits this. A 207 WIN would mean W_O has exploitable low-rank structure at 0.94M; a 207 NULL would mean W_O is approximately full-rank and the rank-r correction is noise.
