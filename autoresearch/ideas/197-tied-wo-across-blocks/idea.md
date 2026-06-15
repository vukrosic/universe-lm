---
id: 197-tied-wo-across-blocks
status: needs-repitch
round: 1
updated: 2026-06-15T08:20:39Z
transfer-risk: med
plain: Force every attention block to use the same output projection matrix W_O (init at the baseline's W_O so step-0 is byte-identical), like tying the final step of attention across depth — a cheap regularizer on what each block is allowed to write back to the residual stream.
---

# 197 — Tied W_O Across Blocks (Share Output Projection Across All Blocks)

## Source
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — share *all* parameters across blocks; 197 shares only W_O.
- Lan et al., "ALBERT" (arXiv:1909.11942, 2020) — shares attention and FFN parameters across blocks; validated at BERT-base/large/xlarge.
- Press et al., "T5" (JMLR 2020, arXiv:1910.10683) — encoder-decoder with shared parameters across layers in some experiments.
- In-repo: closed.md line "layer tying" closed the cross-layer parameter sharing axis. **But** the closed axis tested *all* parameter sharing (full Universal-Transformer-style); 197 is **W_O only** (a much narrower form of tying).
- 021-value-residual (in-repo WIN Δ=−0.034) — cross-block V mixing via residual; orthogonal axis to W_O tying.

## Mechanism
Standard attention: each block b has its own `W_O_b ∈ R^{d_model × d_model}` that projects the attention output back to the residual stream.

Tied W_O: every block uses the *same* `W_O` projection:
```
W_O_eff_b = W_O for all b
```
Or with a soft blend: `W_O_eff_b = (1 − α_b) · W_O_b + α_b · W_O` with α_b = 0 init (each block starts with its own W_O). At α_b = 1, fully shared.

**Hard version (proposed for 197)**: simply use one shared `W_O` for all blocks. Block b's W_O parameter slot is removed; the model has one global `W_O` parameter. Total params: 12 fewer `d_model × d_model = 4096` matrices, saving 49,152 params (-5.2% of 0.94M).

## Design sketch
- **File**: `models/layers.py` — modify `attention_block` to optionally pull W_O from a single shared `self.shared_W_O` rather than `self.W_O`.
- **Config flag**: `tie_wo_across_blocks: bool = False` (default).
- **Bit-identical at step 0**: shared `W_O` is *initialized* to match the baseline's per-block W_O at step 0 (each baseline block has its own W_O drawn from the same init distribution; tying to a single W_O is not bit-identical because the *specific* init values differ).
- **To get bit-identity**: use the *soft* blend formulation: `W_O_eff_b = (1 − α_b) · W_O_b + α_b · W_O_shared`. At α_b = 0 init, `W_O_eff_b = W_O_b` exactly.
- **Params**: 1 shared W_O + 12 α scalars = 4096 + 12 = 4,108 extra params; saves 11 × 4096 = 45,056 (with hard version). Net: -45k params (-4.8% of 0.94M).
- **Intuition**: W_O is the *output* of attention to the residual stream. Tying W_O across blocks forces all attention blocks to write to the residual stream in the same coordinate system — a strong regularizer on what attention can contribute. Different from layer tying (closed) which tied *all* parameters.

## Scale evidence
ALBERT validated at BERT-base/large/xxlarge (110M-235M); Universal Transformers validated at <100M. Both share *all* parameters; 197 shares *only W_O*. No published *W_O-only* tying paper that I'm aware of. Transfer-risk: med (the lever is novel; the closest analogs are full layer tying which is closed in-repo).

## Why it's worth a slot
**Pattern**: layer tying closed null in the closed.md axes line. The closed axis is full-layer tying (every parameter shared). 197 is a much narrower tying: only W_O. The bet: the failure mode of layer tying at 0.94M is too-aggressive regularization (parameters shared across all blocks → no depth-specific learning); tying only W_O (which has a clear functional role — projecting attention output back to the residual stream) may regularize enough without freezing depth-specific learning. A 197 WIN would mean the W_O tying is the binding constraint of layer tying; a 197 NULL would confirm the layer-tying null generalizes to W_O-only.
