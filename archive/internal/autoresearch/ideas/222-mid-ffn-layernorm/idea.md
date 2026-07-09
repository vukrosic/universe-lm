---
id: 222-mid-ffn-layernorm
status: needs-review
round: 1
updated: 2026-06-16T00:46:19Z
transfer-risk: med
plain: Insert a LayerNorm in the middle of the FFN, between the up-projection and the activation/gate (or between the activation and the down-projection). At init the LN is gain=1, bias=0, so it is the identity and the FFN is unchanged; the model gets a free knob to re-scale the FFN's mid-state.
---

# 222 — Mid-FFN LayerNorm

## Source
Several papers have explored mid-network normalization:
- DeepNet (Wang et al. 2022, arXiv:2203.00555) uses Post-LN *and* sub-LN sandwich, but the FFN-internal norm is novel placement.
- PaLM (Chowdhery et al. 2022, arXiv:2204.02311) uses no bias and SwiGLU but does *not* add an internal FFN norm.
- Primer (So et al. 2021, arXiv:2109.08668) uses squared-ReLU activations in FFN but no internal norm.
- LLaMA-3 (Meta, 2024) uses RMSNorm pre-attn and pre-FFN but not internal-to-FFN.
- The closed 017-sub-ln-sandwich null at 0.94M is sandwich *around the block* (pre + post), not inside FFN.

The novel placement here is: a single LN inside the FFN, between the up-proj and the down-proj. Different from 017 (sandwich around block), 142 (LayerScale on residual), 130 (ReZero scalar).

## Mechanism
Standard FFN:
```
y = W_down(act(W_in(x)))           # or act(W_gate(x)) * W_in(x) for SwiGLU
```

Mid-FFN-LN:
```
h = W_in(x)                          # up-proj  (or gate*in for SwiGLU)
h = LN_mid(h)                        # NEW: internal LN, init gain=1, bias=0
y = W_down(act(h))                   # down-proj
```

The mid-LN has `gain=1, bias=0` at init, so `LN_mid(h) = h` exactly — step-0 bit-identical to baseline FFN. After step 1 the optimizer can use the gain/bias to re-scale the FFN's mid-state, which is currently constrained by the activation's range.

## Design sketch
- **Files**: `models/layers.py` — locate the `FFN` class. Add an `nn.LayerNorm(d_ff)` or `RMSNorm(d_ff)` between `self.up` (or `self.gate` for SwiGLU) and `self.down`. Use a config flag to choose `RMSNorm` vs `LayerNorm`.
- **Config flag**: `use_mid_ffn_layernorm: bool = False`, `mid_ffn_norm_type: str = "rms"` (rms vs layer). Default RMSNorm to match the rest of the model.
- **Cost**: 1 RMSNorm per FFN × 12 blocks = +24 RMSNorm instances. Each is d_ff=256 gains = +256 params × 24 = +6,144 params, +0.65% of 0.94M. Non-trivial but cheap.
- **Why it should help at tiny1m3m**: the FFN's mid-state is a 256-dim vector that is then projected down to d_model=64. The current setup has no constraint on this 256-dim distribution — outliers or scale drifts in the FFN mid-state get amplified by W_down. An internal LN gives the model a knob to bound this distribution *before* the down-projection, similar to why pre-norm on the block level helps. At 0.94M/12L, the FFN mid-state has high variance (per closed 153-relu2-ffn null at 0.94M, where ReLU² was tried), so an internal normalizer may help where activation tweaks did not.
- **Why it might be null**: the closed 017-sub-ln-sandwich null at 0.94M is the same family of "extra norm" levers and didn't bind at L=12. The closed 142-layerscale null is also depth-conditional and null at 12L. The mid-FFN-LN lever is *not* depth-conditional (it doesn't compound with L), so it has a better chance than 017/142, but the underlying pattern "extra norm placement" is crowded.

## Scale evidence
Sub-LN-sandwich null at 6L (017, 2026-06-09) is the direct empirical prior and argues the depth-conditional axis doesn't bind at our tier. Mid-FFN-LN is *not* depth-conditional (it's per-block, not per-layer-compounding), so the 017 null doesn't directly apply. Source papers generally test mid-norm at 100M-500M (Primer, PaLM-540M ablation). Transfer-risk **med**.

## Why it's worth a slot
A win would say the FFN's mid-state distribution is a binding constraint at 0.94M, and a learnable internal LN helps — a different axis from 017 (which was about block-level residual stability, not FFN-internal). A null confirms the FFN's mid-state is fine as-is and any extra normalizer is redundant with the existing pre-FFN RMSNorm. The lever is cheap (+6k params, ~15 LoC, bit-identical step 0 at gain=1, bias=0) and structurally novel relative to the closed norm-placement axes.
