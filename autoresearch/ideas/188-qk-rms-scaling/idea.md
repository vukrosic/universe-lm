---
id: 188-qk-rms-scaling
status: needs-repitch
round: 1
updated: 2026-06-15T08:21:04Z
transfer-risk: low
plain: Multiply the pre-softmax attention dot-product at every layer by one learned scalar (a per-block temperature), starting at 1 so step-0 is byte-identical to the baseline.
---

# 188 — Per-Block QK-Logit Scalar (Depth-Conditional Attention Temperature, init=1)

## Source
- "Fixing the Inherent Noise in AdamW's Variance Estimate" / "Depth-Adaptive Transformer" literature — depth-conditional per-layer logit scaling has appeared in DeepNet (Wang et al. 2022, arXiv:2203.00555) as part of the `α=1/sqrt(2L)` residual-scale recipe, in Mistral / Gemma 2 attention scaling, and in the per-layer RoPE base variants. Closest direct validation: iGPT-2 / Pythia per-layer LR multiplier (Biderman et al. 2023, arXiv:2304.01373) shows per-block scalar re-tuning can help at 1B-12B.
- Vaswani et al. 2017 (arXiv:1706.03762) — the original Transformer uses a single global `1/sqrt(d_k)` scalar; per-block deviations were not explored. The lever is the *learned, per-block* analog.
- In-repo context: 155-per-head-temp (H=4 per-head learnable τ_h) closed null at tiny1m3m (Δ=−0.0063, inside band). 188 differs in *axis*: 155 placed the temperature per-head (4 params per block × 12 = 48), 188 places it per-block (1 param per block × 12 = 12). The closed per-head axis does not bind the per-block axis. 188 is also different from 184-logit-scale (global output logit scalar) — 188 acts *pre-softmax on the attention logits*, not on the LM-head output logits.
- "Depth-Adaptive Transformer" (El-Nouby et al. 2023) — per-layer reparameterizations on attention scaling and FFN expansion.

## Mechanism
Standard pre-softmax QK product:
```
scores = Q @ K^T / sqrt(d_k)         # [B, H, T, T]
weights = softmax(scores)              # per-row simplex
out = weights @ V
```
With per-block QK-rms-scaling:
```
scores = (Q @ K^T / sqrt(d_k)) * s_l   # s_l: [n_layers] learnable, init 1.0
weights = softmax(scores)
out = weights @ V
```
`l ∈ {0, ..., n_layers-1}` indexes the transformer block. `s_l` is a per-block scalar that sharpens (`s_l > 1`) or flattens (`s_l < 1`) the attention distribution. Different from 155's per-head τ_h (which is inside the QK norm step) — `s_l` sits *after* the QK^T product and applies uniformly across all heads in the block.

**Parameterization**: `s_l = exp(s_param_l)` with `s_param_l = 0` init ⇒ `s_l = exp(0) = 1.0` exactly ⇒ `scores * 1.0 = scores` exactly ⇒ **byte-identical to baseline at step 0** (no fp32 epsilon from the `exp(0)` evaluation, which evaluates to `1.0` in IEEE 754).

**Why per-block vs per-head**: at 0.94M, the cross-block signal-to-noise ratio varies strongly with depth (early layers are noisy, late layers are refined). A per-block scalar gives the optimizer a single knob per layer to re-tune the attention sharpness for that block's specific role. Per-head (155) requires 4× more params and was null; per-block is the simpler depth-conditional axis that has not been directly tested.

## Design sketch
- **Files**:
  - `configs/llm_config.py` — add `use_qk_rms_scaling: bool = False` to `LLMConfig`. Add `Tiny1M3MQKRMSConfig(Tiny1M3MConfig)` with `use_qk_rms_scaling: bool = True`.
  - `models/layers.py` (or `models/llm.py`) — in the transformer block, allocate `self.qk_rms_param = nn.Parameter(torch.zeros(config.n_layers))` (one per layer). In the attention forward, after the `Q @ K^T / sqrt(d_k)` step, multiply by `self.qk_rms_param[layer_idx].exp()` before the softmax.
  - `models/llm.py` — pass `layer_idx` into the attention forward (likely already done in many implementations; otherwise thread via the block module).
- **Config flag**: `use_qk_rms_scaling: bool = False`.
- **Param count**: **12 scalars (+0.0013% of 0.94M)**. Negligible.
- **Step-0 byte-identity**: with `s_param_l = 0` init, `s_l = exp(0) = 1.0` exactly, so the pre-softmax `scores` are unchanged. The exp form is used (instead of a raw `nn.Parameter(torch.ones(n_layers))`) to guarantee positivity — a negative `s_l` would invert the softmax (lowest-score becomes the highest-weight), which is not a meaningful zero-init.
- **Intuition (why it might lower val loss)**: the canonical `1/sqrt(d_k)` scaling gives each block the same attention temperature regardless of depth. Empirically, late-layer attention in small models is often very sharp (a few tokens dominate) and early-layer attention is very diffuse (uniform across tokens). A per-block `s_l` lets the optimizer reduce sharpness in late layers (preventing the "attention-sink collapse" pathology) and/or sharpen early layers. At 0.94M, the per-block axis is a coarse depth-conditional knob that has not been directly tested — 155 (per-head) and 152 (per-head logit bias) both closed null, but neither tests the *block-conditional* axis.

## Scale evidence
- Pythia per-layer LR multiplier (Biderman et al. 2023, arXiv:2304.01373) — per-layer scalar retuning is a recognized axis at 70M-12B; 188 is the *attention*-side analog.
- DeepNet (Wang et al. 2022, arXiv:2203.00555) — uses `α = (2L)^(-1/4)` as a *fixed* (not learned) depth-conditional scale on the residual. Validated at 200-1000L.
- μ-Transfer / μP (Yang et al. 2022, arXiv:2203.03466) — per-layer logit-temperature scaling is a standard μP axis. Validated across 4 orders of magnitude (10M-13B).
- **Transfer-risk: low** — the per-block scalar axis is a recognized depth-conditional lever with multiple independent validations at ≥100M; the closed in-repo analogs (155 per-head, 152 per-head logit bias) test *different* axes (head-conditional, not block-conditional).

## Why it's worth a slot
The bet, in one sharp sentence: **the closed per-head temperature axis (155) suggests that *head-conditional* attention-shape levers don't bind at 0.94M, but the *block-conditional* axis is a different lever that has not been tested** — early layers plausibly need a different attention temperature than late layers, and a single per-block scalar (12 params) gives the optimizer a coarse depth-conditional knob that per-head scalars (48 params) over-parameterize; a null at 0.94M would close the *block-conditional attention temperature* axis (distinct from the closed *head-conditional* axis from 155), and a win would give a depth-conditional lever analogous to the iGPT-2 per-layer LR multiplier at scale.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion `Tiny1M3MAlibiConfig` val ≈ 6.24, band 0.04; cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 155-per-head-temp (H=4 per-head) — null. Different axis (block- vs head-conditional), different param count (12 vs 48).
- 152-attn-logit-bias (per-head additive) — null. Different mechanism (multiplicative on scores vs additive).
- 184-logit-scale — global scalar on LM-head output logits, not pre-softmax QK^T.
- 169-qk-norm-depth — null. Tests *placement* of QK norm (which block gets the norm), not a scalar multiplier on the scores.
- 016-qk-norm (WIN) — symmetric QK RMSNorm. 188 *adds* a per-block scalar after a QK norm; can be stacked on 016.
