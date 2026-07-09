---
id: 225-ffn-layerscale
status: needs-taste
round: 1
updated: 2026-06-16T01:00:00Z
transfer-risk: med
plain: Add a learnable per-channel diagonal gain (init 1, but with a tiny init epsilon so it starts near-identity) on the FFN output, separate from the existing LayerScale-style axes that were null on the residual stream. The FFN-side gain may bind where the residual-side one did not.
---

# 225 — LayerScale on FFN Output Only

## Source
LayerScale (Touvron et al 2021, arXiv:2103.17239) was originally proposed as a per-channel diagonal gain init `ε=1e-4` on each block's residual output, primarily for very deep transformers (≥50 layers). Closed 142-layerscale null at 0.94M tested per-channel diagonal gain on the *residual stream* (post-block) and lost (wrong-sign tiny, train_loss worse by +0.0312). The 130-rezero null at 0.94M is the scalar-α version on the residual.

**225 is different**: it applies the LayerScale-style per-channel diagonal gain *only on the FFN output*, before the residual add. The FFN residual is `x + α_FFN * ffn(x)` where `α_FFN` is a `(d_model,)` parameter init at `1.0` (NOT the LayerScale-style tiny init — see below). This is closer to "FFN residual gain" than "FFN LayerScale"; the mechanism is: the FFN contribution to the residual stream is multiplied by a learnable per-channel scale, while the attention contribution is not.

## Mechanism
```
# baseline
y = x + attn(rms_pre_attn(x))
y = y + ffn(rms_pre_ffn(y))             # residual add

# 225
y = x + attn(rms_pre_attn(x))
ffn_out = ffn(rms_pre_ffn(y))
ffn_gate = self.ffn_layerscale           # shape (d_model,), init 1.0
y = y + ffn_gate * ffn_out              # per-channel gated FFN residual
```

Init: `ffn_gate = 1.0` (NOT `1e-4` like LayerScale — we want bit-identity at step 0, not the LayerScale warm-up). The optimizer can shrink any channel toward 0 (kill that FFN feature) or grow it. **Important**: at `ffn_gate = 1.0`, the lever is bit-identical to baseline FFN.

## Design sketch
- **Files**: `models/layers.py` — locate the `FFN` forward. Add an `nn.Parameter(d_model)` `ffn_gate` initialized to 1.0 (NOT `1e-4`). Multiply the FFN output by `ffn_gate` before returning.
- **Config flag**: `use_ffn_layerscale: bool = False`, `ffn_layerscale_init: float = 1.0`. Default init = baseline (no scaling).
- **Cost**: 64 params per block × 12 blocks = +768 params, +0.082% of 0.94M. Cheap.
- **Why it should help at tiny1m3m**: the closed 142-layerscale null showed the *block-output* LayerScale doesn't bind at 12L, because the lever compounds with depth (per-channel gain × per-channel gain × ...). At L=12 with init ε=1e-4 the effective block-output magnitude is `(1e-4)^12 ≈ 0` — the warm-up never completes in 92 steps. **225 fixes this by init at 1.0** (not `1e-4`), so the lever starts at baseline and the optimizer can move it. The FFN-side gating is a *complementary* axis from the residual-side LayerScale (142): 142 controls "how much of each block's output to trust", 225 controls "how much of each FFN feature to trust". The closed 142 null doesn't directly apply to 225 because of the different init and different placement.
- **Why it might be null**: the FFN residual at 0.94M is already well-conditioned (the activation function + W_down constrain the magnitude), so a per-channel gain is redundant. The closed 217-mix-norm, 142-layerscale null, and 130-rezero null all target residual-stream or norm-stream conditioning at 0.94M and lost.

## Scale evidence
LayerScale paper (Touvron et al. 2021) validated at 50L-200L; closed 142-layerscale null shows it doesn't bind at 12L. 225 is "LayerScale-init-at-1.0 on FFN only" which is structurally novel — not in any major paper I know of, but a natural composition. Transfer-risk **med** because the underlying LayerScale lever is well-validated at deep scales but the FFN-only + init=1 variant is novel.

## Why it's worth a slot
A win would say the FFN-side residual gain binds at 0.94M even though the block-side doesn't (because the FFN path is shallower per block — single residual hop, not compounding over 12 layers). A null would close the "per-channel diagonal gain on FFN" axis at 0.94M alongside 142 (residual) and 130 (scalar). The lever is cheap (+768 params, ~10 LoC), bit-identical step 0 at init=1.0, and structurally distinct from the closed LayerScale axis.
