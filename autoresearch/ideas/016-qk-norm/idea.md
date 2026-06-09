---
id: 016-qk-norm
status: running
round: 1
updated: 2026-06-09T13:47:10Z
---

# 016 — QK-Norm (LayerNorm on Q and K head-dim before attention)

## Source
"Scaling Vision Transformers to 22 Billion Parameters" (Dehghani et al.,
Google, 2023 — arXiv:2302.05442). Also adopted in Meta's Chameleon and
Stability AI's Stable LM 2 as a depth-stability lever.

## Mechanism
In the MHA forward, after the Q and K projections reshape into
`(B, H, T, d_head)`, apply a LayerNorm along the `d_head` axis of both
tensors *before* the attention dot product:

```
Q = q_proj(x).view(B, H, T, d_head)
K = k_proj(x).view(B, H, T, d_head)
Q = LayerNorm(Q)        # normalizes the last dim per head per token
K = LayerNorm(K)
attn = softmax(Q @ K^T / sqrt(d_head)) @ V
```

Because each Q/K vector is unit-RMS along `d_head`, the per-head attention
logit `Q·K / sqrt(d_head)` is bounded in magnitude by `sqrt(d_head)`,
preventing attention logit explosion. < 15 LoC: two `nn.LayerNorm(d_head)`
modules with elementwise-affine default, wired into `MHA.forward`. The
projection weights still identity-init cleanly; the new LNs start at
identity (γ=1, β=0) so step-0 is the baseline. Replace 0.5×d_head init in
the output projection if scaling was relying on logit magnitude — likely
not at tiny1m3m.

## Why it's worth a slot
The bet: even at 6 layers, the residual stream can grow in scale across
depth (well-known pre-norm drift), pushing attention logits up and
softening the softmax. QK-Norm is a *mechanism* (per-head logit
bounding) and is one of the few levers that has been shown to enable
clean training at 20+ layer ViTs and 30+ layer Chameleons. It is also
trivially composable with our existing GQA/MHA code — a strict drop-in
that only changes two tensor sites in `MHA.forward`. A null at 6 layers
would teach us QK-Norm's benefit is concentrated in deeper stacks; a
win would mean the logit-bound mechanic helps even at our scale, and
would license promoting it as a default in the 001-cautious-muon and
other optimizer-orthogonal A/Bs.
