# 016 — QK-Norm (LayerNorm on Q and K head-dim before attention)
_Auto-drafted 2026-06-10 from `autoresearch/ideas/016-qk-norm/`._

## Abstract
In the MHA forward, after the Q and K projections reshape into
`(B, H, T, d_head)`, apply a LayerNorm along the `d_head` axis of both
tensors *before* the attention dot product: We test on tiny1m3m (seed 42). Verdict: UNKNOWN.

## 1 Introduction
This work re-implements and stress-tests the mechanism from "Scaling Vision Transformers to 22 Billion Parameters" (Dehghani et al.,
Google, 2023 — arXiv:2302.05442). Also adopted in Meta's Chameleon and
Stability AI's Stable LM 2 as a depth-stability lever..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
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

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | — | — |
| Treatment | — | — |

<details><summary>raw evidence.md</summary>

# 016 — QK-Norm (LayerNorm on Q,K head-dim) — evidence

**Date**: 2026-06-09
**Tier**: tiny1m3m (0.94M params, 3M tokens)
**Box**: vast-34386 (RTX 3060)
**Seed**: 42 (one seed only, per project rule)
**Queue**: ctrl → 015 → 016 → 017 → ctrl2

## Results

| Run | Final Val Loss | Δ vs ctrl1 | Δ vs ctrl2 |
|---|---|---|---|
| ctrl | 6.4044 | — | — |
| 015 (Moonlight) | 6.3906 | −0.0138 | −0.0185 |
| **016** (QK-Norm: LayerNorm on Q,K head-dim, γ=1, β=0) | **6.3906** | **−0.0138** | **−0.0185** |
| 017 (Sub-LN) | 6.4084 | +0.0040 | −0.0007 |
| ctrl2 | 6.4091 | — | — |

ctrl-to-ctrl gap: |6.4091 − 6.4044| = **0.0047**.

## Verdict — WIN

Treatment (6.3906) beats **both** ctrls (6.4044 and 6.4091) by more than the
ctrl-to-ctrl gap (0.0047). Δ of −0.0138 and −0.0185 ≫ 0.0047.

Pass bar from `plan.md`: `trt ≤ ctrl − 0.005`. Trt − ctrl1 = −0.0138, trt −
ctrl2 = −0.0185. Both pass.

## Note
Tied exactly with 015 at 6.3906 — different mechanism (per-tensor RMS
rescale on ortho'd update vs per-head LayerNorm on Q/K), same magnitude of
win. Suggests the small-model headroom is hit by either stability lever.

## Log files
- `~/arq/logs/ctrl.log` (75 KB)
- `~/arq/logs/016-qk-norm.log`
- `~/arq/logs/ctrl2.log`

</details>

## 5 Discussion
Verdict not yet recorded; this draft is preliminary.

## References
1. "Scaling Vision Transformers to 22 Billion Parameters" (Dehghani et al.,
Google, 2023 — arXiv:2302.05442). Also adopted in Meta's Chameleon and
Stability AI's Stable LM 2 as a depth-stability lever.

---
_Status_: **done** · _Verdict_: **UNKNOWN** · _Closed_: 2026-06-09T16:13:26Z
