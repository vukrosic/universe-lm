# 015 — Moonlight Muon (per-tensor RMS scaling on orthogonalized update)
_Auto-drafted 2026-06-10 from `autoresearch/ideas/015-moonlight-muon-rms/`._

## Abstract
Standard Muon replaces AdamW's element-wise update with a Newton–Schulz
orthogonalized 2D-matrix update `O = NS5(G)` (so `O` has singular values ≈ 1).
The Moonlight extension multiplies the orthogonalized update by a per-tensor
shape-dependent RMS-rescale: We test on tiny1m3m (seed 42). Verdict: UNKNOWN.

## 1 Introduction
This work re-implements and stress-tests the mechanism from "Muon is Scalable for LLMs" (Moonlight team, Moonshot AI / Kimi, 2025) — extends
Keller Jordan's Muon (https://x.com/kellerjordan0/status/1853203334560575586).
Kimi's writeup + arXiv preprint from Feb 2025; arXiv:2502.16982 (best-known
citation at filing time — please verify in taste gate)..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Standard Muon replaces AdamW's element-wise update with a Newton–Schulz
orthogonalized 2D-matrix update `O = NS5(G)` (so `O` has singular values ≈ 1).
The Moonlight extension multiplies the orthogonalized update by a per-tensor
shape-dependent RMS-rescale:

```
update = c * sqrt(max(d_in, d_out)) * O
```

with `c ≈ 0.2` (the paper's tuned constant, single global knob). For an
attention head of shape `(d_head, d_head)` and an FFN up-proj of shape
`(d_model, 4·d_model)`, this rescales the update so every 2D weight has
approximately unit RMS in its entries — uniform step magnitude across all
matrix shapes in the network. Implemented as a single post-mul inside the
Muon optimizer; < 10 LoC on top of the existing `optimizers/muon.py`.

Step-0 is unchanged (orthogonalization of a zero gradient yields a zero
update), so the baseline matches the control at step 0 — identity-init
compatible.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | — | — |
| Treatment | — | — |

<details><summary>raw evidence.md</summary>

# 015 — Moonlight Muon RMS rescale — evidence

**Date**: 2026-06-09
**Tier**: tiny1m3m (0.94M params, 3M tokens)
**Box**: vast-34386 (RTX 3060)
**Seed**: 42 (one seed only, per project rule)
**Queue**: ctrl → 015 → 016 → 017 → ctrl2

## Results

| Run | Final Val Loss | Δ vs ctrl1 | Δ vs ctrl2 |
|---|---|---|---|
| ctrl | 6.4044 | — | — |
| **015** (Moonlight `c·√max(d_in,d_out)`, c=0.2) | **6.3906** | **−0.0138** | **−0.0185** |
| 016 (QK-Norm) | 6.3906 | −0.0138 | −0.0185 |
| 017 (Sub-LN) | 6.4084 | +0.0040 | −0.0007 |
| ctrl2 | 6.4091 | — | — |

ctrl-to-ctrl gap: |6.4091 − 6.4044| = **0.0047**.

## Verdict — WIN

Treatment (6.3906) beats **both** ctrls (6.4044 and 6.4091) by more than the
ctrl-to-ctrl gap (0.0047). Δ of −0.0138 and −0.0185 ≫ 0.0047 — inside the
two-ctrl rule.

Pass bar from `plan.md`: `trt ≤ ctrl − 0.01`. Trt − ctrl1 = −0.0138, trt − ctrl2
= −0.0185. Both pass.

## Log files
- `~/arq/logs/ctrl.log` (75 KB)
- `~/arq/logs/015-moonlight-muon-rms.log`
- `~/arq/logs/016-qk-norm.log` (compositional — same A/B session)
- `~/arq/logs/017-sub-ln-sandwich.log` (compositional)
- `~/arq/logs/ctrl2.log`

</details>

## 5 Discussion
Verdict not yet recorded; this draft is preliminary.

## References
1. "Muon is Scalable for LLMs" (Moonlight team, Moonshot AI / Kimi, 2025) — extends
Keller Jordan's Muon (https://x.com/kellerjordan0/status/1853203334560575586).
Kimi's writeup + arXiv preprint from Feb 2025; arXiv:2502.16982 (best-known
citation at filing time — please verify in taste gate).

---
_Status_: **done** · _Verdict_: **UNKNOWN** · _Closed_: 2026-06-09T16:13:26Z
