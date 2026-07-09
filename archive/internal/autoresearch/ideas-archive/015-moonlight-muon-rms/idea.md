---
id: 015-moonlight-muon-rms
status: done
round: 1
updated: 2026-06-09T16:13:26Z
---

# 015 — Moonlight Muon (per-tensor RMS scaling on orthogonalized update)

## Source
"Muon is Scalable for LLMs" (Moonlight team, Moonshot AI / Kimi, 2025) — extends
Keller Jordan's Muon (https://x.com/kellerjordan0/status/1853203334560575586).
Kimi's writeup + arXiv preprint from Feb 2025; arXiv:2502.16982 (best-known
citation at filing time — please verify in taste gate).

## Mechanism
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

## Why it's worth a slot
The bet: tiny1m3m has 6 layers and a mix of attention and FFN matrices with
shape ratios from 1:1 (Q/K/V heads) to 1:4 (FFN up). Plain Muon gives every
tensor a unit-spectral update but wildly different RMS, so the wide FFN
matrices get much larger element-wise steps than the attention heads —
implicitly re-balancing a fixed LR across shapes. Per-tensor RMS rescaling
is a *mechanism* (geometric calibration), not a hyperparameter sweep, and
it's the lever Moonlight credits with closing most of the gap between Muon
and a tuned AdamW. A null at 6 layers × 3M tokens would still teach us
that the shape-dependent RMS mismatch only bites at deeper / wider scales.
