---
id: 018-ademamix
status: rejected
round: 1
updated: 2026-06-09T12:28:08Z
---

# 018 — AdEMAMix (dual-EMA AdamW)

## Source
"The AdEMAMix Optimizer: Better, Faster, Older" (Pagliardini, Ablin,
Grangier — Apple; Sept 2024 — arXiv:2409.03137). Reports robust train-loss
and val-loss wins over tuned AdamW across LLM (124M-1.3B) and ViT scales
without extra memory of the optimizer state at inference; one new
scalar HP `α` (mixing weight) plus a slow EMA half-life.

## Mechanism
Standard AdamW keeps one EMA of the gradient (`m_t`) and one of its
square (`v_t`). AdEMAMix keeps a **second** EMA of the gradient,
`m2_t`, with a much slower decay (`β3 ≈ 0.9999`, half-life ~7k steps)
and combines the two before the update:

```
m1_t = β1 m1_{t-1} + (1-β1) g_t            # fast EMA, β1 = 0.9 as in AdamW
m2_t = β3 m2_{t-1} + (1-β3) g_t            # slow EMA, β3 ≈ 0.9999
v_t  = β2 v_{t-1} + (1-β2) g_t^2           # unchanged
update = (m1_t + α·m2_t) / (√v_t + eps)    # α scheduled 0 → α_final (~5-8)
θ ← θ - lr·update - lr·wd·θ                # decoupled weight decay (unchanged)
```

`α` warms up from 0 over `T_α` steps (paper: linear or sigmoid), so at
step 0 the optimizer is **bit-identical to AdamW**. The only new state
is one extra fp32 buffer per parameter (`m2`) — ~+33% optimizer-state
memory vs AdamW (which keeps `m1,v`); negligible at tiny1m3m. < 80 LoC
on top of the existing AdamW `step()` — a single new EMA, a single
schedule, a single line at the update.

## Why it's worth a slot
The bet: every closed/active optimizer lever to date — cautious-muon
(null), cautious-adamw (null), cautious-lion (planning), schedule-free
(null), SOAP (null), Moonlight-Muon-RMS (queued) — modifies the *update
direction* (sign mask, ortho update, eigenbasis precond, RMS rescale).
AdEMAMix modifies the *gradient signal itself*: the fast EMA tracks
recent gradient, the slow EMA tracks old gradient, and at α>0 the
update mixes both — exploiting the empirical fact that the *direction*
of gradients computed many steps ago is still useful late in training.
This is the orthogonal axis no optimizer ablation here has tested. A
win would mean the gradient temporal-mixing window is the lever; a null
would teach us the 3M-token horizon at tiny1m3m is too short for a slow
EMA to acquire useful information (paper claims the gap *widens* with
training length, so a null at tiny1m3m is informative either way).
Strictly distinct from closed schedule-free (which removes the LR
schedule and weight-averages parameters) — AdEMAMix keeps the schedule
and weight-averages *gradients*. < 80 LoC, identity-init at step 0.
