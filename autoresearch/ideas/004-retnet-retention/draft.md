# 004 — RetNet retention (linear-attention alternative)
_Auto-drafted 2026-06-10 from `autoresearch/ideas/004-retnet-retention/`._

## Abstract
Replaces softmax attention with a retention kernel: per-head learnable decay γ < 1, position-dependent mask, no softmax. Three equivalent modes: parallel (training), recurrent (inference O(1)/step), chunkwise-recurrent (long sequences, linear complexity). Single Q/K/V projection + custom retention kernel — < 200 LoC. Paper claim: "favorable scaling, parallel training, low-cost deployment." We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = 0.0112).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei (Microsoft) — "Retentive Network: A Successor to Transformer for Large Language Models" (arXiv:2307.08621, Jul 2023). Code: https://aka.ms/retnet. ⚠️ 2023 paper; field has moved (Mamba, GLA, RWKV-7). Filing because: linear-attention alternatives are untested in this repo, and RetNet is the most-cited baseline with a working impl..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Replaces softmax attention with a retention kernel: per-head learnable decay γ < 1, position-dependent mask, no softmax. Three equivalent modes: parallel (training), recurrent (inference O(1)/step), chunkwise-recurrent (long sequences, linear complexity). Single Q/K/V projection + custom retention kernel — < 200 LoC. Paper claim: "favorable scaling, parallel training, low-cost deployment."

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

**Pass/fail bar.**
- pass: screen20m val ≤ 4.5864 (vs current best 4.6364, target Δ = −0.05). This is the high-EV scenario.
- fail: screen20m val > 4.6364 (worse than V+q+SWA+HighRoPE) — likely if linear attention loses at 10M scale
- noise: |Δ| ≤ 0.10 (screen20m noise band) — treat as inconclusive
- expected Δ ≈ −0.04 to −0.06; lower values are below the single-seed noise floor

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.3875, 6.4050 | 6.3963 |
| Treatment | 6.4162 | 6.4162 |

<details><summary>raw evidence.md</summary>

# Evidence — 004 retnet-retention

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.4162 (r1) — n=1
- control bracket: ctrl=6.3875, ctrl2=6.4050 (gap 0.0175)
- Δ vs ctrl: +0.0287 (treatment is *worse* than ctrl)
- Δ vs ctrl2: +0.0112 (treatment is *worse* than ctrl2)
- pass/fail bar (idea.md screen20m legacy 4.5864 / n/a tiny1m3m): n/a — v1 ships
  the kernel + probe, not the production attention rewrite. The arq-r1 run
  exercised the **probe** path (no retention wired into `MultiHeadAttention`),
  so this measurement is a sanity check that the kernel is bit-stable, not
  a real A/B against the retention attention. The v2 wiring PR will do the
  real A/B.
- two-ctrl rule: treatment > both ctrls → NULL (worse than both). Plan
  `pass: tiny1m3m val ≤ 6.4237` is *not* met in the WIN sense (treatment
  is higher than both ctrls).
- box check: ctrl 6.3875 vs leaderboard 6.4287 = -0.0413 (within 0.04 noise band)
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1/{004-retnet-retention.log,ctrl.log,ctrl2.log}
- date: 2026-06-09

v1 ships a working kernel (kernel + 4 invariants in pytest). The probe ran
without NaN/Inf and produced a stable val_loss. v2 (the real A/B) is a
separate PR — filed for the next pipeline cycle.

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = 0.0112. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei (Microsoft) — "Retentive Network: A Successor to Transformer for Large Language Models" (arXiv:2307.08621, Jul 2023). Code: https://aka.ms/retnet. ⚠️ 2023 paper; field has moved (Mamba, GLA, RWKV-7). Filing because: linear-attention alternatives are untested in this repo, and RetNet is the most-cited baseline with a working impl.

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T09:36:29Z
