# 003 — Soap (Shampoo + Adam)
_Auto-drafted 2026-06-10 from `autoresearch/ideas/003-soap/`._

## Abstract
Showed Shampoo (1/2-power) is mathematically equivalent to Adafactor in the preconditioner's eigenbasis. SOAP runs Adam inside that rotated basis and refreshes the eigenbasis only every K steps (one new hyperparameter: preconditioning frequency). Inherits Adam's simplicity + Shampoo's curvature benefits. Paper: 40%+ fewer iterations, 35%+ wall-clock savings vs AdamW on 360M/660M LM pre-training. Implementation is < 200 LoC. Propose: SOAP replaces AdamW (1D + embedding), Muon stays for 2D hidden. We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = 0.0113).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Vyas, Morwani, Zhao, Kwun, Shapira, Brandfonbrener, Janson, Kakade — "SOAP: Improving and Stabilizing Shampoo using Adam" (arXiv:2409.11321, Sep 2024; v2 Jan 2025). Code: https://github.com/nikhilvyas/SOAP..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Showed Shampoo (1/2-power) is mathematically equivalent to Adafactor in the preconditioner's eigenbasis. SOAP runs Adam inside that rotated basis and refreshes the eigenbasis only every K steps (one new hyperparameter: preconditioning frequency). Inherits Adam's simplicity + Shampoo's curvature benefits. Paper: 40%+ fewer iterations, 35%+ wall-clock savings vs AdamW on 360M/660M LM pre-training. Implementation is < 200 LoC. Propose: SOAP replaces AdamW (1D + embedding), Muon stays for 2D hidden.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

**Pass/fail bar.**
- pass: screen20m val ≤ 4.5887 (ctrl 4.6364, target Δ = −0.0477). V+q+SWA+HighRoPE baseline still applies.
- fail: screen20m val > 4.6364
- noise: |Δ| ≤ 0.05 — within the screen20m single-seed noise band; treat as inconclusive
- expected Δ ≈ −0.03 to −0.05; lower values are below the single-seed noise floor

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.4078, 6.4072 | 6.4075 |
| Treatment | 6.4191 | 6.4191 |

<details><summary>raw evidence.md</summary>

# Evidence — 003 soap

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.4191 — n=1 (single seed, per ONE-SEED-ONLY protocol)
- control bracket: ctrl=6.4078, ctrl2=6.4072 (gap 0.0006 — very tight this batch)
- Δ vs ctrl: +0.0113 (treatment is *worse*); Δ vs ctrl2: +0.0119 (worse)
- two-ctrl rule: treatment loses to *both* ctrls → NULL (wrong sign)
- raw: ~/arq/logs/{003-soap.log,ctrl.log,ctrl2.log} on box (batch 2026-06-09T10:39–10:51Z)
- date: 2026-06-09

Notes:
- First two attempts OOM'd: `torch.eye(d_out=49152)` = 9 GiB preconditioner on
  the vocab-sized params. Fixed by `MAX_PRECONDITIONER_DIM=2048` → AdamW
  fallback for any dim > 2048 (optimizers/soap.py). Third attempt ran clean.
- Caveat from that fix: at tiny1m3m the embedding/lm_head (vocab=49152) are the
  bulk of params and now take the plain-AdamW path, so SOAP only preconditions
  the small transformer matrices. The benefit SOAP is supposed to provide is
  largely bypassed at this tier — a fairer test of SOAP would need a tier where
  the preconditioned 2D blocks dominate. As-run at tiny1m3m: no gain.

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = 0.0113. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Vyas, Morwani, Zhao, Kwun, Shapira, Brandfonbrener, Janson, Kakade — "SOAP: Improving and Stabilizing Shampoo using Adam" (arXiv:2409.11321, Sep 2024; v2 Jan 2025). Code: https://github.com/nikhilvyas/SOAP.

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T10:55:03Z
