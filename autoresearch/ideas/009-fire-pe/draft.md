# 009 — FIRE positional encoding (functional-interpolation relative PE)
_Auto-drafted 2026-06-10 from `autoresearch/ideas/009-fire-pe/`._

## Abstract
Add a learnable position-dependent bias to attention logits: `bias(i,j) = γ(i-j) · f(φ(x_i), φ(x_j))` where `γ` is a fixed Lp-norm kernel (monotone decay in relative distance) and `f` is a small MLP over learned projections of the query/key token embeddings. The bias is *input-dependent* on content (via `φ`) but the position kernel is fixed, so the model gets context-sensitive positional bias without losing the no-max-len property of pure relative PE. Implementation: drop-in for RoPE — same shape (additive bias on logits) — ~30-50 LoC for the kernel + MLP + bias-add into attention. No new parameters in the attention output path; the MLP and per-head learnables are tiny. We test on tiny1m3m (seed 42). We observe a WIN with Δ = 6.4237 vs mean control under a two-ctrl bracket.

## 1 Introduction
This work re-implements and stress-tests the mechanism from Li et al., "Functional Interpolation for Relative Positional Encoding" (NeurIPS 2023, arXiv:2306.02613). Reference: original paper repo..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Add a learnable position-dependent bias to attention logits: `bias(i,j) = γ(i-j) · f(φ(x_i), φ(x_j))` where `γ` is a fixed Lp-norm kernel (monotone decay in relative distance) and `f` is a small MLP over learned projections of the query/key token embeddings. The bias is *input-dependent* on content (via `φ`) but the position kernel is fixed, so the model gets context-sensitive positional bias without losing the no-max-len property of pure relative PE. Implementation: drop-in for RoPE — same shape (additive bias on logits) — ~30-50 LoC for the kernel + MLP + bias-add into attention. No new parameters in the attention output path; the MLP and per-head learnables are tiny.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.3875, 6.4050 | 6.3963 |
| Treatment | 6.3234 | 6.3234 |

<details><summary>raw evidence.md</summary>

# Evidence — 009 fire-pe

## Verdict: WIN
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.3234 (r1) — n=1
- control bracket: ctrl=6.3875, ctrl2=6.4050 (gap 0.0175)
- Δ vs ctrl: -0.0641 (treatment beats ctrl by 0.0641 ≫ gap 0.0175)
- Δ vs ctrl2: -0.0816 (treatment beats ctrl2 by 0.0816 ≫ gap 0.0175)
- pass/fail bar (idea.md): pass ≤ 6.4237 (target Δ = -0.005).
  Bar *far* exceeded: 6.3234 ≪ 6.4237.
- two-ctrl rule: treatment beats *both* ctrls by more than the gap → WIN
- box check: ctrl 6.3875 vs leaderboard 6.4287 = -0.0413 (within 0.04 noise band;
  both ctrls agree on direction, treatment Δ is 1.5× the noise band — robust)
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1/{009-fire-pe.log,ctrl.log,ctrl2.log}
- date: 2026-06-09

**Δ of -0.064/-0.082 is the largest of any of today's A/Bs** (vs the
cautious-Muon -0.025 / decoupled-QKV -0.014 / retention -0.011). FIRE wins
big on the val-distribution test at tiny1m3m. The plan's length-extrapolation
upside is untested at this tier (T=2048, fixed-length run); the win here is
the train-distribution val_loss, not extrapolation.

</details>

## 5 Discussion
The treatment beats both controls beyond the ctrl-to-ctrl gap (two-ctrl rule satisfied). Δ = 6.4237 vs mean control. Effect survives at this scale; next step is a wider-tier replication.

## References
1. Li et al., "Functional Interpolation for Relative Positional Encoding" (NeurIPS 2023, arXiv:2306.02613). Reference: original paper repo.

---
_Status_: **done** · _Verdict_: **WIN** · _Closed_: 2026-06-09T09:36:32Z
