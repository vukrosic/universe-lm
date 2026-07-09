---
id: 260-k-only-norm-alibi
author: claude-opus-4-8
status: needs-plan
round: 1
updated: 2026-06-16T03:00:00Z
transfer-risk: low
plain: K-only RMSNorm on the alibi champion. +192 params. Step-0-active. Re-tests 165-k-only-norm (closed on base, Δ=-0.0293 borderline) on alibi. Asymmetric QK symmetry probe.
---

# 260 — K-only RMSNorm + alibi

## Hypothesis
Asymmetric pre-softmax RMSNorm on K only (Q raw). 016-qk_norm WIN was carried by **joint** QK symmetry (162-q-only Δ=-0.0043, 165-k-only-on-base Δ=-0.0293 both null on the base config). Re-running 165 on the alibi champion is a meaningful retest: alibi's additive linear-distance bias means K's gradient signal is asymmetric to Q's (K is shared across the attention logits in a way Q isn't), so the K-side normalization might bind here even though it didn't on the base.

## Mechanism
K-only RMSNorm rescales K's per-head-dim distribution to unit RMS, removing the magnitude scale that the dot-product would otherwise pick up. Combined with Q-raw and alibi's additive logit bias, the result is: per-head magnitude scale is determined entirely by K's norm and alibi's slope, not by Q's. At 12L/4H/d_model=64, this gives the optimizer an axis Q/K gradient updates don't trivially absorb (since Q's scale is set by attention and K's scale is set by the norm).

## Null expectation
Δ expected: < 0.02 either direction; if it lands inside the 0.04 band, NULL. Borderline prior from 165 (Δ=-0.0293 inside the 0.0524 cache band on base) suggests alibi might shift this — either more negative (alibi helps K-norm bind) or less (alibi dominates the magnitude axis).

## A/B
- Champion: `Tiny1M3MAlibiConfig` (val 6.2539, band 0.04, WIN gate < 6.2003)
- Treatment: same + `use_k_only_norm=True`
- Seed 42, no warmup, tiny1m3m
- Inline config: `@dataclass class C(Tiny1M3MAlibiConfig): use_k_only_norm=True`
- No new model code (flag already on box, line 1158 of configs/llm_config.py)

## Why this is staged
Fallback for if 256/257 (deepnet confirm) fails. QK-norm-attribution axis was 3/3 null on the base (016 WIN, 162 Q-only null, 165 K-only null); the alibi retest is the cheapest way to test whether the joint QK symmetry matters in the alibi regime.
