---
id: 212-t5rpe-challenger
author: claude-opus-4-8
status: running
round: 2
updated: 2026-06-15T16:40:40Z
transfer-risk: low
plain: Challenge the ALiBi champion head-on with a STRICTLY MORE EXPRESSIVE positional mechanism on the SAME axis. ALiBi gives each head ONE slope and a fixed straight-line distance decay. T5-RPE gives each head a full learnable bias vector over log-spaced relative-distance buckets — it can reproduce ALiBi's line as a special case AND any curved/non-monotonic distance profile ALiBi can't. This is NOT a stack (those washed out 3× in a row); it REPLACES ALiBi with a superset of its hypothesis class. If a richer distance profile helps at this tier, T5-RPE alone beats the record.
---

# 212 — T5-RPE Bucketed Relative-Position Bias as an ALiBi Challenger

## Source
- Raffel et al. 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (JMLR; arXiv:1910.10683). T5's relative-position bias; reused in BigBird, REALM, LongT5.
- **In-repo:** `Tiny1M3MT5RPEConfig` / flag `use_t5_rpe` (`configs/llm_config.py:2807`), `t5_rpe_buckets=32`. Zero-init, manual attention path so the bucket bias is exact (not routed through SDPA).

## Why CHALLENGE, not stack
208/209/210 all tried to **stack** a second lever on ALiBi and washed out (NULL), because the marginal lever's effect was smaller than the 0.04 noise band. But ALiBi itself was a **+0.18 structural win** over base (6.42 → 6.2403) — a *large* effect — precisely because it added a positional inductive bias. The way to beat a large structural win on an axis is **not** another small orthogonal bolt-on; it is a **strictly more expressive mechanism on the same axis**.

- **ALiBi's hypothesis class** = one scalar slope per head: `score −= m_h·(t−s)`. A single straight line per head.
- **T5-RPE's hypothesis class** = a learnable bias `rpe_bias ∈ R^{H×B}` indexed by `bucket(|t−s|)` = log-spaced bins. It can represent ALiBi's linear decay AND any non-monotonic distance profile. **It subsumes ALiBi.**

If the optimal distance profile at this tier is not exactly linear, T5-RPE captures the residual ALiBi cannot — and clears 6.2403 outright.

## Mechanism
Per head, add to the pre-softmax score a bias indexed by the log-bucket of relative distance:
```
score[h,i,j] += rpe_bias[h, bucket(|i−j|)],  bucket(d)=floor(log2(d+1)).clamp_max(B−1)
```
`rpe_bias` is **zero-init** ⇒ step-0 byte-identical to the no-RPE baseline. At seq_len 16–2048 only buckets 0..11 are indexed; the rest stay zero (kept for T5 param-count parity). Cost: H×B×L = 4×32×12 = 1,536 params (+0.16%).

## Config (on-box class, no llm_config.py edit)
`_arq_212-t5rpe-challenger.py` runs `configs.llm_config.Tiny1M3MT5RPEConfig` directly (added since 166) — **no inline override, no edit** to the shared config file. NOTE: this config does **not** include ALiBi; T5-RPE *replaces* the positional mechanism.

## A/B design
- **Bar to beat**: the ALiBi champion, val 6.2403, band 0.04 (cache-authoritative; the daemon judges the treatment against this pinned champion val, no control re-measure).
- **Treatment**: `Tiny1M3MT5RPEConfig` (`use_t5_rpe=True`, 32 buckets — NO alibi).
- **PASS / WIN** (daemon gate): val < 6.2403 − 0.04 = **6.2003**.
- **NULL** band |Δ| < 0.04.
- Single seed (42); sub-noise INCONCLUSIVE per the one-seed-only rule.

**Known risk:** T5-RPE has 1,536 params to learn (vs ALiBi's 48) and only 92 update steps — the extra capacity may not converge, landing at/above ALiBi (NULL). That is an acceptable, informative outcome: it tells us ALiBi's linear shape is already optimal at this budget. Zero-init means it never *starts* worse than no-RPE.

Tier: tiny1m3m (0.94M, 12L, 4H, d_model=64), 92 update steps, seed 42, no warmup.

## Pre-run verification (done locally, claude-opus-4-8)
- config flags: `use_t5_rpe=True`, `t5_rpe_buckets=32` on the instance ✓
- model builds: 950,592 params (alibi 949,104, Δ+1,488 = 12×(4×32) rpe_bias − 48 alibi slopes) ✓
- path consumed: forcing `rpe_bias` to vary per bucket changes logits by 0.0156 (a uniform fill is a no-op — softmax shift-invariance — confirming the bias is added per relative-distance, not as a constant) ✓
