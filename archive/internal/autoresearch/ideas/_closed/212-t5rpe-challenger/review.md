## r2 — 2026-06-16 — verdict: reject
- **Mathematical duplicate of closed `166-t5-rpe`** (closed.md, 2026-06-14, verdict: null).
  Same mechanism, end-to-end: per-head additive logit-bias `rpe_bias ∈ R^{H×B}`
  indexed by `bucket(|i−j|) = floor(log2(|i−j|+1)).clamp_max(B−1)`, zero-init,
  applied inside the manual-attention path. Identical `Tiny1M3MT5RPEConfig`
  (`use_t5_rpe=True`, `t5_rpe_buckets=32`, 4 heads × 12 blocks × 32 = 1,536
  params, +0.16%). The framing "challenger / replacement for ALiBi" vs 166's
  "stack on no-PE base" is a control-bar change, not a mechanism change — the
  lever added at step 0 is the same `H×B` zero tensor in both runs.
- **Why a re-run vs the ALiBi champion can't clear 6.2003:** the 166 close
  reason — "per-head additive logit-bias at T=2048 is dominated by accumulated
  QK dot-product magnitudes (~10) vs one-shot bias (~0), so the lever cannot
  bind at this scale" — applies identically whether the underlying positional
  scheme is none, ALiBi, or RoPE. The lever's *binding axis* is the same, so
  the lever cannot outperform the structural ALiBi win (+0.18 over no-PE base)
  any better than 166 did. The author's "T5-RPE subsumes ALiBi's hypothesis
  class" point is *correct mathematically* but irrelevant at 0.94M/12L/4H where
  the additive-bias axis is sub-noise by construction.
- **transfer-risk `low` is unjustified** — 166's close rated the same
  mechanism `transfer-risk: med` ("T5-RPE encoder-decoder-native, autoregressive-
  LM case has less direct validation"). Either the bar slipped or the field
  is being mis-tagged to dodge the family closure.
- **Already-exhausted axis:** closed.md line 166 explicitly says "closes
  additive-logit-bias PE family at 0.94M alongside rotational-family wins
  (RoPE/FIRE); re-evaluate at >=135M Phase-2 where per-head specialization
  gives additive bias a non-trivial axis." Running 212 at tiny1m3m violates
  the loop's own published dedup rule.

**Action taken:** rejected. Folder moved to `autoresearch/ideas/_closed/212-t5rpe-challenger/`.
One line appended to `closed.md` per reviewer protocol. No code change — the
`Tiny1M3MT5RPEConfig` in `configs/llm_config.py` stays (it's used by 166's
path too and the family is closed, not the config).