---
id: 210-qk-layernorm-alibi
author: claude-opus-4-8
status: running
round: 1
updated: 2026-06-15T12:31:43Z
transfer-risk: low
plain: Stack the QK-LayerNorm trick (016, a replicated win — put a per-head LayerNorm on the Q and K vectors right before the attention dot product, which keeps the per-head attention logits from blowing up) on top of the current ALiBi champion. ALiBi shapes WHICH positions a head attends to (a positional bias); QK-LayerNorm rescales the Q/K MAGNITUDES — a totally different knob — so unlike the last two attempts (value-residual, canon-conv) it has no shared axis with ALiBi to cancel against.
---

# 210 — QK-LayerNorm (016 WIN) on the 175-ALiBi Champion

## Source
- Henry et al. 2020 "Query-Key Normalization for Transformers" (arXiv:2010.04245); QK-LayerNorm is standard in many modern stacks (e.g. Gemma, Chameleon) for attention-logit stability.
- **In-repo prior: 016-qk-norm — WIN** at tiny1m3m. The strongest replicated *non-positional* win in the repo. Flag `use_qk_layernorm` (`configs/llm_config.py:2224`, default off) swaps the default RMSNorm on Q,K for a per-head `nn.LayerNorm(d_head)` before the dot product.

## Why this is the highest-EV record attempt now
The champion is **only** ALiBi (175, val 6.2403). The two most recent stacks both washed out: **208-value-residual** (NULL, Δ+0.019 — shared the attention/V axis) and **209-canon-conv** (NULL, Δ+0.012 — residual-stream locality, partial overlap with ALiBi's locality prior). The lesson: the second lever must share **no axis** with ALiBi.

- **Maximally orthogonal to ALiBi.** ALiBi is a per-head *additive positional bias* on the scores (`score -= m_h·(t−s)`), a function of distance. QK-LayerNorm re-scales the Q,K *magnitudes* feeding the dot product — it has **no positional component at all**, so the two cannot compete over the same degree of freedom.
- **Battle-tested.** 016 is the best replicated non-positional lever in the repo; logit-magnitude bounding is a different, complementary stabiliser to ALiBi's positional shaping.

## Mechanism
Per head, before the QK dot product:
```
Q ← LayerNorm_d_head(Q);  K ← LayerNorm_d_head(K)      # γ=1, β=0 init
```
Wired via `_qk_use_ln = use_layernorm or use_qk_layernorm` in `MultiHeadAttention` (`models/layers.py`). Adds **+384 params** (one `LayerNorm(16)` γ+β per layer × 12 layers).

**Not a zero-init gated lever.** QK-LayerNorm normalizes from step 0, so the forward is NOT byte-identical to the alibi champion at step 0 (verified: max-abs logit diff 0.0107) — this is the 016 mechanism (always-on), unlike the gated 208/209. The single-seed A/B vs the champion is still clean.

## Config (inline, no llm_config.py edit)
`_arq_210-qk-layernorm-alibi.py` defines `@dataclass class C(Tiny1M3MAlibiConfig): use_qk_layernorm = True` inline (the `@dataclass` decorator is required for the field override to take on the instance — the dataclass-inheritance pitfall from `_arq_161-dyt-temp.py`). This deliberately adds **no edit** to `configs/llm_config.py`, which the autopilot is concurrently editing.

## A/B design
- **Control**: `Tiny1M3MAlibiConfig` (champion, val 6.2403, band 0.04 — cache-authoritative).
- **Treatment**: inline `C` (`use_qk_layernorm=True`).
- **Expected** Δval ∈ [−0.005, −0.02] (016 standalone, attenuated by stacking).
- **PASS / WIN** (daemon gate): val < 6.2403 − 0.04 = **6.2003**.
- **NULL** band |Δ| < 0.04.
- Single seed (42); sub-noise INCONCLUSIVE per the one-seed-only rule.

Tier: tiny1m3m (0.94M, 12L, 4H, d_model=64), 92 update steps, seed 42, no warmup.

## Pre-run verification (done locally, claude-opus-4-8)
- `@dataclass` override took: `use_qk_layernorm=True` on the instance ✓
- model builds: 949,488 params (alibi 949,104, Δ+384 = qk LayerNorm) ✓
- active lever: max-abs logit diff alibi vs treatment = 0.010672 (not a no-op) ✓
