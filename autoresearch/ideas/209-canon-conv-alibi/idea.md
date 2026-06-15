---
id: 209-canon-conv-alibi
status: needs-run
round: 1
updated: 2026-06-15T10:30:00Z
transfer-risk: low
plain: Add the canon-conv trick (a tiny 3-wide causal convolution on the residual stream of every block, gated to zero at start so step-0 is byte-identical) on top of the new ALiBi champion. It mixes each token with its 2 left-neighbors *before* attention runs — a cheap local-context boost that lives completely outside the attention math, so it shouldn't collide with ALiBi the way value-residual did.
---

# 209 — Canon-Conv (Residual-Stream Local Mixing) on the 175-ALiBi Champion

## Source
- De, Smith, Fernando et al. 2024, "Griffin" (arXiv:2402.19427); Allen-Zhu et al. Canon-layer line — one causal depthwise `Conv1d(kernel=3, left-pad 2)` on the residual stream per block, before the attention pre-LN, with a per-block scalar output gate `g`.
- In-repo prior: **023-canon-conv — WIN** at tiny1m3m (Δ≈−0.06 after stripping the buggy FIRE control; the best of the entire 020–025 cluster, flagged "best for Phase-2").

## Why this is the highest-EV shot at a new record
The current champion is **only** ALiBi (175, val 6.2403). The first attempt to stack a prior win — 208-value-residual — washed out to a null (Δ +0.019). The lesson: a second lever must not share an axis with alibi. Canon-conv is the cleanest such lever:

- **Maximally orthogonal to ALiBi.** ALiBi is a per-head additive *positional bias on attention scores* — entirely inside the attention computation. Canon-conv lives on the *residual stream itself*, before pre-LN, and never touches the scores. Zero shared axis for the two to compete over (contrast 021's V-stream overlap inside attention).
- **Large effect margin.** 023 cleared the WIN bar by ~6× (≈−0.06 vs −0.01), so even heavy stacking attenuation should still clear PASS.
- **Local + global positional priors are a classic additive pair** (Griffin, Mamba-2): canon supplies cheap k=3 neighbor context *before* the global attention pass; alibi shapes *which* distant tokens that pass attends to.

## Mechanism
Per block, before the attention sublayer's pre-LN:
```
x ← x + g · depthwise_conv1d(x)      # kernel=3, left-pad 2 (causal), per-channel
```
- `g` is a per-block scalar output gate init **0** → `x ← x + 0·conv(x) = x` exactly → **byte-identical to the 175-ALiBi champion at step 0** (max-abs-diff = 0.0).
- Depthwise (per-channel) causal conv → each position mixes only itself and its 2 left-neighbors; no future leakage.

## Existing wiring (already in repo from 023 — no new model code)
- Flag `use_canon_conv` at `configs/llm_config.py:487-498`. The per-block depthwise Conv1d + zero-init gate are built only when the flag is on; the baseline path is bit-identical when off.
- 209 only flips `use_canon_conv=True` on top of `Tiny1M3MAlibiConfig` (config `Tiny1M3MCanonConvAlibiConfig`). It adds no new wiring.

## A/B design
- **Control**: `Tiny1M3MAlibiConfig` (current champion, val 6.2403, band 0.04 — cache-authoritative).
- **Treatment**: `Tiny1M3MCanonConvAlibiConfig` (`use_canon_conv=True`).
- **Expected** Δval ∈ [−0.02, −0.06].
- **PASS** ≤ 6.2403 − 0.01 = **6.2303**.
- **NULL** band |Δ| < 0.01.
- **DRIFT** > +0.01.
- Single seed (42); sub-noise is INCONCLUSIVE per the one-seed-only rule.

Tier: tiny1m3m (0.94M, 12L, 4H, d_model=64), 92 update steps, seed 42, no warmup.
