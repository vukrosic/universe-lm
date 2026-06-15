---
id: 208-value-residual-alibi
status: running
round: 1
updated: 2026-06-15T10:35:03Z
transfer-risk: low
plain: Take the value-residual trick that already won on its own (each attention layer blends in a shortcut to the very first layer's "value" stream, starting at zero so step-0 is byte-identical) and stack it on the new ALiBi champion. The two knobs touch different parts of attention, so they should add up into a new record rather than cancel out.
---

# 208 — Value-Residual Learning on the 175-ALiBi Champion

## Source
- Zhou, Wang, Huang et al. 2024, "Value Residual Learning", arXiv:2410.17897 — each layer's value stream is blended with a residual shortcut to the first layer's value, `V_l ← (1−λ_l)·V_l + λ_l·V_1`, with a learned per-layer blend `λ_l`. Reported consistent improvements for decoder-only LMs at small-to-mid scale.
- In-repo prior: **021-value-residual — WIN** at tiny1m3m (Δ=−0.034 vs plain; closed.md). Also a WIN on top of FIRE (`Tiny1M3MVResidualOnFireConfig`), showing the lever is additive with a score-side positional win, not redundant with it.

## Why this is a high-probability new record
The current champion is **only** ALiBi (175, val 6.2403). The earlier validated wins (qk-norm 016, value-residual 021, canon-conv 023, gated-attn 024, ssmax 025) were measured against the *old* baseline and were never composed into this fresh champion. So stacking a battle-tested winner back on is a clean shot at a new record.

Value-residual is the strongest single candidate because it is **mechanistically orthogonal** to ALiBi:
- ALiBi (175) is a *score-side* per-head positional bias — it changes *which key* a head attends to.
- Value-residual is on the *projected V stream* — it changes *which value representation* the winners read from.

021 fired on the plain baseline **and** on top of FIRE (a different score-side positional lever), which is direct in-repo evidence that the V-stream shortcut compounds with a positional score-side win rather than washing out.

## Mechanism
Stash the projected V at layer 0 (post-W_V, post-GQA repeat, post-transpose, `[B, n_heads, T, d_k]`); in every later layer `l > 0`:
```
V_l ← (1 − λ_l)·V_l + λ_l·V_1      # before attn_weights @ V
```
- `λ_l = nn.Parameter(torch.zeros(()))` per block on the MHA → `λ_l = 0` at init → `V_l ← V_l` exactly → **byte-identical to the 175-ALiBi champion at step 0** (max-abs-diff = 0.0).
- `.detach()` on the V_1 stash → each layer's W_V trains on its own attention path; only the blend weight learns the cross-layer shortcut.

## Existing wiring (already in repo from 021 — no new model code)
- Flag `use_value_residual` read in `MinimalLLM.__init__` (`models/llm.py:400-401`) via `getattr(config, "use_value_residual", False)`.
- Layer-0 V stash + later-layer blend at `models/llm.py:845,1187,1697`.
- 208 only flips `use_value_residual=True` on top of `Tiny1M3MAlibiConfig` (config `Tiny1M3MVResidualAlibiConfig` in `configs/llm_config.py`). It adds no new wiring.

## A/B design
- **Control**: `Tiny1M3MAlibiConfig` (current champion, val 6.2403, band 0.04 — cache-authoritative, no re-measure).
- **Treatment**: `Tiny1M3MVResidualAlibiConfig` (`use_value_residual=True`).
- **Expected** Δval ∈ [−0.02, −0.05] (021 gave −0.034 on plain; the bet is most of that survives the alibi stack since the axes don't overlap).
- **PASS** ≤ 6.2403 − 0.01 = **6.2303**.
- **NULL** band |Δ| < 0.01 (V-stream shortcut redundant with alibi at this scale).
- **DRIFT** > +0.01.
- Single seed (42); sub-noise is INCONCLUSIVE per the one-seed-only rule.

Tier: tiny1m3m (0.94M, 12L, 4H, d_model=64), 92 update steps, seed 42, no warmup.
