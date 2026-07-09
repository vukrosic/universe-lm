---
id: 213-gated-attn-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-15T13:06:15Z
transfer-risk: low
plain: Stack a multiplicative sigmoid gate on the attention sub-block OUTPUT (per-channel) on top of the ALiBi champion. ALiBi adds a positional bias to the attention SCORES; this gate rescales the attention OUTPUT channels — different end of the block, no shared axis. Crucially it is STEP-0 ACTIVE (not a zero-init gate like 211-SwiGLU which washed to Δ0.0000), so it can actually move inside the 92-step budget. In the local probe it was the orthogonal stack that trained furthest in the right direction.
---

# 213 — Gated Attention Output (use_gated_attn) on the 175-ALiBi Champion

## Mechanism
Per-channel sigmoid gate on the attention sub-block output, post-AV / pre-residual: `out = sigmoid(g(x)) ⊙ AttnOut`. Flag `use_gated_attn` (already in `configs/llm_config.py`). +3,120 params (gate projection).

## Why this is high-EV after 208–211 all washed out
The decisive lesson from **211-SwiGLU (NULL, Δ0.0000)**: a **zero-init** lever cannot grow enough in 92 update steps to register — it reverts to the champion exactly. The 208/209/210 gated levers did the same. The fix is to pick a lever that is **active from step 0**.

- **Step-0 active.** Local probe: max-abs logit diff **0.077** vs the alibi champion at step 0 (the gate is not init to identity) — it contributes from the first update.
- **Orthogonal to alibi.** ALiBi = additive *positional* bias on pre-softmax SCORES. Gated-attn = multiplicative *channel* gate on the post-AV OUTPUT. No shared degree of freedom (unlike 208 V-axis / 209 locality / 210 logit-magnitude).
- **Probe direction.** 15-step random-data probe trained **Δ−0.0041 below** the alibi baseline — one of only two orthogonal stacks (with logit-scale) that moved the right way.

## A/B design
- **Control / bar**: champion `Tiny1M3MAlibiConfig` val 6.2403, band 0.04 (cache-authoritative, no re-measure; daemon judges treatment vs this pinned val).
- **Treatment**: inline `@dataclass C(Tiny1M3MAlibiConfig): use_gated_attn=True`.
- **PASS / WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42); sub-noise INCONCLUSIVE.

## Config (inline, no llm_config.py edit)
`_arq_213-gated-attn-alibi.py` — `@dataclass` subclass (the decorator is required for the field override to take, per the `_arq_161-dyt-temp.py` pitfall). No edit to the shared config the autopilot is touching.

## Pre-run verification (local, claude-opus-4-8)
- builds, flag set on instance ✓ (+3,120 params)
- **step-0 active**: max-abs logit diff 0.077 vs alibi (NOT a zero-init no-op) ✓
- 15-step probe: Δ−0.0041 vs alibi baseline (right direction) ✓
