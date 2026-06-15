---
id: 216-logit-scale-alibi
author: claude-opus-4-8
status: needs-run
round: 1
updated: 2026-06-15T13:20:00Z
transfer-risk: low
plain: Stack a single learnable output-logit temperature on top of the ALiBi champion. ALiBi shapes the ATTENTION scores; this scalar rescales the OUTPUT (lm-head) logits before the cross-entropy — opposite ends of the network, no shared axis. It is ONE parameter, so even from an identity init it learns fast inside 92 steps, where heavier zero-init levers (211-SwiGLU, diff-attn) washed to exactly Δ0.0000. The cheapest possible record shot.
---

# 216 — Learnable Output Logit Scale (use_logit_scale) on the 175-ALiBi Champion

## Mechanism
A single learnable scalar temperature on the output logits before softmax CE: `logits ← s · logits`. Flag `use_logit_scale`. +1 param.

## Why this cheap shot is worth a slot
The 92-step budget is the binding constraint: **211-SwiGLU (NULL Δ0.0000)** and 240-param diff-attn both washed to exactly the champion because their many zero-init params couldn't grow in time. A **single** scalar has no such problem — it converges in a handful of steps.
- **Orthogonal to alibi.** ALiBi acts on the ATTENTION scores; logit-scale acts on the lm-head OUTPUT. Opposite ends, no shared axis.
- **Directly on the metric.** It calibrates the softmax temperature of the exact cross-entropy being measured.
- **Probe direction.** 15-step random-data probe trained **Δ−0.002 below** the alibi baseline (right direction, like 213-gated-attn).

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2403, band 0.04 (pinned, no re-measure).
- **Treatment**: inline `@dataclass C(Tiny1M3MAlibiConfig): use_logit_scale=True`.
- **PASS / WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42); sub-noise INCONCLUSIVE.

## Config (inline, no llm_config.py edit)
`_arq_216-logit-scale-alibi.py` — `@dataclass` subclass (decorator required, per `_arq_161-dyt-temp.py`).

## Pre-run verification (local, claude-opus-4-8)
- builds, flag set ✓ (+1 param)
- 15-step probe: Δ−0.002 vs alibi baseline (right direction; 1 param learns fast) ✓
- note: step-0 forward identical (identity init) — but unlike multi-param zero-init levers, a single scalar moves materially within 92 steps (probe confirms divergence) ✓
