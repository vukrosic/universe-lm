---
id: 255-cosine-attn-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T03:01:45Z
transfer-risk: low
plain: Cosine attention on the alibi champion — L2-normalize Q and K before the dot product, with a per-head learnable temperature τ, so attention scores are bounded cosine similarities. Different axis (attention scoring geometry). Step-0-active (+48 params), learns fast at 92 steps. No new model code.
---

# 255 — cosine attention on the alibi champion (use_cosine_attn)

## Why now
Sampling the few step-0-active levers in the flag zoo (most are zero-init → wash at 92 steps). Curvature axis closed (poly null). Attention-scoring geometry is a distinct axis from positional (alibi), residual (deepnet-alpha, 253), and sparsity (entmax, 254).

## Mechanism
L2-normalize Q and K per head, score = τ_h · cos(Q,K); τ_h learnable init 1. Bounded similarity instead of unbounded dot product. +48 params (τ/head). Distinct from 016-qk-norm (magnitude RMSNorm) — cosine fully normalizes direction + learns temperature.

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2539, band 0.04.
- **Treatment**: inline `@dataclass C(Tiny1M3MAlibiConfig): use_cosine_attn=True`.
- **PASS/WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42).

## Pre-run verification (local CPU)
- **step-0 active** ✓ — diff vs alibi = 4.8e-2. **few-param** ✓ +48.
- **SMOKE_OK** ✓ — `_box_smoke.py _arq_255-cosine-attn-alibi.py`. No new model code (on box).
