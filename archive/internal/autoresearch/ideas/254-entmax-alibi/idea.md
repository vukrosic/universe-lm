---
id: 254-entmax-alibi
author: claude-opus-4-8
status: rejected
round: 3
updated: 2026-06-16T03:43:53Z
transfer-risk: low
plain: α-entmax sparse attention on the alibi champion — a learned per-head sparsity that generalizes softmax (α=1) toward sparsemax (α=2), letting heads assign exactly-zero attention to irrelevant tokens. Different axis (attention normalization/sparsity). Step-0-active (+48 params), so it learns fast in 92 steps. No new model code.
---

# 254 — α-entmax sparse attention on the alibi champion (use_entmax)

## Why now
Curvature axis is closed (poly-alibi 3-seed confirm mean ≈ +0.0017, straddles the champion — a confirmed null). Pivoting to the **attention-normalization/sparsity** axis. Most of the flag zoo is zero-init (washes at 92 steps); entmax is one of the few **step-0-active** levers (local probe: max-abs logit diff vs alibi = 5.8e-2), with only +48 params (one learnable α per head), so it can actually move in the budget.

## Mechanism
α-entmax (Peters, Niculae, Martins 2019, arXiv:1909.00015) replaces softmax over attention scores with α-entmax, a sparse normalization (α=1 ≡ softmax, α=2 ≡ sparsemax). Per-head learnable α. Orthogonal to alibi: alibi shapes WHICH positions are favored (distance prior); entmax shapes HOW SHARPLY/sparsely attention concentrates.

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2539, band 0.04.
- **Treatment**: inline `@dataclass C(Tiny1M3MAlibiConfig): use_entmax=True`.
- **PASS/WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42).

## Pre-run verification (local CPU)
- **step-0 active** ✓ — diff vs alibi = 5.8e-2 (genuine forward change, not zero-init).
- **few-param** ✓ — +48 (one α/head), fast-learning at 92 steps.
- **SMOKE_OK** ✓ — `_box_smoke.py _arq_254-entmax-alibi.py`. No new model code (on box).
