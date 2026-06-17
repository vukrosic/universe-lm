---
id: 253-deepnet-alpha-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T02:37:05Z
transfer-risk: low
plain: DIFFERENT AXIS pivot — DeepNet-α fixed residual scaling (x = x + α·f(x), α=1/√24) on the alibi champion. 0 new params, step-0-active, residual-conditioning / early-optimization axis (orthogonal to positional curvature, which came back noise-bound). Never A/B'd on alibi. No new model code.
---

# 253 — DeepNet-α fixed residual scaling on the alibi champion (use_deepnet_alpha)

## Why now (pivot reasoning)
The positional-curvature sweep (230 poly Δ−0.011, 231 kerple Δ+0.045, 232 stack Δ+0.016) is noise-bound: the poly 3-seed confirm straddles the champion (seed42 −0.011, seed123 +0.012), so the curvature axis yields no reliable gain. Pivoting to a **different axis**: residual-stream conditioning.

## Mechanism
DeepNet α (Wang et al. 2022, arXiv:2203.00555 §3.1): every sublayer output (attn + FFN) is multiplied by a fixed global scalar `α = (2·n_layers)^(-1/2) = 1/√24 ≈ 0.204` before the residual add, bounding residual growth to O(1). 0 new params (α is a float from n_layers). Step-0-ACTIVE (different operating point from step 0 — the survivor profile at 92 steps).

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2539, band 0.04.
- **Treatment**: inline `@dataclass C(Tiny1M3MAlibiConfig): use_deepnet_alpha=True`.
- **PASS/WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42).
- Honest prior: DeepNet's effect is largest at 100s of layers; at 12L the depth-drift fixed is small (√12≈3.5×), so expected effect is modest — but the axis is unexplored on alibi and the lever shape (0-param, step-0-active) is right.

## Pre-run verification
- **SMOKE_OK** ✓ — `_box_smoke.py _arq_253-deepnet-alpha-alibi.py`.
- No new model code (use_deepnet_alpha already wired + on box).
