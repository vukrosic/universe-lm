---
id: 258-per-head-rope-base-alibi
author: claude-opus-4-8
status: needs-plan
round: 1
updated: 2026-06-16T03:00:00Z
transfer-risk: low
plain: Per-head RoPE base on the alibi champion. +48 params. Step-0-active (init probe diff=2.71e-02). One of the 3 ready live fallback levers for if the deepnet 3-seed confirm fails.
---

# 258 — per-head RoPE base + alibi

## Hypothesis
Per-head RoPE base (each head gets its own base frequency, h-th head base = 500000^(h/H)) gives heads different positional-frequency specializations. This is one of the few well-validated knobs at LLaMA-3 / PaLM 2 scale (≥7B) that has not been A/B'd on the alibi champion. Live candidate per the init-probe (diff=2.71e-02, +48 params = +0.005% overhead, param-fair).

## Mechanism
RoPE rotation frequency varies by head, so the early/late token relative-position encoding has different "spatial wavelength" per head. Combined with the additive linear-distance alibi bias, the result is: alibi gives a global locality prior; per-head RoPE gives each head a different angular resolution. At 12L/4H, the per-head specialization is what we want to test (heads in same layer do different things at LLaMA-3 scale).

## Null expectation
At 0.94M, the rope base variations are absorbed by Q/K gradient updates (same null pattern as 152/155/160/166/172). Δ expected: < 0.02 either direction; if it lands inside the 0.04 band, NULL.

## A/B
- Champion: `Tiny1M3MAlibiConfig` (val 6.2539, band 0.04, WIN gate < 6.2003)
- Treatment: same + `use_per_head_rope_base=True`
- Seed 42, no warmup, tiny1m3m
- Inline config: `@dataclass class C(Tiny1M3MAlibiConfig): use_per_head_rope_base=True`
- No new model code (flag already on box, line 1809 of configs/llm_config.py)

## Why this is staged
This is the **fallback** experiment. The 3-seed confirm of 253-deepnet-alpha (256 + 257) is the pivotal data point. If deepnet's 3-seed mean is clearly < 6.2539 → it becomes the new champion and the next experiments stack on it (deepnet + per-head-rope-base, etc.). If deepnet fails → this lever (and 259/260) become the standalone candidates.
