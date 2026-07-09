---
id: 259-layernorm-alibi
author: claude-opus-4-8
status: needs-plan
round: 1
updated: 2026-06-16T03:00:00Z
transfer-risk: low
plain: use_layernorm on the alibi champion (swap default RMSNorm → LayerNorm). +1984 params. Step-0-active (init probe diff=2.88e-02). Norm axis; closes a gap left by 017-sub-ln closing.
---

# 259 — use_layernorm + alibi

## Hypothesis
Switch the default norm (RMSNorm) to LayerNorm across all pre-norm sites. Pre-LN sandwich (LayerNorm before attn/ffn) is one of the few norm-axis levers not closed on alibi (Sub-LN 017 closed at 0.94M with the depth-conditional lever cluster). Live candidate per the init-probe (diff=2.88e-02, +1984 params = +0.21% overhead — heavier than the other 2 fallbacks but acceptable on 0.94M).

## Mechanism
LayerNorm centers and scales per token; RMSNorm only scales. The extra mean-subtraction can matter for activations that develop a non-zero mean drift over training. At 12L/4H/d_model=64, the gradient signal per token is too small for the centering to bite — this is the same null pattern as 017-sub-ln, 130-rezero, 142-layerscale. But unlike those, use_layernorm is a **global** swap, not depth-conditional, so the parameter overhead is uniform across all 12 blocks. May have a different null pattern at 0.94M.

## Null expectation
Δ expected: < 0.02 either direction; if it lands inside the 0.04 band, NULL. Heavy-param overhead makes this a slightly higher-risk test than 258/260 (0.21% param budget is non-trivial; if the optimizer can't fit the per-norm mean+scale in 92 steps, it could underperform).

## A/B
- Champion: `Tiny1M3MAlibiConfig` (val 6.2539, band 0.04, WIN gate < 6.2003)
- Treatment: same + `use_layernorm=True`
- Seed 42, no warmup, tiny1m3m
- Inline config: `@dataclass class C(Tiny1M3MAlibiConfig): use_layernorm=True`
- No new model code (flag already on box, line 1068 of configs/llm_config.py)

## Why this is staged
Fallback for if 256/257 (deepnet confirm) fails. Norm axis is closed-but-not-fully at 0.94M (Sub-LN/ReZero/LayerScale/Residual all null on depth-conditional levers; the global LN swap is the one not A/B'd on alibi).
