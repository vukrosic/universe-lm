---
id: 214-ssmax-alibi
author: claude-opus-4-8
status: needs-run
round: 1
updated: 2026-06-15T13:20:00Z
transfer-risk: low
plain: Stack Scalable-Softmax (SSMax) on top of the ALiBi champion. SSMax multiplies the attention logits by a per-head learnable scalar times log(sequence length) before softmax, so attention doesn't over-flatten as context grows. ALiBi biases WHICH positions a head attends to; SSMax rescales the softmax TEMPERATURE by length — different axis. It is STEP-0 ACTIVE (unlike 211-SwiGLU's zero-init gate that washed to Δ0.0000), so it can register inside the 92-step budget.
---

# 214 — Scalable-Softmax (use_ssmax) on the 175-ALiBi Champion

## Source
SSMax — Nakanishi 2025, "Scalable-Softmax Is Superior for Attention" (arXiv:2501.19399). `attn = softmax(s · log(n) · QKᵀ/√d)`, per-head learnable scalar `s`.

## Why this is high-EV after 208–211 all washed out
**211-SwiGLU (NULL Δ0.0000)** proved zero-init levers can't grow in 92 steps. SSMax is **step-0 active** (local probe: max-abs logit diff **0.042** vs alibi at step 0; +48 params = one scalar per head per layer) and on a **different axis** from alibi:
- **ALiBi**: additive *positional* bias on the scores (function of t−s).
- **SSMax**: multiplicative *temperature* on the scores as a function of *length* — controls softmax sharpness, not position.

Length-aware temperature is a known small-scale stabiliser; complementary to alibi's distance prior.

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2403, band 0.04 (pinned, no re-measure).
- **Treatment**: inline `@dataclass C(Tiny1M3MAlibiConfig): use_ssmax=True`.
- **PASS / WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42); sub-noise INCONCLUSIVE.

## Config (inline, no llm_config.py edit)
`_arq_214-ssmax-alibi.py` — `@dataclass` subclass (decorator required, per `_arq_161-dyt-temp.py`).

## Pre-run verification (local, claude-opus-4-8)
- builds, flag set ✓ (+48 params, one scalar/head/layer)
- **step-0 active**: max-abs logit diff 0.042 vs alibi ✓
- 15-step probe: Δ+0.0034 (within random-data noise; active and roughly neutral) ✓
