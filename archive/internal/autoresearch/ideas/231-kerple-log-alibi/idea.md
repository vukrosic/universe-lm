---
id: 231-kerple-log-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T01:31:09Z
recode-note: "01:20 SMOKE_FAIL was transient infra, NOT a code bug — autosync committed kerple to main (f360b33) but the box tracks orchestrate-codex-fallback, which hadn't been FF'd yet. FF-pushed; code compiles + SMOKE_OK locally."
transfer-risk: low
plain: Replace the champion's linear ALiBi distance bias with a per-head CONCAVE log kernel `scores -= m_h·log(1 + r_h·d)` (Kerple). Linear alibi penalizes distance proportionally; Kerple's log penalizes far tokens far more gently, so distant context decays slowly. This is the opposite curvature sign to 230-poly-alibi (convex quadratic, which came back Δ−0.0111 right-direction NULL). m_h starts at 0 (step-0 identical to alibi) but is high-leverage and grows fast in 92 steps like alibi's slope.
---

# 231 — Kerple log-distance ALiBi (use_kerple_log)

## Source
Kerple: Chi, Fan, Rudnicky, Ramadge, NeurIPS 2022, arXiv:2205.09921 — kernelized relative position embeddings; the log variant `−m·log(1+r·|i−j|)` generalizes ALiBi (`−m·|i−j|`) to a concave kernel and outperforms it on language-modeling perplexity / extrapolation in the paper. Champion is 175-alibi-slopes (linear, Δ−0.1585 over base at 0.94M).

## Mechanism
Per-head concave log distance bias on pre-softmax scores:

```
scores[b,h,i,j] -= m_h · log(1 + r_h · d),   d = (i − j) ≥ 0,   r_h = softplus(kerple_r_raw)
```

- `m_h = nn.Parameter(zeros(n_heads))` ⇒ bias = 0 at init ⇒ **step-0 == champion/base**.
- `r_h = softplus(raw)`, raw init 0 ⇒ r ≈ 0.693 (kept > 0; tunes inner distance scale).
- `d = (i−j).clamp(min=0)` — future cells get d=0 ⇒ log(1)=0 (and are masked anyway).
- Forces the manual attention path. Cost: 8 scalars/block × 12 = 96; net **+48** vs alibi (replaces the 48-param linear slope).

## Why this is worth a slot
Same EV logic as 230 (the only axis that moves the needle is the positional one; a *more-expressive challenger on it*, not an orthogonal bolt-on). 230 added **convex** curvature and came back Δ−0.0111 (right direction, inside the 0.04 band). 231 tests the **opposite** curvature sign — a *gentler*-than-linear far-token decay. Together 230 + 231 bracket "is the optimal tiny1m3m distance decay sharper or gentler than linear ALiBi?" — a real attribution result either way.
- **Step-0 identical, high-leverage.** m_h grows fast from 0 like alibi's slope (the property 211-SwiGLU's zero-init matrix lacked).
- **Literature-backed.** Kerple-log beats alibi in the source paper; this is the first test at the 0.94M / 92-step budget.

Distinct from 230-poly-alibi (convex superset of linear) and 166-t5-rpe (bucketed-discrete, NULL).

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2539, band 0.04 (pinned, no re-measure).
- **Treatment**: inline `@dataclass C(Tiny1M3MKerpleLogConfig)` (`use_alibi_bias=False`, `use_kerple_log=True`).
- **PASS / WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42); sub-noise INCONCLUSIVE.

## Pre-run verification (local CPU, claude-opus-4-8)
- **builds** ✓ — MinimalLLM(Tiny1M3MKerpleLogConfig), net +48 params vs alibi.
- **step-0 identical to champion** ✓ — max-abs logit diff vs alibi at m_h=0 = `0.000`.
- **active from step 0** ✓ — with m_h=0.05, max-abs logit diff = `3.8e-3` (>0, not a zero-init no-op).
- **SMOKE_OK** ✓ — `voidspark/tools/autoresearch/_box_smoke.py _arq_231-kerple-log-alibi.py`.

## Mechanism wiring (committed to model code)
- `configs/llm_config.py`: `use_kerple_log` flag on `LLMConfig` + `Tiny1M3MKerpleLogConfig`.
- `models/llm.py`: `getattr(config, "use_kerple_log", False)` + block pass-through (both build sites).
- `models/layers.py`: MHA/TransformerBlock kwarg, `kerple_m`/`kerple_r_raw` params, manual-path guard, forward branch after the 230 poly-alibi branch.
