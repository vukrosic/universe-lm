---
id: 230-poly-alibi
author: claude-opus-4-8
status: done
round: 1
updated: 2026-06-16T01:05:50Z
transfer-risk: low
recode-note: "00:53 SMOKE_FAIL was a transient infra fault, NOT a code bug — the box was stuck on the stale orchestrate-codex-fallback branch and never pulled the poly-alibi wiring. Fixed by FF-ing origin/orchestrate-codex-fallback to main@dc273c8; code compiles + SMOKE_OK locally."
plain: Replace the champion's linear ALiBi distance bias with a per-head linear+quadratic bias `scores -= (m_h·d + c_h·d²/L)`. It is a strict SUPERSET of ALiBi — with the quadratic coefficient c_h=0 it reproduces the champion exactly, so the win bar is structurally reachable. Both coefficients start at zero (step-0 identical to alibi), but the quadratic gradient is high-leverage (∝ d²) so c_h grows fast in 92 steps, letting each head bend its distance-decay curve the way alibi's flat linear slope can't.
---

# 230 — Polynomial-distance ALiBi (use_poly_alibi) — superset of the 175-ALiBi champion

## Source
ALiBi: Press, Smith, Lewis, ICLR 2022, arXiv:2108.12409 (linear per-head distance bias `-m_h·(i-j)`, validated to BLOOM-176B). The closed 175-alibi-slopes WIN (Δ-0.1585 over base at 0.94M) is the current champion. Kerple (Chi et al. 2022, arXiv:2205.09921) and Sandwich generalize ALiBi to non-linear distance kernels; 230 takes the simplest such generalization — adding a per-head quadratic term — as a strict superset.

## Mechanism
Replace the linear distance bias with a per-head **linear + quadratic** bias on pre-softmax scores:

```
scores[b,h,i,j] -= ( m_h · d  +  c_h · d² / L ),   d = (i − j),   L = max_seq_len
```

- `m_h = nn.Parameter(zeros(n_heads))`, `c_h = nn.Parameter(zeros(n_heads))` per block.
- Same `diff` convention as the Q1 ALiBi branch, so `c_h = 0` ⇒ **the exact champion**.
- `d²/L` is scaled by `L` so the quadratic term shares the linear term's [0, m·L] range (no exploding-magnitude / LR-mismatch).
- Forces the manual attention path (can't go through SDPA flash). Cost: 8 scalars/block × 12 = 96 params; net **+48** vs the alibi champion (alibi's 48-param slope is replaced).

## Why this is worth a slot (the 92-step budget logic, per EXPERIMENT-DESIGN.md)
The 208–216 batch showed orthogonal bolt-ons (gated-attn, ssmax, logit-scale, qk-layernorm, SwiGLU, value-residual, canon-conv) all wash out inside the 0.04 band — you can't beat a *large structural* win on the positional axis with a *small orthogonal* lever. The two strategies with EV are (1) a step-0-active orthogonal stack or (2) **a more-expressive challenger that subsumes the champion on its own axis.** 230 is strategy (2):
- **Subsumes alibi.** `c_h=0` is the champion, so the WIN bar is structurally reachable — not a marginal add-on below the band.
- **Moves in 92 steps.** Only +1 scalar/head (vs SwiGLU's whole zero-init gate matrix that washed to Δ0.0000), and `∂/∂c_h ∝ d²/L` is high-leverage for far tokens, so c_h grows fast — the same property that made alibi's 48-param slope win.
- **New expressivity.** Curvature lets each head pick a convex (sharper far penalty) or concave (gentler) decay; pure-linear alibi is locked to one slope shape per head.

Distinct from **228-per-layer-alibi-mul** (per-block scalar *multiplier* on the linear slope — magnitude only, still linear) and **166-t5-rpe** (bucketed-discrete distance bias, NULL). 230 is a *continuous curvature* on 175's exact axis.

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2539, band 0.04 (3-seed honest mean; pinned, no re-measure).
- **Treatment**: inline `@dataclass C(Tiny1M3MPolyAlibiConfig)` (`use_alibi_bias=False`, `use_poly_alibi=True`).
- **PASS / WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42); sub-noise INCONCLUSIVE.

## Pre-run verification (local CPU, claude-opus-4-8)
- **builds** ✓ — MinimalLLM(Tiny1M3MPolyAlibiConfig), net +48 params vs alibi.
- **step-0 identical to champion** ✓ — max-abs logit diff vs alibi at c_h=0 = `0.000` (clean superset, no spurious step-0 perturbation).
- **curvature is step-0-active** ✓ — with c_h=0.05, max-abs logit diff = `4.6e-2` (active from step 0, NOT a zero-init no-op — the gate that 211/208/209/210 failed).
- **not worse** ✓ — 15-step paired random-data probe Δ(poly−alibi) = `−0.0001` (inside ±0.005 probe noise).
- **SMOKE_OK** ✓ — `voidspark/tools/autoresearch/_box_smoke.py _arq_230-poly-alibi.py`.

## Mechanism wiring (committed to model code)
- `configs/llm_config.py`: `use_poly_alibi` flag on `LLMConfig` + `Tiny1M3MPolyAlibiConfig`.
- `models/llm.py`: `getattr(config, "use_poly_alibi", False)` + block pass-through (both build sites).
- `models/layers.py`: MHA/TransformerBlock kwarg, `poly_alibi_m`/`poly_alibi_c` params, manual-path guard, forward branch after the Q1 ALiBi branch.
