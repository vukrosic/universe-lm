# 013 — CoPE (Contextual Position Encoding)
_Auto-drafted 2026-06-10 from `autoresearch/ideas/013-cope/`._

## Abstract
- For each head, learn a small "importance" probe `p ∈ R^D`.
- Position offset between i and j = count of k in [i, j] where `dot(x_k, p) > threshold` (or a soft sigmoid variant).
- Add this offset to the relative-position bias term in attention.
- Implementation: a `CoPE` module (~50 LoC, bumped from the original 40 — see RoPE-audit below) that replaces the RoPE application in `models/attention.py`. RoPE removed when CoPE is on.
- Distinct from FIRE (009, WIN): FIRE has a fixed decay kernel on absolute position; CoPE has a *content-conditional* position. Different inductive bias. We test on tiny1m3m (seed 42). Verdict: UNKNOWN.

## 1 Introduction
This work re-implements and stress-tests the mechanism from Golovneva et al., "Contextual Position Encoding: Learning to Count What's Important" (arXiv:2405.18719, 2024, Meta). Position is computed *per-head* from the *content* of nearby tokens: the position offset between token i and j is the number of "important" tokens (those with high dot-product to a learned probe) between them — not the literal index distance. Drop-in alternative to additive/relative position encodings..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
- For each head, learn a small "importance" probe `p ∈ R^D`.
- Position offset between i and j = count of k in [i, j] where `dot(x_k, p) > threshold` (or a soft sigmoid variant).
- Add this offset to the relative-position bias term in attention.
- Implementation: a `CoPE` module (~50 LoC, bumped from the original 40 — see RoPE-audit below) that replaces the RoPE application in `models/attention.py`. RoPE removed when CoPE is on.
- Distinct from FIRE (009, WIN): FIRE has a fixed decay kernel on absolute position; CoPE has a *content-conditional* position. Different inductive bias.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | — | — |
| Treatment | — | — |

<details><summary>raw evidence.md</summary>

# 013 — CoPE (Content-aware Positional Encoding) — evidence

**Date**: 2026-06-09
**Tier**: tiny1m3m (0.94M params, 3M tokens)
**Box**: vast-34386 (RTX 3060)
**Seed**: 42 (one seed only, per project rule)
**Queue**: ctrl → 013-cope (FIRE+CoPE stacked) → ctrl2

## Results

| Run | Final Val Loss | Δ vs ctrl1 | Δ vs ctrl2 |
|---|---|---|---|
| ctrl | 6.3969 | — | — |
| **013** (FIRE + CoPE, stacked) | **6.4659** | **+0.0690** | **+0.0768** |
| ctrl2 | 6.3891 | — | — |

ctrl-to-ctrl gap: |6.3891 − 6.3969| = **0.0078**.

## Verdict — DRIFT (clear regression)

Treatment (6.4659) is **+0.069 to +0.077 worse** than both ctrls — far
outside the 0.0078 ctrl-to-ctrl gap. Stacking CoPE on FIRE produces a
**large negative effect** (vs in-session plain baseline: +0.069; vs the
closed 009 FIRE-alone WIN at 6.3234: **+0.143** — CoPE added on top of
FIRE is *worse than no positional encoding at all*).

This kills the stacked lever for tiny1m3m. The CoPE bias + FIRE bias
likely interact destructively at this scale (both add per-position bias
to the attention scores; the combined bias is too large).

## Note (composition)
- 009 FIRE alone: 6.3234 (WIN, closed)
- 013 FIRE+CoPE: 6.4659 (DRIFT, this run)
- Difference: +0.143. **CoPE stacked on FIRE ruins the FIRE win.**

## Log files
- `~/arq/logs/ctrl.log`
- `~/arq/logs/013-cope.log`
- `~/arq/logs/ctrl2.log`

</details>

## 5 Discussion
Verdict not yet recorded; this draft is preliminary.

## References
1. Golovneva et al., "Contextual Position Encoding: Learning to Count What's Important" (arXiv:2405.18719, 2024, Meta). Position is computed *per-head* from the *content* of nearby tokens: the position offset between token i and j is the number of "important" tokens (those with high dot-product to a learned probe) between them — not the literal index distance. Drop-in alternative to additive/relative position encodings.

---
_Status_: **done** · _Verdict_: **UNKNOWN** · _Closed_: 2026-06-09T16:36:51Z
