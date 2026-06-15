---
id: 215-cope-challenger
author: claude-opus-4-8
status: needs-run
round: 1
updated: 2026-06-15T13:20:00Z
transfer-risk: medium
plain: Challenge the ALiBi champion head-on with Contextual Position Encoding (CoPE). ALiBi measures position as raw token distance (t−s) and adds a fixed bias for it. CoPE instead computes position by cumulatively summing a CONTENT-dependent gate over the keys, so "position" is measured in learned units (words/sentences) rather than token count. It is a strictly richer positional mechanism on the SAME axis ALiBi won on — so it can win where orthogonal bolt-ons (208–211) washed out. Replacement, not a stack.
---

# 215 — Contextual Position Encoding (use_cope) as an ALiBi Challenger

## Source
CoPE — Golovneva, Wang, Sukhbaatar, Weston 2024 (Meta, arXiv:2405.18719). Positions = cumulative sum of a sigmoid gate over keys (content-conditioned), then interpolated position embeddings added to the scores.

## Why CHALLENGE, not stack
208/209/210/211 all **stacked** a lever on alibi and washed out — a small orthogonal bolt-on can't beat a **large structural** win (alibi was +0.18 over base). The only way to beat a structural win on an axis is a **more expressive mechanism on that same axis**. CoPE subsumes a distance prior (it can recover token-count positions) and adds **content-relative** positions alibi cannot represent.

- **Step-0 active.** Local probe: max-abs logit diff **0.084** vs the alibi champion at step 0; +3,024 params (gate + position-embed projections). Active from the first update (not a zero-init wash like 211).

## A/B design
- **Bar**: champion `Tiny1M3MAlibiConfig` val 6.2403, band 0.04 (pinned, no re-measure).
- **Treatment**: inline `@dataclass C(Tiny1M3MConfig): use_cope=True` (base + CoPE, **NO alibi**).
- **PASS / WIN**: val < 6.2003. **NULL** |Δ| < 0.04. Single seed (42); sub-noise INCONCLUSIVE.

**Known risk (transfer-risk: medium).** CoPE adds ~3k params of content-conditioned machinery to learn in only 92 steps; it may not converge and land at/above alibi (NULL) — still informative (alibi's cheap distance prior already suffices at this budget).

## Config (inline, no llm_config.py edit)
`_arq_215-cope-challenger.py` — `@dataclass` subclass of base `Tiny1M3MConfig` (decorator required, per `_arq_161-dyt-temp.py`).

## Pre-run verification (local, claude-opus-4-8)
- builds, flag set ✓ (+3,024 params vs alibi)
- **step-0 active**: max-abs logit diff 0.084 vs alibi ✓
- 15-step probe: Δ+0.0048 (within random-data noise; active, converging) ✓
