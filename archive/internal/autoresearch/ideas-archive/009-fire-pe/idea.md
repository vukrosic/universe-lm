---
id: 009-fire-pe
status: done
round: 1
updated: 2026-06-09T09:36:32Z
---

# 009 — FIRE positional encoding (functional-interpolation relative PE)

## Source
Li et al., "Functional Interpolation for Relative Positional Encoding" (NeurIPS 2023, arXiv:2306.02613). Reference: original paper repo.

## Mechanism
Add a learnable position-dependent bias to attention logits: `bias(i,j) = γ(i-j) · f(φ(x_i), φ(x_j))` where `γ` is a fixed Lp-norm kernel (monotone decay in relative distance) and `f` is a small MLP over learned projections of the query/key token embeddings. The bias is *input-dependent* on content (via `φ`) but the position kernel is fixed, so the model gets context-sensitive positional bias without losing the no-max-len property of pure relative PE. Implementation: drop-in for RoPE — same shape (additive bias on logits) — ~30-50 LoC for the kernel + MLP + bias-add into attention. No new parameters in the attention output path; the MLP and per-head learnables are tiny.

## Tier, seed
**tiny1m3m, seed 42 only.** Per the new pipeline rules (🔴 one tier / one seed), this idea runs at tiny1m3m (0.94M params · 3M tokens, seed 42). No screen20m, no multi-seed, no seed sweeps. A sub-noise effect is **inconclusive, not real** — log it and move on.

## Pass / fail bar (tiny1m3m, V+q+SWA+HighRoPE control 6.4287)
- pass: tiny1m3m val ≤ 6.4237 (Δ ≤ −0.005)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 (single-seed, tiny1m3m) — treat as inconclusive
- expected Δ ≈ −0.005 to −0.02; |Δ| ≤ 0.005 is noise / inconclusive
- control source: `autoresearch/queue.md` Remote run log row 1 (tiny1m3m ctrl, 1B data, T4, 6.4287)

## Why it's worth a slot
RoPE is closed-by-sweep (500k base won); FIRE is the strongest *non-RoPE* relative PE in the 2023-2024 literature, with a reported SoTA on length-extrapolation. FIRE has been the leading non-RoPE relative PE since 2023; no clear successor in 2024-2025 has dethroned it. The bet: the content-aware component (via `φ`) lets the model learn that certain tokens (e.g. paragraph breaks, function words) should get a different effective distance, which RoPE's pure-rotation scheme can't express. If true, we get a small val-loss win on the train distribution. Transferable (mechanism is positional-only), identity/zero-init safe (the `γ` is a fixed kernel — no learnable weight init change — and `φ` can be zero-init to start as pure RoPE-equivalent).

## Transfer argument / future-work
Length-extrapolation is the headline upside of FIRE in the paper, but tiny1m3m is a 3M-token fixed-length run — we will NOT see length-extrapolation at this tier. The val-loss bar above is train-distribution loss only. Length-extrapolation is left as a future test: if FIRE passes tiny1m3m, the next step is to revisit at a longer-context tier (currently out of scope per the new pipeline rules).
