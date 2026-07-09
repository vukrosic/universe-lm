---
id: 008-gated-deltanet
status: rejected
round: 1
updated: 2026-06-09T03:17:01Z
---

# 008 — Gated DeltaNet (linear attention with delta rule + gate)

## Source
Yang et al., "Gated DeltaNet: Sequence Modeling with Linear Attention" (2024). Reference impl: `fla-org/flash-linear-attention` (gated_deltanet). (Taste-reviewer: confirm arXiv ID; the gated-delta variant is the 2024-2025 follow-up to DeltaNet by the same group.)

## Mechanism
Replace softmax attention with a linear recurrence: `S_t = g_t * (S_{t-1} + k_t^T ⊗ (v_t - S_{t-1} k_t))`, `o_t = S_t q_t` (the delta rule replaces `v_t` with the *residual* `v_t - S_{t-1} k_t` so stored state isn't overwritten by uninformative keys; `g_t = σ(W_g x_t)` is a per-token input-dependent gate). Computed in parallel form via a chunkwise prefix-scan, so training is still O(n) memory and O(n) compute per head (not O(n²)). Implementation: ~150-180 LoC borrowing from `fla-org/flash-linear-attention`'s reference kernel; train path goes through the parallel scan, not the recurrent form.

## Why it's worth a slot
Softmax attention is O(n²) in compute and KV-cache size; gated DeltaNet is O(n) and KV-free. If val loss holds (or improves) at our scale, this is a free compute/latency win and a real architectural lever (not a HP tweak). Distinct from 004-retnet-retention (retention is a simpler decay-only linear attention; the *delta rule* + *gating* are the new mechanisms and they're the part that closes the softmax gap). Transferable across scale (mechanism is shape-agnostic in head dim), identity-safe (gate is zero-init at step 0, so output is zero and the residual stream is unchanged — the loop is bit-exact at step 0 by construction). A null result still teaches us the linear-attention gap at our scale, which is useful prior. Tier: screen20m+ (architectural change to attention; need enough tokens to see a val-loss signal on a 135M target).
