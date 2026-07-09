---
id: 004-retnet-retention
status: done
round: 3
updated: 2026-06-09T09:36:29Z
---

# 004 — RetNet retention (linear-attention alternative)

## Source
Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei (Microsoft) — "Retentive Network: A Successor to Transformer for Large Language Models" (arXiv:2307.08621, Jul 2023). Code: https://aka.ms/retnet. ⚠️ 2023 paper; field has moved (Mamba, GLA, RWKV-7). Filing because: linear-attention alternatives are untested in this repo, and RetNet is the most-cited baseline with a working impl.

## Mechanism
Replaces softmax attention with a retention kernel: per-head learnable decay γ < 1, position-dependent mask, no softmax. Three equivalent modes: parallel (training), recurrent (inference O(1)/step), chunkwise-recurrent (long sequences, linear complexity). Single Q/K/V projection + custom retention kernel — < 200 LoC. Paper claim: "favorable scaling, parallel training, low-cost deployment."

## Pass / fail bar
- pass: screen20m val ≤ 4.5864 (vs current best 4.6364, target Δ = −0.05). This is the high-EV scenario.
- fail: screen20m val > 4.6364 (worse than V+q+SWA+HighRoPE) — likely if linear attention loses at 10M scale
- noise: |Δ| ≤ 0.10 (screen20m noise band) — treat as inconclusive
- expected Δ ≈ −0.04 to −0.06; lower values are below the single-seed noise floor

## Routing (committed)
RetNet **replaces the attention module only**. The V-embed and Q-gain/K-gain levers are orthogonal to attention math and are preserved. Concretely:
- **Swapped**: the attention block (`MultiHeadAttention` in `models/layers.py`) — the softmax + QK^T/sqrt(d_k) is replaced by the retention kernel.
- **Kept**: V-embed (`use_value_embed`), Q-gain (`use_q_gain`), K-gain (`use_k_gain`), SWA (`use_sliding_window`), HighRoPE (`rope_base=500000`). The V+q+SWA+HighRoPE baseline applies as the control; only the inner attention math changes.
- **Kept**: the LM head, embeddings, FFN, norms. Only attention differs.
- This is the right call: it isolates "softmax vs retention" cleanly while holding all the other wins constant.

## LoC estimate (realistic breakdown)
The "< 200" line in the source is the kernel only. Realistic integration LoC:
- **Kernel body**: ~80 LoC (parallel/retention kernel with learnable γ, position mask, per-head decay). Paper's reference impl.
- **Model integration**: ~80 LoC (attention block rewrite, RoPE hook decision — γ × RoPE(Q) — and KV-cache stub for the recurrent path).
- **Trainer hooks**: ~40 LoC (no GQA reshape, no flash-attn path; chunkwise path uses a custom Triton kernel or PyTorch fallback).
- **Total: ~200 LoC, at the budget edge.** If integration exceeds 250 LoC, the idea splits: kernel-only PR + integration PR, or downscope to a probe (run a tiny synthetic to verify the kernel works, don't try to land it as the production attention).

Only the parallel + chunkwise paths ship in v1; the recurrent (inference) path is a stub.

## Seed protocol
Seed 42, single seed, per the pipeline hard rule. |Δ| ≤ 0.10 is the noise band; a sub-noise result is logged inconclusive, not re-seeded. A null at |Δ| < 0.04 with seed 42 is *itself* the evidence the kernel doesn't catch up at this scale.

## Transfer argument
The mechanism that scales is the **O(N) memory and O(N) compute** of the retention kernel. At 135M+ with long sequences, softmax attention's O(N²) memory becomes the dominant cost; retention's linear profile is the compute-advantage story. The unknown is whether the kernel is competitive with softmax at the *quality* level — softmax attention at 10M-20M is extremely well-tuned in this repo (V+q+SWA+HighRoPE at 4.6364 is a tight baseline), and the retention kernel may not catch up at small scale. The honest transfer story: "expect null or marginal at screen20m; expect a compute-advantage at 135M+ if the kernel works at all."

## Field-moved caveat — why RetNet is the right probe
RetNet is the most-cited linear-attention baseline with a working impl and clean math. Mamba is a state-space model underneath (different mechanism, not a fair comparison in the "softmax vs retention" question). GLA and RWKV-7 are closer cousins to RetNet but neither is in the stack. If we're testing the *class* of linear-attention alternatives, RetNet is a representative.

## Closed-list migration rule for linear-attention nulls
A null at screen20m → close the lever, re-test only if a 135M+ compute grant materializes. Do NOT file a Mamba variant as a re-probe of the same lever (different mechanism family, different question).
