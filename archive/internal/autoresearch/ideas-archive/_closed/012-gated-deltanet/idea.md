---
id: 012-gated-deltanet
status: rejected
round: 1
updated: 2026-06-09T11:35:15Z
---

# 012 — Gated DeltaNet

## Source
Yang et al., "Gated DeltaNet: Sequence Model with Adaptive State Expansion" (2025, recent). Linear attention with the *delta rule* (update state by *difference* between prediction and actual, like a fast-weight controller) + an input gate (decide how much new info to write). Strictly stronger than RetNet's pure-decay retention (004): delta rule solves the "can't un-write" problem of pure decay, gate solves the "always write" problem of pure delta.

## Mechanism
- Replace the softmax attention in `models/layers.py` with a gated delta-rule linear attention block: `S_t = α_t · (I - β_t k_t k_t^T) S_{t-1} + β_t v_t k_t^T`, where α_t is the input gate and β_t is the write gate (both small sigmoids).
- Two extra projections per head: α (input gate), β (write gate). Same KV projection structure.
- Implementation: chunk-wise parallel form (à la Mamba/DeltaNet papers) for the forward pass; recurrent form for inference. ~50-100 LoC.
- Distinct from 004 (RetNet) in two ways: delta rule (not pure decay) and explicit input gate.

## Why it's worth a slot
- **Strong recent result** (top-3 in long-context arena benchmarks at the time of paper).
- **Orthogonal to 004** — 004 is decay-only; this is decay + delta + gate. If 004 is null and 012 wins, we know the delta rule mattered. If both win, we have additive / modular lever.
- **O(n) compute and memory** for the attention path — could be a real efficiency win if it transfers.
- **Risk**: transfer to tiny scale (0.94M params) is the big unknown. Linear-attention benefits often show up at longer context / larger model. The taste critic should weight this.

## Hypothesis
Δ in [−0.02, −0.06] val loss IF it transfers to tiny1m3m. If null at tiny scale, that's a clean negative — the mechanism is correct but doesn't fire at this scale. Either outcome is informative.

## Wiring
- New file: `models/gated_deltanet.py` — block + chunk-wise kernel.
- `LLMConfig.use_gated_deltanet: bool = False` (replaces MHA when True, only in `models/transformer.py`).
- Smoke test: must match MHA forward shape (B, T, D).
- Pass/fail: PASS ≤ −0.01 vs V+q+SWA+HighRoPE ctrl. NULL = |Δ| < 0.01. DRIFT > +0.02 (linear attn can hurt at small scale — explicitly watch for it).

## Notes
- Closely related to 004 (retention) and 008 (gated-deltanet in PENDING — this filing subsumes 008). Mark 008 as superseded.
- Kernel choice: prefer the reference PyTorch implementation over a custom triton kernel for the first run; revisit kernel only if the reference is the bottleneck.
