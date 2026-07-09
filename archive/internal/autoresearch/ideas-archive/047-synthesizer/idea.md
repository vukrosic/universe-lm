---
id: 047-synthesizer
status: needs-plan
round: 1
updated: 2026-06-11T01:34:07Z
transfer-risk: high
---

# 047 — Dense Synthesizer (zero-gated hybrid head)

## Source
Tay et al., "Synthesizer: Rethinking Self-Attention in Transformer Models" (arXiv:2005.00743), 2020. Variant pinned: **Dense Synthesizer** (paper §2.1) — the only one of the paper's four variants (Dense / Random / Factorized / Mixture) that roughly matches vanilla on language modeling; Random is paper-confirmed bad on LM (Table 2), Mixture is just Vanilla+Dense by another name.

## Mechanism
Add a **content-conditional, query-independent** score branch in parallel to the standard QK dot-product, mixed by a per-head sigmoid gate initialised to **−6** (sigmoid(−6) ≈ 0.0025, i.e. step-0 ≡ baseline). For each head:

- Dense Synthesizer branch: `S_dense[i,:] = W₂ · relu(W₁ · h_i)`, where `h_i` is the post-norm residual at position `i`, `W₁ ∈ ℝ^{d_head × d_model}`, `W₂ ∈ ℝ^{L_max × d_head}`. `S_dense` is the synthetic score matrix (the row depends only on the *content* at row `i`, not on any pairwise interaction with column `j`).
- QK branch: unchanged scaled dot-product.
- Mix: `A = softmax(gate · S_dense + (1 − gate) · S_qk)` where `gate = σ(g_h)`, `g_h ∈ ℝ` per head, init `−6`.

Causal mask applied after mix; `L_max = max_seq_len = 2048`. Adds ~`H · (d_model·d_head + L_max·d_head + 1)` ≈ 12k params/layer at our shape — ≪ 200 LoC, fits in `models/layers.py` MHA path.

## Scale evidence
Tay 2020 ran Dense Synthesizer on enc-dec MT and LM up to ~T5-base scale (~220M) but the LM result is **parity, not gain** — closest direct precedent for the bet here. No published >100M LM-pretrain *win* for any Synthesizer variant. **Transfer-risk stays `high`**, with a mechanistic argument (not a hedge): at tiny1m3m we run d_head=32 with 6 heads, where QK dot products are unusually noisy (small d_k); a learned, content-conditional, query-independent prior could complement noisy QK signal in a regime that 100M+ models don't see. If that mechanism is real, the gate learns mass >0.1 and we get a non-trivial Δ; if it isn't, gate decays to ~0 and we get a clean informative null at tiny scale.

**Closed-axis distinction:** `closed.md` retires `NSA / diff-attn / hybrid heads` at screen20m. Those are all **QK-derived** hybrids (NSA = sparse QK selection, diff-attn = QK − QK). Dense Synthesizer is the **first non-QK hybrid** filed on this repo — the score matrix is generated without any pairwise content interaction. The closed sweep does not constrain this; if anything, "all QK-flavoured hybrids lost" raises the prior that the next informative test is the one that drops QK from the second branch.

## Why it's worth a slot
**Bet (committed numbers, both required for an informative outcome):**

1. **Gate-mass prediction.** Mean over heads/layers of `σ(g_h)` at end of training:
   - `≤ 0.10` → model rejected the synthetic path → confirms QK content-based similarity is essential at tiny1m3m (informative null).
   - `0.10 – 0.30` → mild useful prior; expect val-loss Δ in `[−0.005, +0.005]` vs fire-ctrl.
   - `> 0.30` → synthetic path materially engaged; **must** be paired with a val-loss win to be informative (otherwise the gate is just absorbing noise).
2. **Val-loss-Δ bar** vs fire-ctrl baseline (`Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`, 6.3419 last week, ctrl-gap ~0.005):
   - Δ ≤ −0.01 with gate >0.10 → **WIN** (non-QK hybrid actually helps at this scale).
   - Δ ∈ [−0.005, +0.005] with gate ≤ 0.10 → **informative null** (gate said "no thanks", QK is enough).
   - Δ ≤ −0.01 with gate ≤ 0.05 → suspicious (effect must be coming from the extra params, not the synthetic branch); flag as uninformative.
   - Δ ≥ +0.01 → **reject** (Synthesizer family closed for this repo).

**Why a null still teaches us:** every prior hybrid-head we tried was QK-derived. A clean "gate-collapses-to-zero" result is the first evidence that the QK-vs-non-QK choice — not the sparsity pattern — is what mattered in the screen20m hybrid-head sweep. That reframes how the next hybrid (if any) gets pitched.
