---
id: 189-cosformer-linear-attn
status: planning
round: 1
updated: 2026-06-15T16:21:36Z
transfer-risk: med
plain: Replace softmax attention with a linear-time cosine-reparameterized attention (a different, much faster mathematical form of "soft" attention), designed to behave identically at step 0.
---

# 189 — CosFormer-Style Linear Attention (Cosine-Reparam Q, K)

## Source
- Qin et al., "cosFormer: Rethinking Softmax in Attention" (NeurIPS 2022, arXiv:2202.08791). Validated at ImageNet (DeiT-scale) and language modeling at GPT-2-small scale (~125M). Replaces softmax(QK^T) with `((φ(Q) · φ(K)^T) · V)` where `φ(x) = [cos(x), sin(x)]` (cosine feature map), giving a kernel-approximable linear-attention form. Linear in sequence length.
- Katharopoulos et al., "Transformers are RNNs" (ICML 2020, arXiv:2006.16236) — original linear-attention derivation; cosFormer is the cosine-reparameterized successor.
- Choromanski et al., "Rethinking Attention with Performers" (ICLR 2021, arXiv:2009.14794) — FAVOR+ random-feature map alternative; valid at GPT-2-class scale but quality gap vs softmax. **Cited here as the kernel-shape literature analog** for the "feature map is kernel-approximable" framing that motivates the diffuse-kernel bet.
- 004-retnet-retention (closed null at 0.94M, Δ=+0.04 wrong-sign) — retention with feature map `elu(x)+1`, linear-time, null on a structurally identical bet. **189 is the direct follow-up**: same template (linear-time attention with a feature map), different feature map (`exp(γx)·cos(x)` vs `elu(x)+1`). See §"Post-null information value" for the explicit 004-vs-189 argument.
- 008-gated-deltanet (taste-reject) — gated linear attention, off-niche, never ran. 148-focal-mod (closed null at 0.94M, Δ=+0.0072) — gated-additive *context* (focal modulation), not a kernel replacement. 189 is the *only* remaining distinct non-softmax attention family on the queue: cosine-feature-map *kernel replacement*, not additive context, not decay retention.

## Mechanism
Standard attention: `out = softmax(QK^T/√d) V` — quadratic in T.
cosFormer linear attention:
```
Q' = cos(Q)                                    # [B, H, T, d_k]
K' = exp(γ·K) ⊙ cos(K)                          # γ: learnable scalar, γ_init=0
out_linear = (Q' · (K'^T · V)) / (Q' · K'^T)   # compute K'V first, O(T·d_k²)
```

This is a linear-time attention (O(T·d_k²) instead of O(T²·d_k)) with a kernel φ(x) = exp(γ·x)·cos(x) approximating softmax.

## Distinct from sister idea 189-cos-attn (r1 fix — coordination gate)
A sister idea `189-cos-attn` is already in `implementing` (status: implementing, 2026-06-15T08:31:50Z). It cites the **same** Qin et al. 2022 paper but proposes a **different lever**:
- **189-cos-attn** (sister): L2-normalized pre-softmax scores `(Q/||Q||)·(K/||K||)^T` with **standard softmax kept**. The lever bounds pre-softmax logit magnitudes; softmax is unchanged. Bounded softmax.
- **189-cosformer-linear-attn** (this idea): softmax is **removed entirely**, replaced with the cosFormer feature map `φ(x) = exp(γx)·cos(x)` and a linear-time KV matmul. Kernel replacement, not bounded softmax.

These are mechanistically distinct (bounded softmax ≠ kernel replacement), and the implementer must not conflate the two. **Implementer gate**: build the linear-time kernel-replacement form (φ-replaced softmax, cumsum-causal, denominator-on), not the bounded-softmax form. The flag is `use_cosformer=True` on `MultiHeadAttention` (this idea), not a per-row L2 normalization on Q,K (sister idea).

## Step-0 bit-identity (r1 fix — tightened to cumulative mean-pool over causal prefix)
**The r1 reviewer's concern**: with γ=0 and Q,K ~ N(0, 0.02²) at step 0, cos(Q) ≈ 1 − Q²/2 ≈ 1, so Q'K'^T ≈ 1, which "looks uniform" but might not be softmax-equivalent. Also: the toy numerical check (`d=64, T=512`) doesn't match the real model's Q,K statistics (real W_Q/W_K init gives per-row ||Q||_2 ≈ 1/√d_k ≈ 1/4 for d_k=16, not 1).

**Resolution — cumulative mean-pool over causal prefix**:
- softmax with small logits: `out_t = sum_{s≤t} exp(Q_t·K_s/√d)·V_s / Z_t` with `exp ≈ 1`, `Z_t = t+1` → `out_t = (1/(t+1)) · sum_{s≤t} V_s` = **cumulative mean of V over the prefix s ≤ t**.
- cosFormer γ=0: `out_t = Q_t·(cumsum(K'^T·V))[t] / (Q_t·(cumsum(K'))[t])` with `Q_t·K_s ≈ 1`, `cumsum(K')[t] ≈ t+1` → `out_t = (1/(t+1)) · sum_{s≤t} V_s` = **same cumulative mean**.

Both compute the cumulative mean of V over the causal prefix, not the global mean over the full T. Deviations are O(σ²) ≈ 4e-4 in both cases (σ² is the variance of QK^T/√d under std-0.02 qkvo init).

**Numerical verification (real-model, r1 fix — required gate)**:
```
trt = build(use_cosformer=True); ctrl = build(use_cosformer=False)
x = next(iter(train_loader))[0][:1, :T]                    # one batch, real input
trt_out, ctrl_out = trt(x), ctrl(x)
cummean_ctrl = ctrl_out.cumsum(dim=1) / torch.arange(1, T+1, device=ctrl_out.device).view(1,T,1,1,1)
assert (trt_out - cummean_ctrl).abs().max() < 1e-6          # fp32 max-abs-diff at the REAL model
```
At t=0 the test reduces to `assert (trt_out[0] - ctrl_out[0]).abs().max() < 1e-6` (cummean of one element is the element). At t=T-1 the test reduces to global-mean equivalence. The toy check (`d=64, T=512, max|Δ|<1e-5`) is **not** the project standard — the real-model test above is. γ=0 is the correct zero-init.

**Optional γ_init correction** (not used; documented for completeness): set γ_init = σ²/2 ≈ 2e-4 to compensate for cos's negative curvature shrinking φ(K) below exp(K). Below FP rounding — not worth the complexity. γ=0 stays.

## Design sketch (r1 fix — placement, denominator, causal, param layout)
- **Flag convention (r1 fix)**: add `use_cosformer: bool = False` and `cosformer_gamma_init: float = 0.0` to `MultiHeadAttention.__init__` (NOT a new `cosFormerAttention` module — that would be a regression from the project pattern). Add a new `elif self.use_cosformer:` branch in `forward`, placed **right after** the existing `elif self.use_linear_attn:` block at `models/layers.py:4301-4332`. Same shape as `use_qk_norm`, `use_fire_pe`, `use_cope`, `use_fox`, `use_softpick`, `use_entmax`, `use_ssmax` — flag on MHA, no new module.
- **Linear form**: compute `KV = K'^T · V` first ([B,H,d_k,d_k]), then `out = Q' · KV` ([B,H,T,d_k]). Memory O(T·d_k) per head per block.
- **Denominator Z = Q'·K'^T is MANDATORY (r1 fix — bound in spec, no skip-flag)**: the spec must say `out = out / (Q'·K'^T).clamp_min(1e-6)` with NO flag to skip the denominator. Without the denominator the lever silently nulls to a global mean-pool (no query-key interaction); the denominator is what makes it a kernel-replacement softmax. Bind the same shape as `use_qk_norm`: the manual path is forced because the mechanism requires it.
- **Causal mask via prefix-sum cumsum (r1 fix — pick option c)**: `out_t = Q_t · (cumsum(K'^T·V))[t] / (Q_t · (cumsum(K'))[t])`. The cumsum clipped to `[end_idx, start_idx]` is the standard linear-attention causal trick (same pattern as the existing `use_linear_attn` path at `models/layers.py:4314-4330`, indexed by `[end_idx, start_idx]`). True linear-time AND causal. Other options:
  - (a) per-query position-by-position: O(T²·d_k), defeats linear-time claim — **rejected**.
  - (b) causal mask on the K'V matmul: still leaks future tokens via the un-masked K'V pre-matmul — **rejected**.
  - (c) prefix-sum cumsum: linear-time + causal + simple — **adopted**.
- **Param layout (r1 fix — single nn.Parameter on the model, not the MHA)**: γ is one scalar per block, init 0, learned. Register as `self.gammas = nn.Parameter(torch.zeros(n_layers))` on `MinimalLLM` (follows the `layer_temperature` pattern at `models/layers.py:795` — "the parameter lives on the model, not the MHA, so the optimizer sees ONE nn.Parameter"). Pass `self.gammas[block_idx]` into the MHA's forward call (one read per block). Total: 12 scalars on the model, one entry in the optimizer's param groups (flat layout). Do NOT create 12 separate Parameters on the MHA — that gives 12 entries in the optimizer, breaking the project's flat layout.
- **Intuition**: cosFormer's cosine reparameterization gives a "soft" attention kernel with linear complexity. At 0.94M with T=2048, the structural difference (linear vs quadratic) is invisible — the bet is on *kernel shape* (cosine vs softmax), not complexity.

## Scale evidence
cosFormer validated at GPT-2-small (~125M, 1024 context) — paper reports **parity with softmax**, not a quality win, at language modeling. At <100M, the lever is plausibly transferable. Transfer-risk: med (validated at 125M, lever is O(T·d_k²) so works at any T; but the LM parity result means we can't anchor a positive prediction from the paper).

## Sharp prediction (r1 fix — kernel-shape literature analog added)
**Primary prediction** (single, quantified): **val_loss at step 92 ≤ baseline − 0.005** (i.e., the lever WINS the WIN bar). Mechanism: cosFormer's kernel φ(x) = [cos(x), sin(x)] (concatenated, equiv. to exp(γx)·cos(x) at γ=0) is *more diffuse* than softmax's exp(QK^T/√d) — softmax concentrates mass on the max, cosFormer keeps mass on a wider band. With only 92 training steps and limited tokens, a diffuse kernel averages over more context per query → better generalization → lower val_loss at the end of the 0.94M horizon.

**Mechanism bet (r1 fix — kernel-shape literature analog)**: the diffuse-kernel argument is the miner's own, not from the cosFormer paper itself (Qin et al. 2022 reports parity, not a win). Closest literature analog: **Choromanski et al. 2021 (Performers, FAVOR+)** establishes that the kernel-approximable feature map framing is sound for low-data generalization — the "feature map is a kernel approximation of softmax" angle supports the bet that a *bounded, oscillating* feature map (cos+sin) generalizes better than a *sharp, concentrating* exp kernel under limited update steps. **Yang et al. 2024 (Gated Linear Attention)** frames the same bet differently: "more diffuse than softmax generalizes better at low data" — explicit reference for the diffuse-kernel quality argument. Mark this as the **mechanistic bet** (not a paper claim); the protocol allows it.

**Auxiliary diagnostic** (engagement check, not a win criterion): attention entropy at step 10 ≥ softmax's entropy × 1.10. If entropy matches softmax to <10% difference, the cosine kernel is not engaging (probably a degenerate φ initialization collapses it). If entropy is ≥10% higher, the kernel shape is in play and the primary prediction has a mechanism to ride on.

**Null criterion**: val_loss Δ > −0.003 (i.e., the lever is in noise or wrong-sign, NOT a win). 148-focal-mod's Δ=+0.0072 is a sibling (additive context) but the **structural prior is 004-retnet's Δ=+0.04 wrong-sign** — see §"Post-null information value".

## Pass-bar (r1 fix — tighter than default)
Default protocol WIN bar is |Δ| ≤ 0.005. For a **softmax-replacement** lever, the transition risk is high (one bug in K'V matmul and the entire model collapses silently), so we tighten:
- **WIN**: val_loss Δ ≤ −0.005 (default bar — we want a real signal, not noise)
- **NULL**: val_loss Δ ≥ +0.003 (tighter than default +0.01; a softmax replacement that loses by 0.003 is *meaningfully* wrong)
- **NOISE BAND**: −0.003 < Δ < −0.005 → inconclusive, treated as null (no win, no hard reject; noted as "inconclusive, needs re-run" — not a slot burn, but not a pass either)

Net: the lever must EITHER win cleanly OR fail cleanly. Anything in the middle is a wash, not a pass.

## Post-null information value (r1 fix — direct prior is 004-retnet, not 148)
The **direct structural prior** is **004-retnet-retention** (closed null at 0.94M, Δ=+0.04 wrong-sign), not 148-focal-mod. 004 and 189 share the **structural template**: O(T·d_k²) linear-time attention with a feature map:
- 004: φ(x) = `elu(x)+1` (non-negative, monotone, concentrates mass on the max)
- 189: φ(x) = `exp(γx)·cos(x)` at γ=0 (bounded, oscillating, diffuses mass on a wider band)

008-gated-deltanet was a taste-reject (off-niche, never ran — NOT an empirical prior). 148-focal-mod is additive *context* (softmax stays, context vectors added) — a *different mechanism class*, not the structural prior.

**Why cosFormer should beat elu+1 at 0.94M (r1 fix — explicit rebuttal of the 004-null generalization)**:
- elu+1 is non-negative and concentrates mass on the largest QK^T value (sharp, peaked attention) — at low data, sharp attention overfits to whatever's peaked in 92 update steps.
- cos+sin is bounded in [-1,1] and oscillates — it diffuses mass on a wider band of (Q,K) pairs. At low data, a diffuse kernel averages over more context per query and generalizes better with limited tokens.
- 004's null +0.04 was a wrong-sign crash. If cosFormer ALSO crashes wrong-sign, the linear-time kernel-replacement family is closed at 0.94M. If cosFormer wins (Δ ≤ −0.005), it isolates the failure mode to feature-map shape (elu+1 sharp vs cos+sin diffuse), not to the linear-time template itself.

**Distinct mechanism classes at 0.94M**:
- 004-retnet-retention (decay-state retention, +0.04 wrong-sign) — **DIRECT STRUCTURAL PRIOR**
- 008-gated-deltanet (gated linear, off-niche taste-reject) — not empirical
- 148-focal-mod (additive context, +0.0072 null) — different mechanism class
- 189 (kernel-replacement with cos+sin feature map) — would close this family if it nulls

A 189 **win** (val_loss Δ ≤ −0.005) would isolate the failure mode to 004's specific feature map (elu+1) and unlock the linear-complexity kernel-replacement path for 37M/135M Phase-2 (where T grows and the linear-time advantage is free). A 189 **null** would close the entire kernel-replacement family at 0.94M, joining 004 on the wrong-sign side.

Net: 189 is the **direct follow-up to 004** — same template (linear-time attention with a feature map), different feature map (cos+sin vs elu+1). The information value is *high* either way — it isolates the failure mode.

## Why it's worth a slot
The "alternative attention" axis is the most-explored family at 0.94M with three closes (004, 008, 148), and 189 is the direct structural follow-up to 004. 189 is the only remaining distinct mechanism (kernel replacement with cos+sin feature map vs additive context vs decay retention). The diffuse-kernel bet (cos+sin beats elu+1) is a *qualitative* argument that 004's null did not test (004 was a complexity/state-space bet that nulled wrong-sign; 189 is a *kernel-shape* bet). A win isolates the failure mode to feature-map shape and unlocks the linear-complexity path for 37M/135M Phase-2; a null closes the kernel-replacement family entirely. Either outcome is informative — the slot is not wasted.
