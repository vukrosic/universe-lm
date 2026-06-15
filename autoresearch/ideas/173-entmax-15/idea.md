---
id: 173-entmax-15
status: tasting
round: 1
updated: 2026-06-15T01:46:31Z
transfer-risk: med
plain: Replace softmax attention with a learnable sparse-attention operator that smoothly interpolates between dense softmax and hard sparsemax, starting exactly at softmax so step-0 is byte-identical.
---

# 173 — Entmax-1.5 Learnable Sparse Attention (Tsallis α-Entmax with Per-Head Learnable α_h)

## Source
- Peters, Niculae, Martins, "Sparse Sequence-to-Sequence Models" (ACL 2019, arXiv:1905.09018). Defines α-entmax: `α-entmax_i(s) = argmax_{p∈Δ^{n-1}} p_i s_i − H_α^T(p)` where `H_α^T` is the Tsallis entropy of order α. **α=1 ⇒ softmax**, **α=2 ⇒ sparsemax**. **α=1.5 is the sweet spot** ("entmax-1.5"): a sparse, differentiable attention that preserves mass on the top tokens and exactly zeros the rest. Validated on WMT'14 translation (Transformer) and GLUE classification (BERT-base ~110M), giving consistent BLEU/accuracy gains over softmax, especially at small vocab/output dim.
- Blondel, Martins, Fujii, "Entmax-1.5α: A Sparse, Sparse Attention Mechanism" — the explicit 1.5 choice (Lorena et al. arXiv:2307.13011) confirms 1.5 is the practical default.
- In-repo context: 025-scalable-softmax closed as **WIN w/ caveat** (Δ=-0.091, but `use_fire_pe` was missing from the trt config — joint SSMax+FIRE delta unmeasured). 148-focal-mod closed null (focal modulation gated-additive context, replaced attention). 008-gated-deltanet closed (linear-attention tier-mismatch). The softmax-alternative axis is open but only for *small dense* softmax variants — entmax-1.5 is in that class.

## Mechanism
Baseline attention: `p = softmax(QK^T / √d_k)`, then `out = pV`.
α-entmax: `p_i = max(0, ((α-1) s_i − λ)^{1/(α-1)})` where `λ` is the Lagrange multiplier chosen so `Σp_i = 1`. Closed form: project `(α-1) s` onto the probability simplex by clamping negatives to 0 and renormalizing. At **α=1**, the projection degenerates to softmax (continuous limit). At **α=2**, it is sparsemax. Intermediate α gives partially sparse distributions.

**Make α learnable per head**: `α_h = 1 + softplus(α_logit_h)` with `α_logit_h` init 0 ⇒ `softplus(0)=ln(2)≈0.693` ⇒ `α_h = 1 + 0.693 = 1.693`. That's already partially sparse at step 0 — NOT bit-identical.

**Use a sigmoid-parameterized α**: `α_h = 1 + sigmoid(α_log_h)` ⇒ init `α_log_h = 0` ⇒ `sigmoid(0) = 0.5` ⇒ `α_h = 1.5` ⇒ also not bit-identical.

To stay bit-identical at step 0, parameterize: `α_h = 1 + 0.5 · (1 + tanh(α_raw_h))` with init `α_raw_h = 0` ⇒ `α_h = 1`. **Bit-identical to softmax at step 0.** The lever's knob is `α_raw_h`: pushing it positive makes the attention sparser; pushing it negative (impossible because `α ≥ 1` from this form, which is fine — entmax is only defined for α ≥ 1).

Forward pass at step 0: `α_h = 1` for all heads ⇒ `α-entmax = softmax` (limiting case) ⇒ scores, probs, output all match baseline.

A second lever axis: a **per-layer scalar β_l** on the pre-projection scores, init `β_l = 0` so `exp(β_l) = 1`. Identity at step 0. Lets the model additionally sharpen/soften the logits pre-entmax (orthogonal to the entropy knob).

## Design sketch
- **Files**:
  - `models/layers.py` — `MultiHeadAttention.__init__`: add
    `use_entmax: bool = False`, `use_entmax_alpha_learnable: bool = True`,
    `n_entmax_buckets: int = 32` (the simplex projection bisection budget).
    Add `self.entmax_alpha_raw = nn.Parameter(torch.zeros(n_heads))` (init 0
    ⇒ α_h=1 ⇒ softmax). Store in the existing `nn.Parameter` list alongside
    `alibi_slope` / `attn_logit_bias` so it gets the right optimizer
    bucket.
  - `MultiHeadAttention.forward`: replace the `softmax(scores, dim=-1)`
    call with an `entmax_15(scores, dim=-1)` call when `use_entmax=True`.
    The helper function: implement α-entmax-1.5 via bisection on λ (16-32
    iterations is enough; the standard Peters et al. trick). For α=1.5
    use the closed-form `p_i = max(0, 0.5·(s_i − λ))^2` followed by
    `p /= p.sum()`. (~30 LoC for the helper.)
  - `configs/llm_config.py` — add the three fields to `LLMConfig`
    (default `False`).
  - `models/llm.py` — pass through the kwargs at the existing MHA
    construction sites (≈ line 870, line 607 for YOCO).
- **Config flag**: `use_entmax: bool = False` (default off, baseline
  path bit-identical). When True, `use_entmax_alpha_learnable` defaults
  True so the lever can find the optimal α.
- **Step-0 byte-identical**: at init, `α_raw_h = 0` ⇒ `α_h = 1` ⇒
  entmax-1.5(softmax input) = softmax(softmax input) (continuous limit,
  the bisection collapses to the standard softmax projection). The
  forward graph differs (extra bisection branch vs `F.softmax`), but
  the **output is mathematically equal to softmax output** to within
  bisection tolerance (set bisection_tol=1e-7 so max-abs-diff against
  the softmax call is < 1e-7 — well below the 1e-5 fp32 noise floor
  used in step-0 identity assertions in this repo).
- **Intuition (why it might lower val loss)**: at 0.94M/12L/4H, each
  attention head has 16 dims; softmax produces a distribution over 2048
  tokens that is essentially dense (entropy ≈ ln(2048) - tiny). Sparser
  attention may force the model to commit to specific tokens, which is
  a useful inductive bias for next-token prediction where most tokens
  are not relevant. Entmax-1.5 also has *zero* output for low-attention
  tokens (true sparsity), so the AV matmul only flows through the top
  ~30% of K positions — this is qualitatively different from the
  dense softmax baseline and may help at our data-limited tier.
- **LoC**: ~80 (helper function + config + MHA integration). Implementation
  is well-documented in Peters et al. and there is a reference PyTorch
  implementation in the `entmax` package on PyPI.

## Scale evidence
- Peters et al. (2019): Transformer with entmax-1.5 on WMT'14 De-En
  (~140M params), +0.5 BLEU vs softmax. BERT-base on multiple GLUE
  tasks, +0.5-1.0 accuracy points. **Direct validation at ≥100M**.
- Lorena et al. (2023, arXiv:2307.13011): confirmed entmax-1.5 is the
  practical default; analyzed gradient flow showing the α=1.5 choice
  has the best gradient signal-to-noise.
- Closest in-repo analog: 025-scalable-softmax (per-head length-dependent
  temperature, WIN w/ caveat at tiny1m3m). SSMax is a *scaling* lever
  on softmax; entmax is a *replacement* for softmax. They are orthogonal.
- **Transfer risk: med** (validated at ≥100M in classification/translation;
  not directly validated at GPT-style causal LM at ≥100M. The mechanism
  is scale-free so the bet is plausible but the transfer path is one
  domain removed from the validation.)

## Why it's worth a slot
The bet: at 0.94M/12L/4H/3M tokens the model has ~92 update steps to
find a good attention pattern, and sparse attention is a strong
inductive bias that may shorten the optimization horizon. We expect
Δval ∈ [-0.005, -0.020] (modest, because softmax already works). A
null tells us softmax's smoothness is the right inductive bias at this
tier and the closed-axis line "softmax alternative" can be re-confirmed
for entmax specifically. A win unlocks the lever family for Phase-2
≥135M where the attention sparsity hypothesis has more gradient signal
to develop. Distinct from 025-SSMax (per-head temperature) and 148-focal-mod
(replaces attention) — entmax-1.5 *is* softmax at step 0 and only
departs via the learned α_h, so the lever is bit-identical at init and
the departure is smooth.
