---
id: 173-entmax-15
status: needs-plan
round: 1
updated: 2026-06-15T02:01:52Z
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
departs via the learned **α_h**, so the lever is bit-identical at init
and the departure is smooth.

## Why it's worth a slot (r2)

**Sharp mechanistic bet.** The binding constraint at 0.94M/12L/4H/3M
tokens is **gradient signal per token on per-head-attention-shape
axes**. There are 92 update steps, and 8 prior per-head-attention-shape
levers (152 logit-bias, 155 temp, 160 post-AV gain, 162 Q-only-norm,
165 K-only-norm, 166 T5-RPE) all closed in-band null at this tier
because **smooth-perturbation directions get absorbed by the existing
Q/K gradient updates** — the lever's gradient is small and
Q/K-norm-redundant. The two softmax-*replacement* siblings (148
focal-mod, 156 moa) also closed (148: replaces attention with
gated-additive context; 156: injects capacity via router). Entmax-1.5
is the only post-init lever in the family that is **not** a smooth
perturbation — replacing softmax with the α=1.5 simplex projection
introduces a **hard zero-mass regime** (bottom ~70% of K positions
receive p=0 ⇒ ∂L/∂V_i=0 exactly, not approximately small). At
α_h>1 the lever is **non-perturbative**: a single bit of α_h movement
crosses a discontinuity in ∂L/∂V for the zeroed-out rows. Either the
optimizer exploits this (WIN), or the discrete operator change
destabilizes the gradient floor that 8 prior siblings hid inside
(DRIFT). The bet is on the **discrete operator change**, not on a
smooth correction.

**Differentiation from the 8 prior nulls (the one the r1 reviewer
asked for).** Three families in the softmax-shape axis at 0.94M:

1. **Operator perturbation** (152, 155, 160, 162, 165, 166) — smooth,
   small-Lipschitz levers on the softmax output. Absorbed by Q/K
   gradient updates. The lever has a small axis and a small gradient.
2. **Operator replacement (non-attention)** (148) — focal modulation
   *replaces the attention block* with a gated-additive context.
   Not a softmax-shape lever; a different architecture.
3. **Capacity injection** (156, plus 117/118/146) — MoE/router/expert
   levers that *add* parameters to the attention or FFN path.
   Capacity-budget axis, not a softmax-shape axis.

Entmax-1.5 is **none of these three**. It is the only post-init lever
that is **(a) bit-identical to softmax at step 0** (α_h=1 ⇒
entmax-1.5=softmax in the continuous limit; the bisection collapses
to the standard softmax projection) **AND (b) a non-perturbative
operator change** as α_h moves. The lever is **isolated to the α_h
axis by construction** — there is no other parameter to absorb. The
"we are softmax at step 0" framing is the strongest differentiation,
and the discrete-sparsity framing makes the test **stronger** than
the 6 smooth siblings: a smooth lever's null is consistent with
"gradient was too weak to push the lever." A non-smooth lever's null
is consistent only with "the operator change is genuinely not
helpful at this tier."

**Honest Δ prior (committed).** With 8 nulls in the family, the
prior on a per-head-attention-shape lever at 0.94M is heavily
weighted toward the in-band null. The r1 Δ range [-0.005, -0.020]
had 60% of its mass inside the |Δ|<0.01 null band — that was
optimistic. The honest r2 prior: **70% in-band null (|Δ|<0.01),
20% mild WIN (Δ ∈ [-0.01, -0.03]), 10% DRIFT (Δ > +0.01) or
strong WIN (Δ < -0.05)**. The reviewer asked for a stronger
commitment; I commit to **Δ ≤ -0.015 OR clear DRIFT (Δ ≥ +0.05) as
the bar for "lever binds at this tier"** — anything inside the null
band is a clean close, not a WIN. This is a tight bar; the lever
either wins meaningfully or fails meaningfully.

**What a null teaches (the info-value case).** The 70% in-band null
is the most likely outcome and **closes the softmax-replacement axis
with a non-perturbative test**. The close line reads:
*"operator-replacement (entmax-1.5) closes alongside operator-
perturbation (152/155/160/162/165/166) and capacity-injection
(156/117/118/146) — no softmax-shape lever binds at
0.94M/12L/4H/92 update steps; re-evaluate the family at Phase-2
≥135M where per-token gradient signal is ~140× larger."* A null
from entmax-1.5 is a **stronger** null than the 8 prior siblings
because the lever is non-smooth and isolated to one axis — if even
a non-perturbative operator change can't bind at this tier, the
soft-perturbation nulls are confirmed by an independent test.
A WIN (Δ ≤ -0.015) unlocks the lever family for Phase-2 (the
operator-replacement axis). A DRIFT (Δ ≥ +0.05) confirms the
discrete operator change destabilizes the gradient and rules out
the family for re-evaluation at Phase-2 (saves a Phase-2 slot).

**Field-veto signal addressed.** 6+ years of softmax dominance in
production LMs is a real soft negative. The mechanism's target
metric is *gradient sparsity on the bottom K rows*, not perplexity
per se — perplexity is the metric where softmax wins. Entmax-1.5
needs a downstream task that *rewards* sparse attention
(long-context retrieval, structured attention) to show a clear
edge. We are not running that task. What we *are* doing is a clean
ablation of the operator at our tier; that ablation is informative
regardless of the result. The field veto says "entmax-1.5 doesn't
help perplexity at production scale" — and the close line will
read exactly that. The 6 years of softmax dominance don't argue
against running the ablation; they argue that the close is the
likely outcome, and that is a known outcome.

**Distinct from in-repo winners.** 025-SSMax (per-head
length-dependent temperature, WIN w/ caveat Δ=-0.091) is a
*scaling* lever on softmax (multiplies the temperature by
`1 + α_h·log(seq_pos)`). Entmax-1.5 is an *operator* replacement.
They are orthogonal: SSMax is a small smooth lever that binds
because the temperature axis has a strong per-head gradient
(L=12 gives 12 distinct seq-position patterns). Entmax-1.5 has
no per-head axis to exploit *except* the operator change.
148-focal-mod (NULL) *replaces* the attention block; entmax-1.5
*replaces the softmax inside* the attention block. Different
mechanism, different failure mode, different slot.

**A milder entmax-1.2 variant — deferred.** The r1 reviewer
suggested entmax-1.2 (closer to softmax, only mildly sparse) as a
"small dose" play. I defer this to a follow-up: if r2 entmax-1.5
nulls, the close line points at entmax-1.2 as the next test in
the family (with the same r2 bar). Splitting the r2 slot into
1.5+1.2 doubles the run budget and dilutes the signal; a clean
entmax-1.5 result is a better r2.

## Plan
- **Files to change**:
  - `models/layers.py`:
    - Add top-level `entmax_15` helper (after `softpick`): bisection on
      the Lagrange multiplier `λ` to project scores onto the α=1.5
      simplex. Closed form `p_i = max(0, 0.5·(s_i − λ))^2` with
      `p /= p.sum()`. ~25 LoC.
    - Add `use_entmax: bool = False` and `entmax_buckets: int = 32` to
      `MultiHeadAttention.__init__` kwargs. Allocate
      `self.entmax_alpha_raw = nn.Parameter(torch.zeros(n_heads))` when
      on (init 0 ⇒ `α_h = 1 + 0.5·(1 + tanh(0)) = 1` ⇒ entmax-1.5
      collapses to softmax at step 0).
    - Force the manual attention path when `use_entmax=True` (added to
      the elif at line 2881). Replace `torch.softmax(scores, dim=-1)`
      with `entmax_15(scores, dim=-1, alpha_per_head=…)` at both
      softmax call sites (FIRE branch line 2878 and manual branch line
      3035).
  - `configs/llm_config.py` — add `use_entmax: bool = False` and
    `entmax_buckets: int = 32` to `LLMConfig` (default off).
  - `models/llm.py` — pass `use_entmax=self.use_entmax` and
    `entmax_buckets=self.entmax_buckets` to both the YOCO upper-half
    block (~line 680) and the standard transformer block (~line 960).
- **Config flag**: `use_entmax: bool = False` (off by default).
- **Step-0 byte-identical**: `entmax_alpha_raw = 0` ⇒ `α_h = 1` for all
  heads ⇒ the bisection degenerates to the softmax projection (the
  quadratic form `max(0, s − λ)^2 / Σ` with `λ` chosen for
  `Σp=1` is equivalent to softmax in the `α=1` limit). With
  `bisection_tol=1e-7` the max-abs-diff vs `torch.softmax` is well below
  the 1e-5 fp32 noise floor used by the repo's step-0 identity checks.
  When `use_entmax=False` the helper is not called at all and the
  forward graph is bit-identical to baseline.
- **Run command** (tiny1m3m, seed 42):
  ```bash
  /venv/main/bin/python -m scripts.train \
      --config tiny1m3m \
      --variant trt \
      --seed 42 \
      --override 'use_entmax=True' --override 'entmax_buckets=32' \
      --run-tag 173-entmax-15-trt
  ```
- **Reading the val loss**: parsed from the JSONL training log
  (`runs/<tag>/log.jsonl`, last line's `val_loss` field) per the
  standard pipeline convention.
- **Out of scope**: per-layer β_l scalar (the second lever axis in the
  idea sketch) — keeps the LoC budget tight and the lever isolated.
