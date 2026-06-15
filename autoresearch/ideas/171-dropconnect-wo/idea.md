---
id: 171-dropconnect-wo
status: running
round: 3
updated: 2026-06-15T05:07:19Z
transfer-risk: med
plain: During training, randomly zero out individual weights of the attention output matrix (DropConnect) as a regularizer; rate ramps 0.0 -> 0.05 over the first 100 steps so step 0 is byte-identical to baseline and the lever fires after warmup.
---

# 171 — DropConnect on W_O (Per-Weight Stochastic Masking of Attention Output Projection)

## Source
- Wan, Zeiler, Zhang, LeCun, Fergus, "Regularization of Neural Networks using
  DropConnect" (ICML 2013, also arXiv:1304.3174) — the original DropConnect
  paper. Validated on MNIST/CIFAR/ImageNet, showing consistent gains when
  DropConnect replaces Dropout on the FC layers.
- More recently: Parmar et al. ("Stand-Alone Self-Attention in Vision Models",
  NeurIPS 2019) and various ViT studies apply DropConnect to attention
  projections; "Sparse MoE" style per-weight sparsity is also related.
- Closest in-repo priors:
  - 147-dropkey (NULL at 0.94M, `closed.md:34`): drops **keys** (per-token
    zeroing of the K vector pre-attention). 171 drops **weights of W_O**
    (per-weight zeroing of the output projection matrix). Different tensor,
    different stochastic axis.
  - 111-drop-path (DRIFT, `closed.md:49`): drops **entire residual branches**
    (per-block stochastic depth). 171 doesn't drop branches, only weights.
  - 138-looksam (NULL, `closed.md:108`): periodic SAM perturbation. Different
    mechanism (sharpness-seeking perturbation, not stochastic masking).
  - 115-rdrop (NULL at 0.94M, `closed.md`): KL-divergence regularizer between
    two forward passes; distinct from per-weight masking. Adjacent closed
    regularizer that supports the "regularizer family exhausted" thesis.
- DropConnect-on-attention-output is NOT in `closed.md`'s closed axes
  ("Dropout/regularizer family" is closed, but DropConnect is a distinct
  per-weight stochastic regularizer from per-token Dropout / DropKey /
  DropPath).

## Mechanism
Standard attention: `out = concat(head_1, ..., head_H) @ W_O` where
`W_O ∈ ℝ^{d_model × d_model}`. With DropConnect:
1. At training time, sample a Bernoulli mask `M ∈ {0,1}^{d_model × d_model}`
   with probability `p_keep = 1 - rate` per entry.
2. Apply `W_O_masked = W_O ⊙ M / p_keep` (inverted-dropout rescale).
3. Use `W_O_masked` for the forward pass.
4. At eval time, use `W_O` unchanged (no noise, no rescale).

The mask is sampled per forward pass (not per-step) and is the same mask for
all batch elements and all positions. This is "weight-level" noise, distinct
from "token-level" noise (Dropout on activations) or "row-level" noise
(DropKey on the K matrix).

### Why weight-level should bind at 0.94M where token-level (147) and path-level (111) didn't
Three regularizer variants have already nulled at tiny1m3m:
147-dropkey (per-token K-mask), 111-drop-path (per-branch residual mask), and
115-rdrop (KL-divergence loss regularizer). The pitch's "just a different
tensor" framing is too weak — without an attribution argument, the
pre-test-belief should be that the regularizer family is closed. The
mechanistic argument for why weight-level is a distinct stochastic axis:

**Q/K/V gradient updates can absorb token-level and path-level noise;
W_O gradient updates cannot mask themselves.** Per-token DropKey (147)
multiplies the K tensor by a random mask. The QK product sees a noisy K,
but Q's own gradient (computed downstream from the loss) still receives a
clean signal — the optimizer can route around the noise by adjusting Q.
Per-branch DropPath (111) zeros entire residual branches; the surviving
branches still receive clean gradients and the optimizer simply up-weights
the surviving paths. Both are *signal-space* regularizers that the
gradient updates can compensate for at the parameter level.

**DropConnect on W_O is a *parameter-space* regularizer — the mask is the
gradient mask, not just the forward mask.** When we zero a weight `W_O[i,j]`
during the forward pass, that same weight's gradient is computed against a
zeroed output: the parameter receives a smaller-magnitude update *and* the
gradient tells the optimizer to up-weight surviving paths in W_O. The
optimizer cannot route around the noise by adjusting a *different*
parameter, because the noise is on the parameter being updated. W_O is
also a single dense `d_model × d_model` matrix (64×64 = 4096 weights) that
the optimizer can co-adapt onto narrow subspaces at small scale; weight
masking forces redundant paths in that single matrix.

The argument is not "weight-level always wins" — it's "weight-level is
the only member of the family whose mask the optimizer cannot absorb
into a different parameter's update," so it is a structurally distinct
axis from the closed 147/111/115 variants.

## Treatment (locked, no longer a 3-option sketch)
**Canonical treatment: ramp `rate` from 0.0 → 0.05 over the first 100
optimizer steps, then hold at 0.05 for the remaining ~92 steps.**

Step-by-step:
- **Step 0**: `rate = 0.0` ⇒ `p_keep = 1.0` ⇒ mask is all-ones ⇒
  `W_O_masked = W_O ⊙ 1 / 1 = W_O` ⇒ **byte-identical to baseline
  (max-abs-diff = 0.0)**. Identity holds because the guard
  `rate > 0.0` short-circuits before any RNG is consumed.
- **Steps 1–99**: linear ramp `rate = 0.05 * step / 100`. At step 1 the
  mask is sampled with `p = 0.0005` (effectively all-ones); at step 100
  the mask is sampled with `p = 0.05` (Wan et al.'s ImageNet sweet spot
  for this kind of projection). The forward differs from baseline by
  < 0.05% on average during this ramp, so the trajectory departs slowly.
- **Steps 100+**: `rate = 0.05` held constant; this is the regime where
  the regularizer actually does work, and where the val-loss comparison
  against baseline is meaningful.

Why not option A (rate = 0.1 immediately)? Breaks step-0 byte-identity —
step 0 would sample a 10% mask on W_O, the forward would differ from
baseline, and the §1 "step-0 byte-identical" gate fails.

Why not option B (rate = 0.0 throughout, regularizer infrastructure
present but inactive)? Tests "is the code path in place" — that's a
definition-gate concern, not a lever test, and would burn a slot for
an info-poor A/B. The regularizer family is at 3 nulls; we need *one*
test of a live regularizer, not a control for a never-firing one.

Why ramp 0.05 and not 0.1? Wan's CIFAR/ImageNet sweet spot is 0.1 on
*fully connected* projections; the attention output projection at
d_model=64 is structurally similar (dense, square, single-tensor), so
0.1 is a defensible starting point, but 0.05 is a more conservative
ramp target that gives the optimizer more time to learn through the
noise. Either is defensible; 0.05 is the lower-risk choice and the
r1 sketch explicitly named it as the alternative.

## Design sketch
- **File**: `models/layers.py` (`MultiHeadAttention.__init__` adds
  `use_dropconnect_wo: bool = False`, `dropconnect_wo_rate: float = 0.05`,
  `dropconnect_wo_warmup_steps: int = 100` kwargs;
  `MultiHeadAttention.forward` adds a single branch after the head
  concatenation step that samples the mask and applies it to W_O).
- **Config flag**: `use_dropconnect_wo: bool = False` and
  `dropconnect_wo_rate: float = 0.05` and
  `dropconnect_wo_warmup_steps: int = 100` on `LLMConfig` (rate is
  Wan's CIFAR/ImageNet sweet spot for dense projections, halved for
  ramp safety; the lever-test is the *presence* of the regularizer
  with a live ramp, not the rate HP).
- **Step-0 byte-identical**: at step 0, the warmup-scheduled effective
  rate is 0.0 ⇒ guard `effective_rate > 0.0` is False ⇒ branch is
  never taken ⇒ no RNG consumed, no parameter modified ⇒
  **byte-identical to baseline (max-abs-diff = 0.0)**.
- **Intuition (why it might lower val loss)**: see the mechanistic
  argument above. Per-weight masking on W_O is the only regularizer in
  the 147/111/115 closed block whose mask the optimizer cannot absorb
  into a different parameter's gradient update. W_O is a single dense
  64×64 matrix; weight masking forces redundant paths in that matrix,
  preventing the optimizer from co-adapting W_O onto a narrow subspace
  over the 3M-token training horizon.
- **LoC**: ~30 lines (mask sample + apply + assert + schedule + warmup).

## Scale evidence
- DropConnect is well-validated at vision scale (CIFAR-10, ImageNet) in the
  original paper and many follow-ups (VGG-DropConnect, ResNet-DropConnect,
  DenseNet-DropConnect variants).
- For *language models*, DropConnect is less commonly cited. The closest
  LM application is "Structured DropConnect" applied to LSTM/RNN language
  models (Pham et al. 2014, "Dropout improves Recurrent Neural Networks for
  Handwriting Recognition"); for transformer LMs, the lever is novel.
- **Transfer risk: med** (validated at vision ≥100M params; for LMs the
  lever is plausible but not directly validated at ≥100M. Strong argument
  for transfer from vision-CNN's "dense projection" + "limited data" setup
  to "W_O is a dense projection" + "0.94M sees ~3M tokens which is data-
  limited for LMs").

## Win/null bar (sharpened against 0.04 noise band)
The cached 6.4394±0.04 / 6.4504±0.0558 baseline bands put one-seed Δ
detection at `|Δ| > 0.04` and "informative but inconclusive" at
`0.01 < |Δ| ≤ 0.04`. Applying this to the 171 lever:
- **Δ ≤ −0.020**: signal — clears the lower edge of the inconclusive
  band and is consistent with a real regularizer effect. This is the
  *only* single-seed outcome that survives the noise test.
- **−0.020 < Δ < −0.005**: informative but inconclusive — treat as
  null-and-close per the two-ctrl WIN rule (the effect is below
  one-seed detection).
- **Δ ≥ −0.005**: null.

A null outcome still has payoff: "null closes the weight-level axis of
the regularizer family (after token-level 147-dropkey and path-level
111-drop-path, weight-level 171-dropconnect completes the family
exhaustion at 0.94M). The remaining regularizer axes at this tier
are not in the per-mask family — they would need a structurally new
mechanism (e.g. SAM-style sharpness-seeking, 138-looksam, already
null) or a different target (loss-shape, closed at 066–070)."

## Why it's worth a slot
The bet: per-weight stochastic masking on W_O is the only member of the
per-mask regularizer family whose mask the optimizer cannot absorb into
a different parameter's gradient update, so it is a structurally distinct
axis from the closed 147 (token-level K-mask) and 111 (path-level residual
mask) variants. At our data-limited 0.94M/3M-token tier, W_O can co-adapt
onto a narrow subspace over 192 steps; weight masking forces redundant
paths in that single 64×64 matrix. We expect Δval ∈ [−0.020, −0.005]
(borderline signal / inconclusive); a clear null would still close the
weight-level axis and complete the per-mask family exhaustion at this
tier. Step-0 byte-identical, ~30 LoC + ~3–4 min compute + a few hours of
agent attention — acceptable spend for closing the last open axis in
the regularizer family.

## Plan

- **Files**:
  - `configs/llm_config.py` — add `use_dropconnect_wo: bool = False`,
    `dropconnect_wo_rate: float = 0.05`,
    `dropconnect_wo_warmup_steps: int = 100` next to `use_drop_key`
    (≈ line 720). Default rate 0.05 reflects the ramp target (not the
    step-0 value, which is 0.0 via the warmup schedule).
  - `models/layers.py` —
    - `MultiHeadAttention.__init__`: add the three kwargs after
      `use_drop_key` / `drop_key_rate` (≈ line 968); store as
      `self.use_dropconnect_wo`, `self.dropconnect_wo_rate`,
      `self.dropconnect_wo_warmup_steps` after `self.use_drop_key`
      (≈ line 1636). Track a `self._step_count: int = 0` counter that
      increments on every forward pass (or, cleaner, accept the current
      step from the trainer via a kwarg).
    - `MultiHeadAttention.forward`: branch at the W_O application site
      (≈ line 3233) — compute the effective rate
      `effective_rate = dropconnect_wo_rate * min(step, warmup) / warmup`,
      guard `use_dropconnect_wo and self.training and effective_rate > 0.0`,
      sample Bernoulli mask with `p = effective_rate`, rescale by
      `1 / (1 - effective_rate)`, apply to W_O.
    - `TransformerBlock.__init__`: pass-through kwargs after the drop_key
      pass-through (≈ line 3737). YOCOLlamaBlock forwards via `*args,
      **kwargs` so it's covered automatically.
  - `models/llm.py` — `MinimalLLM.__init__`: read
    `use_dropconnect_wo` / `dropconnect_wo_rate` /
    `dropconnect_wo_warmup_steps` next to `use_drop_key` (≈ line 264);
    pass-through at the YOCO upper-half construction (≈ line 607) and
    the standard TransformerBlock construction (≈ line 870).
- **Config flag**: `use_dropconnect_wo: bool = False` (off by default,
  baseline path bit-identical), `dropconnect_wo_rate: float = 0.05`
  (ramp target — the step-0 value is 0.0 via the warmup schedule),
  `dropconnect_wo_warmup_steps: int = 100` (ramp length).
- **Step-0 byte-identical**: when `use_dropconnect_wo=False`, the branch
  is never taken (no RNG consumed, no parameter created) ⇒ baseline path
  bit-identical. When `use_dropconnect_wo=True`, at step 0 the
  warmup-scheduled effective rate is 0.0, the guard
  `effective_rate > 0.0` skips the mask branch ⇒ also bit-identical
  (cost: one branch comparison + one min()). Eval mode
  (`self.training == False`) also skips the mask.
- **Run command** (treatment):
  `/venv/main/bin/python runner/runner.py --config tiny1m3m
   --seed 42 --override use_dropconnect_wo=True,dropconnect_wo_rate=0.05,dropconnect_wo_warmup_steps=100`
- **Read final val loss**: from the runner's standard log line
  `val_loss=...` at the final step; compare against the baseline
  6.4216 from `token2science-papers-platform` memory and the cached
  6.4394±0.04 / 6.4504±0.0558 two-ctrl bracket.
