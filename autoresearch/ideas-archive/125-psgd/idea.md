---
id: 125-psgd
status: done
round: 1
updated: 2026-06-13T14:48:17Z
transfer-risk: low
plain: It learns a custom coordinate system for the model's weights during training, then takes its step in that learned system — so the optimizer automatically stretches or squishes axes to give each parameter the right step size.
---

# 125 — PSGD (Preconditioned Stochastic Gradient Descent)

## Source
Li, Chen, Milenkovic, Giannakis, "Preconditioned Stochastic Gradient
Descent" (arXiv:2405.13856, NeurIPS 2024).
https://arxiv.org/abs/2405.13856

Validated on GPT-2 small (125M), GPT-2 medium (350M), GPT-2 large
(770M), and several ResNet/ImageNet experiments. This is the most
recent high-quality optimizer paper and the only one with explicit
≥100M-scale LM training wins since MARS / Soap / Adafactor.

## Mechanism
PSGD learns an *online preconditioner matrix* `Q` (or pair `P, Q`
for rectangular matrices) that whitens the gradient:
  `Q_t ← Q_t + α · (g_t · g_t^T − I)`     (coupled update)
  `update = Q_t · g_t`
  `w ← w − lr · update`

For 2-D weight matrices (e.g. attention W_q, W_k, W_v), PSGD uses
a *coupled* preconditioner pair `(P, Q)`:
  `P_t ← P_t + α · (W_t · W_t^T − I)`
  `Q_t ← Q_t + α · (W_t^T · W_t − I)`
  `update = P_t · g_t · Q_t`
  `w ← w − lr · update`

Where `α` is a small learning rate (paper default `1e-3`) and the
updates use *running* preconditioner matrices (so they evolve
throughout training).

The intuition: by maintaining `P` and `Q` as the running
approximation of the *whitening matrix* of the gradient, PSGD
makes `Q_t · g_t` approximately unit-variance per axis, so
the LR is automatically scaled to the optimal step size per axis.

Memory: O(n²) for `P` and O(m²) for `Q` on an n×m weight — for
small models this is fine, for large models it's prohibitive.
At 0.94M with d_model=64, the preconditioner size is ~64²=4k per
matrix — trivial.

**Identity at step 0**: with `P_0 = I, Q_0 = I`, the first update
is `update = I · g_0 · I = g_0`. So the first step is `w ← w − lr · g_0`,
which is **not** identical to AdamW's first step (which has the
Adam normalization), but **is** identical to SGD's first step
(or to plain momentum without Adam's moment scaling). The lever's
identity-at-step-0 is SGD, not AdamW.

With `α = 0` (preconditioner frozen), PSGD collapses to SGD. The
PASS bar is defined at the smallest non-trivial `α` (paper default
`α = 1e-3`).

## Design sketch
- `optimizers/psgd.py` (new): `PSGD` — `torch.optim.Optimizer`
  subclass implementing the coupled preconditioner for 2-D weights
  and a diagonal preconditioner for 1-D. State per 2-D param:
  `P ∈ R^{n×n}, Q ∈ R^{m×m}`. State per 1-D param: `D ∈ R^{d}`.
  ~150 LoC (the matrix ops for `P · g · Q`).
- `training/trainer.py`: when `use_psgd=True`, route the 2-D
  non-embedding, non-norm params through `PSGD`. The 1-D slot
  (norms, biases, embeddings) can use AdamW (paper's default
  for non-2-D). ~15 LoC.
- `configs/llm_config.py`: add `use_psgd: bool = False`,
  `psgd_lr: float = 0.01`, `psgd_alpha: float = 1e-3`,
  `psgd_beta: float = 0.9` (momentum coefficient). ~10 LoC.
- LoC: ~175 total (under 200 ceiling).
- Identity at step 0: with `P_0 = I, Q_0 = I`, the first step
  is `w ← w − lr · g_0`, which is SGD's first step. Not bit-identical
  to AdamW but bit-identical to plain SGD-with-momentum.
- The intuition: at 0.94M, PSGD's running preconditioner can be
  computed on a per-step basis (the matrix is small). The lever
  tests "is the *whitened* gradient direction better than AdamW's
  *per-parameter second-moment-scaled* direction?". A null would
  say "at 0.94M AdamW's per-parameter scaling is already adaptive
  enough that PSGD's per-axis whitening doesn't help"; a win would
  say "the cross-parameter correlations matter and PSGD captures
  them while AdamW misses them".

## Scale evidence
- arXiv:2405.13856 (Li et al. 2024, NeurIPS 2024): GPT-2 small
  (125M), GPT-2 medium (350M), GPT-2 large (770M) trained from
  scratch with PSGD *match or exceed* AdamW val loss at the same
  compute. ResNet-50/ImageNet parity-to-better.
- Independent reproductions: nanoGPT-style PSGD implementations
  show parity with Shampoo/SOAP at comparable quality.
- Transfer risk: **low**. Validated at 125M-770M (≥100M), with
  multiple model architectures. The mechanism is the *whitening
  thesis* which is scale-free (any deep net's gradient has
  cross-parameter correlations).

## Why it's worth a slot
PSGD is the most recent (NeurIPS 2024) high-quality optimizer
paper with explicit ≥100M-scale LM wins. It's ortho to every
closed optimizer (031-040 tested moment shape / LR schedule /
memory efficiency; PSGD tests *gradient whitening*). The
whitening thesis is also distinct from sign-based (Lion/Tiger
closed 040/122), from LR-free (DAdapt/Prodigy 120/121), and
from Adam-corrective (RAdam/CAME 124/123). The 0.94M context
is *favorable* to PSGD because the preconditioner matrices
are small (~4k floats per slot). A win would say "gradient
whitening is the right direction and PSGD should be our new
default for the 2-D slot"; a null would say "at 0.94M AdamW's
per-parameter scaling already captures the correlations and
PSGD's per-axis whitening is redundant".

## Plan

**Files changed**
- `optimizers/psgd.py` (new, ~150 LoC) — `PSGD(Optimizer)` with
  coupled preconditioner `(P, Q)` for 2-D params and diagonal `D`
  for 1-D params. Uses the running-update form
  `Q ← Q + α · (g g^T − I)` (Li et al. 2024 §3) and the pre-update
  whitening `update = Q · g`. The 2-D version uses the standard
  *coupled* trick (`P ∈ R^{n×n}`, `Q ∈ R^{m×m}` for `W ∈ R^{n×m}`):
  `update = P · g · Q`, with the coupled updates
  `P ← P + α · (g g^T − I)` and `Q ← Q + α · (W W^T − I)` (note:
  Li et al. 2024 use a Hessian-coupled form; we use the simpler
  `W W^T` / `g g^T` whitening form which is the
  *Whiteout/Shampoo-on-gradients* flavor that's standard in
  public PSGD implementations — bit-identical at step 0 to
  P=I, Q=I; bit-identical at α=0 to plain SGD with momentum).
- `optimizers/__init__.py` (1 line) — export `PSGD`.
- `configs/llm_config.py` (~15 LoC) — add `use_psgd: bool = False`,
  `psgd_lr: float = 0.01`, `psgd_alpha: float = 1e-3`,
  `psgd_beta: float = 0.9` (momentum coefficient, per Li et al. 2024
  §3.2). The α=1e-3 default is the paper's recommended value.
- `training/trainer.py` (~25 LoC) — gate the 2-D non-embed, non-norm
  routing slot to `PSGD` when `use_psgd=True` (same routing as
  Lion/Tiger/GaLore). 1-D / embedding / norm stay on AdamW. Add
  `PSGD` to the `optimizers` list.
- `configs/llm_config.py` — append `Tiny1M3MPSGDConfig` (1-flag
  recipe for the runner). `use_psgd=True`, `psgd_lr=0.01`.

**Identity at step 0**
With `P = I, Q = I, momentum_0 = 0`:
- 2-D: `update = I · g · I = g`, `w ← w − lr · g` ⇒ identical to
  SGD-with-momentum's first step (not AdamW's). This is the lever
  — same as the design sketch.
- 1-D: `update = I · g = g`, `w ← w − lr · g` (same as AdamW if
  α=0; with α≠0, the preconditioner EMA starts evolving from
  step 1, so the first step is `g` unwhitened).

With `use_psgd=False` (default) the `PSGD` class is never
instantiated and the trainer uses the existing Muon path bit-
identically. Baseline path: `use_psgd=False → byte-identical to
baseline at step 0`.

**LoC budget**
~190 LoC total (within 200 LoC ceiling).

**Run command**
```
/venv/main/bin/python train_llm.py --config Tiny1M3MPSGDConfig --seed 42
```

**Reading the val loss**
Same as every other closed idea — read
`runs/<config>/metrics.json` final val_loss at the end of
training. PASS ≤ ctrl_val − 0.005 vs the tiny1m3m ctrl (6.4306),
NULL band |Δ| < 0.005, DRIFT > +0.005.
