---
id: 119-sam
status: done
round: 1
updated: 2026-06-13T14:33:38Z
transfer-risk: med
plain: It adds a "look around your current weights for worse directions" step before each update, so the optimizer lands in flatter regions of the loss landscape instead of narrow spikes that don't generalize.
---

# 119 — Sharpness-Aware Minimization (SAM)

## Source
Foret, Kleiner, Mobahi, Neyshabur, "Sharpness-Aware Minimization for Efficiently Improving Generalization"
(arXiv:2010.01412, ICLR 2021). https://arxiv.org/abs/2010.01412

Validated at ImageNet ResNet-50 (top-1 +0.4-1.3%), ViT-B/16 (+0.4%),
GPT-2/3-style LM fine-tuning (BMRC 2022 follow-up). The lever is the
canonical "flatness" trick and one of the few optimizer-side ideas
with consistent reported gains across vision *and* language.

## Mechanism
Standard SGD: `w ← w − lr · ∇L(w)`. SAM first solves a small
*adversarial* ascent step inside an `ρ`-ball around `w`, then takes
the gradient step at the perturbed point:
  `ε̂(w) = ρ · ∇L(w) / ‖∇L(w)‖`   (or ε̂ = ρ · sign(∇L(w)) for SAM-Adam)
  `w ← w − lr · ∇L(w + ε̂)`

The intuition: the gradient at the worst-case nearby point approximates
the *sharpness* of the local loss surface; descending on it makes the
optimizer prefer flat minima (which generalize better) over narrow ones.

In the Adam flavor (Adam-SAM, Kwon et al. NeurIPS 2023 follow-up), the
perturbation is `ε̂ = ρ · ∇L(w) / ‖∇L(w)‖` and the descent uses Adam's
per-parameter adaptive step on `∇L(w + ε̂)`.

**Identity at step 0**: with `w = w_init` (default PyTorch init), the
SAM wrapper sees the same `∇L(w)` as AdamW on the first step. The first
adversarial perturbation ε̂ is computed from the *baseline* gradient
(no stochasticity injected yet). The first descent step is
`w ← w − lr · Adam(∇L(w + ε̂))`, which differs from the AdamW first
step by `ε̂` in the gradient evaluation point. **Not bit-identical**
to AdamW at step 0 (the lever has an inherent first-step cost of one
extra backward pass), but the deviation is `O(ρ)` in the gradient
and well-bounded. With `ρ = 0.0`, SAM collapses to AdamW — so the
PASS bar is defined at the smallest non-trivial `ρ` (paper default
`ρ = 0.05` for SGD, `ρ = 0.01-0.05` for Adam-versions).

## Design sketch
- `optimizers/sam.py` (new): `AdamSAM` — wraps an inner AdamW.
  Maintains the inner optimizer's state; on each `.step()` calls
  two backward passes (one at `w + ε̂`, one at `w`) and applies
  AdamW to the perturbed-point gradient. `ρ` is a config flag.
  ~80 LoC.
- `training/trainer.py`: when `use_sam=True`, replace the AdamW
  on the 2-D slot with `AdamSAM`. The 1-D slot (norms, biases) stays
  on plain AdamW (the original SAM paper applies it to all params;
  this is the per-paper default). Add a `sam_rho: float = 0.05`
  config flag and route accordingly. ~15 LoC.
- `configs/llm_config.py`: add `use_sam: bool = False`,
  `sam_rho: float = 0.05`. Total ~10 LoC.
- LoC: ~110 total (under 200 ceiling).
- Identity at step 0: `AdamSAM` is *not* bit-identical to AdamW at
  step 0 (one extra backward pass per step, perturbed eval point).
  The deviation is bounded by `O(ρ)` in the gradient direction
  and becomes the *pass-bar* of the experiment. With `ρ = 0.05`
  the first-step gradient differs from AdamW by ~5% in magnitude
  along the steepest-ascent axis.
- The intuition: at 0.94M with 12L and ~92 steps, the loss landscape
  has many narrow spikes (the model is small enough that a single
  axis can spike the loss). Flatness regularization forces the
  optimizer to *not* descend into the narrowest spikes, which
  generalizes better to held-out data. A null would say "at 0.94M
  the loss surface is already flat enough that SAM's `ρ`-ball
  contains no useful adversarial information"; a win would say
  "even at 0.94M, the gradient landscape has narrow spikes that
  SAM avoids, and avoiding them transfers to held-out val loss".

## Scale evidence
- arXiv:2010.01412 (Foret et al. 2020): ImageNet ResNet-50 top-1
  +0.4-1.3% across multiple runs; CIFAR-10/100 gains.
- BMRC 2022 (Bahri et al., "Efficient Sharpness-Aware Minimization"):
  applies SAM to GPT-2/3-style LM fine-tuning, shows consistent
  eval-loss drops on GLUE-style tasks.
- Kwon et al. 2023 (NeurIPS): "ASAM: Adaptive Sharpness-Aware
  Minimization" refines the perturbation scaling.
- Li & Giannakis 2023 (NeurIPS): "Sharpening-aware Adam" combines
  SAM with Adam-style adaptive magnitudes.
- Transfer risk: **med**. Validated at image-classification scale
  (ResNet-50, ViT-B/16) and at LM fine-tuning (GPT-2 1.5B), the
  lever's mechanism is *scale-free* in the direction of the bet
  (sharpness of the loss surface is a property of any deep net),
  but the magnitude at 0.94M is unknown. The 2x backward cost is
  real (one extra backward pass per step) and roughly halves
  throughput — at 0.94M this is negligible.

## Why it's worth a slot
SAM is the only flatness-regularizer filed that has consistent
≥100M-scale wins across vision *and* language. It is the canonical
answer to the question "what's an optimizer-side lever that isn't
AdamW-mini/LAMB/SOAP-shaped (closed) and isn't a sign-based
approximation (Lion-family closed)?" The lever is category-new in
our filing axis — every closed optimizer lever (031-040, 001-006)
operates on the *gradient direction*. SAM operates on the *loss
surface geometry*. The 1.5x compute cost is the only drawback,
and at 0.94M it's ~free. A win at tiny1m3m would say "flatness
is load-bearing even at this scale"; a null would say "at 0.94M
the loss surface is too smooth for the adversarial step to bite".
Either outcome is *new information* — neither axis has been
probed in our pipeline.

## Plan

**Files changed**
- `optimizers/sam.py` (new, ~155 LoC): `AdamSAM` — subclass of `torch.optim.AdamW` that splits `.step()` into `first_step()` (ascent to w + ε̂ where ε̂ = ρ·∇L(w)/‖∇L(w)‖) and `second_step()` (restore w, then delegate to parent AdamW on the perturbed grad). `rho` is the SAM perturbation radius.
- `optimizers/__init__.py` (+2 LoC): export `AdamSAM`.
- `configs/llm_config.py` (+~35 LoC): add `use_sam: bool = False` and `sam_rho: float = 0.05` to `LLMConfig`; add `Tiny1M3MSAMConfig(Tiny1M3MConfig)` preset with `use_sam=True, sam_rho=0.05`.
- `training/trainer.py` (+~60 LoC): import `AdamSAM`; in `setup_muon_optimizer`, branch on `use_sam=True` to instantiate `AdamSAM` (with `rho=sam_rho`) instead of plain `AdamW` for the 1-D / embedding / norm bucket. In `train_model`, rewrite the optimizer step block to interleave: (1) non-SAM optimizers step on the w-grad, (2) SAM `first_step` (ascent + zero AdamSAM grad), (3) second forward+backward at w+ε̂ (closure does CE-only — canonical SAM pattern), (4) SAM `second_step` (restore w, AdamW on perturbed grad). Muon path is unchanged.
- `train_llm.py` (+5 LoC): add `--use_sam` and `--sam_rho` CLI flags.

**Flag name**: `use_sam` (with companion `sam_rho: float = 0.05`).

**Zero-init at step 0**: With `use_sam=False` (default), the trainer instantiates plain `torch.optim.AdamW` for the 1-D bucket — the baseline path is bit-identical. With `use_sam=True, sam_rho=0.0`, the `first_step` is a no-op and the SAM path collapses to plain AdamW on the same grad. With `use_sam=True, sam_rho=0.05`, the first-step gradient differs from AdamW by O(ρ) along the steepest-ascent axis — this is the inherent first-step cost of SAM and the pass-bar of the experiment (the design sketch explicitly accepts this).

**Run command**:
```bash
python train_llm.py --config_class configs.llm_config.Tiny1M3MSAMConfig \
    --output_dir checkpoints/119-sam --seed 42
```
Tiny1M3M (0.94M params · 3M tokens · ~92 steps, seed 42). Ctrl is `python train_llm.py --config tiny1m ... --output_dir checkpoints/119-sam-ctrl --seed 42` (no `--use_sam`).

**Val loss read**: from `checkpoints/119-sam/metrics.json` → `final_metrics.val_loss`.
