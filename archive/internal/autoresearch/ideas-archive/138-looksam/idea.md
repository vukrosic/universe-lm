---
id: 138-looksam
status: done
round: 1
updated: 2026-06-13T20:53:38Z
transfer-risk: med
plain: Like SAM, but only every few steps — so the optimizer spends most of its time on cheap normal AdamW steps and only occasionally does the expensive "look for a worse direction" step.
---

# 138 — LookSAM: Periodic SAM (Efficient Sharpness-Aware Minimization)

## Source
Du, Yan, Feng, Zhu, Yang, Sui, "Efficient Sharpness-Aware Minimization
for Improved Training of Neural Networks" (arXiv:2205.13539, ICLR
2023). https://arxiv.org/abs/2205.13539

Validated on CIFAR-10/100 ResNet, ImageNet ResNet-50 (25M),
ImageNet ViT-Small (~22M), and several self-supervised
benchmarks. The lever is the *periodic* variant of SAM (119)
— do the expensive 2x backward pass only every K steps instead
of every step.

## Mechanism
Standard SAM (119): every step does 2 backward passes (one at
`w + ε̂`, one at `w`) for `O(2x)` compute. LookSAM does the
SAM-style 2-backward step only every K steps:
  `if step mod K == 0:`
    `ε̂ = ρ · ∇L(w) / ‖∇L(w)‖`
    `w ← w − lr · ∇L(w + ε̂)`     (SAM step)
  `else:`
    `w ← w − lr · ∇L(w)`     (standard step)

Paper default `K = 5` (SAM every 5 steps), giving ~1.2x compute
(slightly more than standard AdamW but much less than full SAM's
2x).

The intuition: SAM's flat-minima benefit comes from *occasional*
sharpness-aware steps, not from every step being sharpness-aware.
The standard steps between SAM steps benefit from the SAM-induced
flat region without paying the 2x cost.

**Identity at step 0**: with `K = 5`, the first step is a
standard AdamW step (not SAM). Step 5 is the first SAM step
(which does the 2x backward). So LookSAM is **bit-identical**
to AdamW at steps 0..4, and SAM-shaped at step 5+.

The lever's identity-at-step-0 is the *standard* AdamW path
(only the *periodic* SAM steps deviate). This is *more*
bit-identical at step 0 than full SAM (119), which always
has the SAM-style first step.

## Design sketch
- `optimizers/looksam.py` (new): `LookSAM` — wrapper that
  periodically (every K steps) invokes the SAM-style 2-backward
  step. Otherwise applies standard AdamW. ~50 LoC.
- `training/trainer.py`: when `use_looksam=True`, replace
  AdamW with `LookSAM`. The 2-D slot still uses Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_looksam: bool = False`,
  `looksam_k: int = 5`, `looksam_rho: float = 0.05`. ~10 LoC.
- LoC: ~70 total (under 200 ceiling).
- Identity at step 0: bit-identical to AdamW at steps 0..4.
  At step 5+, the SAM-style 2-backward activates.
- The intuition: at 0.94M with 92 steps, full SAM (119) would
  pay 2x compute for ~180 backward passes (2 per step). LookSAM
  pays 2x compute for ~36 backward passes (every K=5 steps).
  The compute savings (~5x) make LookSAM more practical for
  our 92-step window. A null would say "the periodic SAM is
  too sparse to give flatness benefits at 0.94M"; a win would
  say "even every-5-step SAM is enough to find flat regions
  and the win is comparable to full SAM".

## Scale evidence
- arXiv:2205.13539 (Du et al. 2022, ICLR 2023): CIFAR-10/100
  ResNet, ImageNet ResNet-50 (25M), ImageNet ViT-Small (~22M).
  Reports ~95% of full SAM's gains at ~50% of the compute cost.
- Transfer risk: **med**. Validated at image-classification
  scale (ResNet-50, ViT-Small ≤100M), the mechanism is
  scale-free. At 0.94M the compute savings are more impactful
  than the gains (we have a fixed 92-step budget).

## Why it's worth a slot
LookSAM is the *compute-efficient* variant of SAM (119). Both
test the same flatness thesis, but LookSAM makes the test
practical at tiny1m3m by reducing the compute overhead from
2x to 1.2x. Filing both gives us a clean A/B: full SAM vs
periodic SAM at tiny1m3m. If full SAM (119) wins, LookSAM is
the next-tier optimization. If only LookSAM wins, the periodic
form is the sweet spot. If both null, the flatness thesis is
fully closed at 0.94M. The lever is ortho to every closed
optimizer (031-040, 001-006) — LookSAM is a SAM variant, not
an AdamW variant.

## Plan

**Files touched**
- `optimizers/looksam.py` (new, ~50 LoC): `LookSAM` wrapper that
  owns an inner `AdamSAM` and a step counter. On non-SAM steps
  it calls `inner.step()` directly (plain AdamW path); every K
  steps it runs the SAM ascent → closure → descent dance via
  the inner optimizer's `first_step` / `second_step`. The
  closure is supplied by the trainer (same as 119).
- `optimizers/__init__.py`: export `LookSAM`.
- `configs/llm_config.py`:
  - Add to `LLMConfig` (around the SAM block, line ~720):
    `use_looksam: bool = False`, `looksam_k: int = 5`,
    `looksam_rho: float = 0.05`.
  - Add CLI flags in the `--use_sam` neighborhood:
    `--use_looksam`, `--looksam_k`, `--looksam_rho`.
  - Add `Tiny1M3MLookSAMConfig(Tiny1M3MConfig)` that sets
    `use_looksam=True, looksam_k=5, looksam_rho=0.05`
    (paper defaults).
- `training/trainer.py`: when `use_looksam=True` *and*
  `use_sam=False` (mutex — 119 and 138 are alternate paths),
  replace the AdamW optimizer with `LookSAM` and route the SAM
  ascent/closure/descent logic through it. The `step_count`
  lives inside `LookSAM`; the trainer still calls the same
  `sam_opts` / `non_sam_opts` flow but the SAM step fires only
  every K steps (steps 0..K-1 are plain AdamW; step K-1 fires
  the SAM flow at index K-1, so the first step is plain AdamW
  ⇒ byte-identical to AdamW at step 0). When `use_looksam=
  False` (default) the trainer uses plain `AdamW` unchanged.

**Identity at step 0**
- The first optimizer step (step 0) is `inner.step()` on the
  baseline AdamW path (no ascent, no closure). With K=5 the
  SAM step fires at the 5th step (step index 4, i.e. after
  steps 0..3 of plain AdamW). This is *more* bit-identical at
  step 0 than full SAM (119).
- The `use_looksam=False` path is fully inert (no LookSAM
  object constructed).

**Run command**
```
python train_llm.py --config_class configs.llm_config.Tiny1M3MLookSAMConfig --seed 42
```
or equivalently
```
python train_llm.py --config tiny1m3m --use_looksam true --looksam_k 5 --looksam_rho 0.05 --seed 42
```

**Val loss readback**
Final val loss is written to `plots/metrics_<train_tokens>_<ts>.json`
and read by the pipeline from `final_metrics['val_loss']` (the
trainer returns it as `results['final_metrics']['val_loss']`).
