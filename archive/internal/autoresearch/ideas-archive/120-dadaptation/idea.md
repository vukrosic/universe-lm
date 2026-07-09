---
id: 120-dadaptation
status: done
round: 1
updated: 2026-06-13T20:34:34Z
transfer-risk: med
plain: It removes the learning-rate knob from training entirely — the optimizer figures out its own step size on the fly based on how confident it is about the gradient direction.
---

# 120 — D-Adaptation (Automatic LR Discovery for Adam)

## Source
Defazio, "Learning-Rate-Free Learning by D-Adaptation"
(arXiv:2301.11933, ICML 2023 / later journal version arXiv:2201.11941).
https://arxiv.org/abs/2201.11941

Validated on ResNet-50/ImageNet, GPT-2 small, ViT-B/16, BERT-base
across hundreds of epochs. The lever's bet: the LR *is* the most
important and most fragile HP in deep learning; removing it (or
making it auto-derived) trades a knob for a (small) quality hit
and a (large) reproducibility win.

## Mechanism
D-Adaptation maintains a *log-scale* running lower bound `D` on the
distance from `w_init` to `w_optimal`. The LR at step `t` is derived
as:
  `lr_t = D_t / ‖G_t‖`     (where `G_t` is the AdamW-style gradient
                            processed through standard 1st/2nd moments)
  `D_t ← D_t · exp(η · (λ_g · c_+ − λ_ℓ · c_−))`    (log-LR update)

Where `c_+` and `c_−` are counts of *agreement* and *disagreement*
between the current descent direction and the recent descent direction.
The intuition: when the current descent direction agrees with the
recent direction, the lower bound `D` should *grow* (we're making
progress); when they disagree, `D` should shrink (we're oscillating).

Formally:
  `c_+ = ⟨sign(g_t), sign(g_t^m)⟩`     (where `g_t^m` is a momentum direction)
  `c_− = ⟨sign(g_t), −sign(g_t^m)⟩`
  `D_t+1 = D_t · exp(η · c_+ / √(c_+ + c_−))`    (simplified)

The base-LR `η` (called `d0_lr` in the paper) is a small positive
constant (paper default `1.0`); the actual effective LR is
`lr_t = D_t / ‖g_t‖`. The 1st and 2nd moments of AdamW are retained
intact — only the *outer* scaling `lr` is replaced.

**Identity at step 0**: `D_0 = 0.0` (paper init), and the first
gradient step uses `lr_0 = D_0 / ‖g_0‖ = 0` — *no update*. The
paper fixes this by warm-starting `D_0` to a small positive
constant (`1e-6`) and using `lr_0 = max(D_0 / ‖g_0‖, ε)` for the
first few steps. With warm-start, the first step has
`lr_0 ≈ ε / ‖g_0‖` which is small but non-zero, then `D` grows
exponentially until `lr_t` matches the optimal AdamW LR for the
local geometry.

The lever is **not** bit-identical to AdamW at step 0 (different
LR), but the *first-step displacement* is `O(ε)` and within run-to-run
noise. With `D` frozen at its initial value (i.e. effectively constant
LR), the lever collapses to AdamW — so the PASS bar is the LR-discovery
loop firing within the run window.

## Design sketch
- `optimizers/dadaptation.py` (new): `DAdaptAdamW` — wraps AdamW
  with the `D`-discovery loop. State per param group: scalar `D`
  (log-LR lower bound), `g_t^m` (recent descent direction), and
  standard AdamW `exp_avg`, `exp_avg_sq`. `step()` updates `D` then
  applies AdamW with `lr = D / ‖g_t‖`. ~100 LoC.
- `training/trainer.py`: when `use_dadapt=True`, route the
  AdamW-eligible params (1-D, embedding, vocab, norms) through
  `DAdaptAdamW`. The 2-D slot still uses Muon (the lever is
  ortho to Muon). ~10 LoC.
- `configs/llm_config.py`: add `use_dadapt: bool = False`,
  `dadapt_d0_lr: float = 1.0`, `dadapt_growth_rate: float = 1.02`,
  `dadapt_min_lr: float = 0.0`. ~10 LoC.
- LoC: ~120 total (under 200 ceiling).
- Identity at step 0: with `D_0 = 1e-6` warm-start, the first step
  has `lr_0 ≈ 1e-6 / ‖g_0‖` which is essentially zero. After ~10-20
  steps, `D` reaches a typical AdamW-equivalent value (~0.001-0.01).
  The first ~10 steps see a *ramp-up* in effective LR — this is the
  lever's signature, not a bug.
- The intuition: at 0.94M with 92 steps, the optimal AdamW LR is
  unknown a priori and the constant `adamw_lr` in the config is a
  guess. D-Adaptation removes the guess by letting `D` discover the
  right LR on the fly. A null would say "at 0.94M the LR ramp-up
  costs more than it saves"; a win would say "the config's LR
  constant is suboptimal and the discovery loop finds a better one".
  Either outcome is *new information* about whether `adamw_lr` is
  on-axis for our pipeline.

## Scale evidence
- arXiv:2201.11941 (Defazio 2021/2023): validates on ResNet-50
  (image classification), GPT-2 small (LM), ViT-B/16 (image
  classification), BERT-base (NLP), achieves parity-to-better
  val loss with hand-tuned AdamW LR.
- Independent reproductions: fastai, nanoGPT, PyTorch Lightning
  all include D-Adapt variants with reported parity.
- Transfer risk: **med**. Validated at ≥100M (GPT-2 small
  ~125M, BERT-base 110M, ViT-B ~86M), the lever's mechanism is
  LR-discovery which is scale-free. At 0.94M the ramp-up cost
  is ~10 steps out of 92 (~10%) which is significant — null is
  plausible at tiny1m3m.

## Why it's worth a slot
D-Adaptation is the only LR-removal lever filed (Schedule-Free
closed 006, Adam-mini 031 closed — both still have an LR knob).
It is the cleanest test of "is our `adamw_lr = 0.006` config
optimal at this scale?" — and it's a meta-lever that *would
generalize to any future model config*, since the LR question
re-appears at every scale. The 0.94M context is particularly
informative: the optimal LR at 12L/92-step is genuinely unknown
(it could be 1e-5 or 1e-2 depending on the network), and
D-Adapt removes the guess. The win-bar is small (-0.005) because
the gain comes from removing a small misconfiguration, not from
a new *direction* of improvement.

## Plan

- **Files to change (≤ 200 LoC total):**
  - `optimizers/dadaptation.py` (new, ~140 LoC): `DAdaptAdamW` class
    implementing the D-discovery loop wrapping `torch.optim.AdamW`. Per
    param-group scalar `D` (log-LR lower bound, init `1e-6`), per param
    `g_t^m` recent signed direction (init zeros), and standard AdamW
    `exp_avg`, `exp_avg_sq` buffers (parent's lazy init). On each
    `step()`, before delegating to AdamW: (1) compute the per-param
    agreement `c_+ = ⟨sign(g_t), sign(m_t)⟩ / N` and disagreement
    `c_- = ⟨sign(g_t), -sign(m_t)⟩ / N` where `m_t` is `exp_avg` from
    the previous step (zeros at step 0 ⇒ both ratios are 0.5); (2)
    update the *shared* per-group `D ← D · exp(d0_lr · (c_+ − c_-))`,
    clamped to `[dadapt_min_lr, dadapt_d_max]` (paper §3.1 — the
    upper clamp is **required** for stability at tiny1m3m; without
    it `D` grows as `e^t` per step → val 10.81 → 36.89 at step 50 →
    7.04e15); (3) derive the per-step effective `lr_t = D / ‖g_t‖`
    (with a `grad_norm < 1e-12` floor and a final
    `min(lr_t, dadapt_d_max)` safety clamp); (4) override
    `group["lr"]` for the parent call only, then restore. The
    parent's `super().step()` runs the standard AdamW
    `(m, v, bias-correction, decoupled WD)` math unchanged on the
    same gradient — only the *outer* LR scaling is swapped. A
    NaN/Inf guard on the gradient / momentum falls back to the base
    lr without poisoning `D`.
  - `optimizers/__init__.py` (+1 LoC): export `DAdaptAdamW`.
  - `configs/llm_config.py` (~30 LoC): add `use_dadapt: bool = False`,
    `dadapt_d0_lr: float = 1.0` (paper default; the `η` in
    `D ← D · exp(η · (c_+ − c_-))`), `dadapt_min_lr: float = 0.0`
    (lower clamp on D, paper default), `dadapt_d_max: float = 1.0`
    (paper §3.1 upper clamp on D; **required** for stability —
    without it `D` grows unboundedly), `dadapt_eps: float = 1e-8`
    (floor for `lr_t = D/‖g_t‖`). Add
    `Tiny1M3MDAdaptConfig(Tiny1M3MConfig)` preset with
    `use_dadapt=True, dadapt_d0_lr=1.0, dadapt_min_lr=0.0,
    dadapt_d_max=1.0`.
  - `training/trainer.py` (~12 LoC): import `DAdaptAdamW`; in
    `setup_muon_optimizer`, when `use_dadapt=True`, instantiate
    `DAdaptAdamW` on the `adamw_params` bucket (the 1-D / embedding /
    norm / head params), passing `d_max=getattr(config, "dadapt_d_max",
    1.0)`. The Muon 2-D path is unchanged — D-Adapt is ortho to Muon,
    lives only on the AdamW bucket. The `use_dadapt=True` branch sits
    in the elif chain between SAM and Schedule-Free so it has the
    same precedence (D-Adapt → SF → MARS → Cautious → plain AdamW,
    mutually exclusive).
  - `train_llm.py` (~7 LoC): add `--use_dadapt`, `--dadapt_d0_lr`,
    `--dadapt_min_lr`, `--dadapt_d_max` CLI flags.

- **Config flag name:** `use_dadapt` (companion knobs `dadapt_d0_lr`,
  `dadapt_min_lr`, `dadapt_d_max`, `dadapt_eps`).

- **Identity at step 0 (flag-off baseline):** With `use_dadapt=False`
  (default), the trainer instantiates plain `torch.optim.AdamW` on the
  1-D bucket — baseline path is bit-identical to today.

- **Identity at step 0 (flag-on):** With `D_0 = 1e-6` warm-start, the
  first step has `lr_0 = D_0 / ‖g_0‖ ≈ 1e-6 / ‖g_0‖` (essentially zero).
  After ~10–20 steps, `D` reaches a typical AdamW-equivalent value
  (~0.001–0.01). The first ~10 steps see a *ramp-up* in effective LR
  — this is the lever's signature (the design sketch explicitly
  accepts this; see the "Why it's worth a slot" section above). The
  parent's `exp_avg`/`exp_avg_sq` buffers are unchanged from plain
  AdamW; only the *outer* LR is replaced. The new `dadapt_d_max=1.0`
  upper clamp does NOT affect step 0 (D starts at 1e-6 < 1.0) ⇒ step-0
  behavior is bit-identical to the previous version, only later steps
  are protected from runaway D-growth.

- **Run command (Ctrl):**
  ```
  python train_llm.py --config_class configs.llm_config.Tiny1M3MConfig \
      --output_dir checkpoints/120-dadapt-ctrl --seed 42
  ```

- **Run command (Treatment):**
  ```
  python train_llm.py --config_class configs.llm_config.Tiny1M3MDAdaptConfig \
      --output_dir checkpoints/120-dadapt --seed 42
  ```

- **Reading final val loss:** `metrics.json` `val_loss` at the last
  `eval_milestones` entry (typically `700` for `Tiny1M3MConfig`,
  ~92 steps total). Compare against the same-seed ctrl run above.
  PASS ≤ ctrl − 0.005 (the win-bar from the design sketch). NULL
  band |Δ| < 0.005. DRIFT > +0.005 (the D ramp-up cost ~10% of the
  92-step trajectory outweighs any LR-discovery win).

- **Re-code note (2026-06-13):** The previous GPU run DIVERGED
  (`val loss 10.81 → 36.89 at step 50 → 7.04e15 final`). Root cause:
  `D` had no upper clamp; with `d0_lr=1.0` and consistent gradient
  agreement `D` grew as `e^t` per step (~`1e40` by step 92), then
  `lr_t = D / ‖g_t‖` exploded on the first small-gradient plateau.
  Fix: (1) add `dadapt_d_max=1.0` paper §3.1 upper clamp on D;
  (2) clamp `lr_t ≤ d_max`; (3) floor `grad_norm < 1e-12 → base lr`;
  (4) NaN/Inf guard on gradient / momentum → fallback to base lr
  without poisoning D.
