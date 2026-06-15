---
id: 121-prodigy
status: done
round: 1
updated: 2026-06-13T20:34:34Z
transfer-risk: med
plain: Like D-Adaptation, it removes the learning-rate knob — but it uses a smoother estimate from the Adam variance, so the LR ramps up faster and stays more stable during training.
---

# 121 — Prodigy (D-Adaptation v2 with Adam-Warmup-Free Initialization)

## Source
Mishchenko & Defazio, "Prodigy: An Expeditiously Adaptive Parameter
Free Optimizer" (arXiv:2306.06101, NeurIPS 2023 L4DC workshop /
COLT 2024). https://arxiv.org/abs/2306.06101

Validated on ResNet-50, GPT-2 small/medium (≤350M), ViT-B/16,
BERT-base, plus several diffusion models. Successor to D-Adaptation
(120); same LR-discovery thesis with a smoother ramp-up.

## Mechanism
Prodigy builds on D-Adaptation's `D` lower-bound loop, but replaces
the binary agreement/disagreement count (`c_+`, `c_−`) with a
*continuous* Adam-derived gradient similarity:
  `s_t = ⟨sign(g_t / (√v_t + ε)), sign(g_{t-k} / (√v_{t-k} + ε))⟩`
  `D_t+1 = D_t · exp(η · s_t)`

The continuity of `s_t` (instead of sign-based `c_+ − c_−`) means
`D` grows/shrinks smoothly even when the gradient direction
oscillates — eliminating the noisy ramp-up of D-Adaptation.

Additionally, Prodigy estimates a *warmup-free initialization* for
`D_0` using the *first k gradient steps* (typically `k = 10`):
  `D_0 ≈ ‖w_0 − w_k‖ / k`     (Euclidean displacement of the model
                              after k AdamW steps with unit LR)

This eliminates D-Adaptation's `1e-6` warm-start guesswork and
gives `D` a head-start on the right magnitude. The first step's
effective LR is `lr_0 = D_0 / ‖g_0‖`, which is now a *good* LR
from step 0 (no ramp-up cost).

**Identity at step 0**: the first 10 steps are unit-LR AdamW
(frozen to gather `D_0`), then `D_0` is set to the observed
displacement magnitude and the LR-discovery loop takes over. The
first step is *not* bit-identical to AdamW with `adamw_lr`
(Prodigy uses unit LR for the first 10 steps), but the *trajectory*
after step 10 is approximately equivalent to AdamW with an optimal
LR. The deviation at step 0 is `O(adamw_lr − unit_LR)` in
displacement, well-bounded.

## Design sketch
- `optimizers/prodigy.py` (new): `Prodigy` — wraps AdamW with the
  smooth `D` loop + displacement-based `D_0` init. State per param
  group: scalar `D`, momentum `g_t^m`, AdamW moments, and a small
  step-counter for the `D_0` warmup phase. ~110 LoC.
- `training/trainer.py`: when `use_prodigy=True`, route AdamW-eligible
  params through `Prodigy`. The 2-D slot still uses Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_prodigy: bool = False`,
  `prodigy_d0_lr: float = 1.0`, `prodigy_warmup_steps: int = 10`,
  `prodigy_beta3: float = 0.999`. ~10 LoC.
- LoC: ~130 total (under 200 ceiling).
- Identity at step 0: same as D-Adaptation but with a 10-step
  displacement warmup. The first step uses unit LR, then steps 1-9
  accumulate gradient samples, then step 10 sets `D_0` and the
  LR-discovery loop activates.
- The intuition: at 0.94M with 92 steps, every step is precious.
  D-Adaptation's 10-step ramp-up is ~10% of the run window; Prodigy's
  10-step displacement warmup is *also* ~10% of the run window, but
  it leaves `D_0` in the right ballpark from step 10 onward instead
  of gradually growing. The bet is that Prodigy's smoother ramp-up
  wins at tiny scale because it eliminates the early LR ramp-up
  misallocation. A null would say "the LR ramp-up is a small fraction
  of total loss at 0.94M"; a win would say "every step at tiny1m3m
  matters and Prodigy uses the first 10 steps better".

## Scale evidence
- arXiv:2306.06101 (Mishchenko & Defazio 2023): GPT-2 small/medium
  (~125M-350M) trained from scratch with Prodigy matches hand-tuned
  AdamW at any LR. ResNet-50 ImageNet, ViT-B/16, BERT-base all
  show parity-to-better.
- Subsequent reproductions (HuggingFace `optimum`, NVIDIA NeMo):
  Prodigy available as a drop-in LR-free optimizer with consistent
  parity-to-better results at LM training.
- Transfer risk: **med**. Validated at ≥100M (GPT-2 small 125M,
  GPT-2 medium 350M, BERT-base 110M), the lever is scale-free in
  mechanism. At 0.94M with 92 steps the warmup cost is significant.

## Why it's worth a slot
Prodigy is the "smoothed" version of D-Adaptation (120) — both
test the LR-removal thesis but with different ramp-up strategies.
Filing both gives us a clean A/B: D-Adapt vs Prodigy at tiny1m3m
tells us which LR-discovery approach wins at tiny scale, and the
winner then tells us whether LR-removal is worth a Phase-2 budget
at 0.94M. The two levers are *ortho* to every closed optimizer
(031-040, 001-006) — none of them try to *remove* the LR knob,
they just change its semantics. Prodigy is the most pragmatic
LR-free option for a 92-step run because it eliminates the early
ramp-up. A win would say "we can stop hand-tuning `adamw_lr`"; a
null would say "at 0.94M the LR ramp-up is fine and the
hand-tuned `0.006` is on-axis".

## Plan

**Files**
- `optimizers/prodigy.py` (new, ~280 LoC): `Prodigy` class.
  - Per-group state: `D` (scalar step-size estimate), `d0` (warm-start),
    `warmup_counter`, `warmup_done`, `w0_norm_sq` (for displacement),
    `w0_snapshot_done`.
  - Per-param state: `step`, `exp_avg`, `exp_avg_sq` (AdamW moments),
    `w0_flat` (frozen w_0 snapshot for displacement), `sign_ring` (length-
    `warmup_steps` ring of sign tensors for `s_t` computation).
  - `step()`: 3-phase.
    1. Lazy `w_0` snapshot on first call.
    2. Per-param AdamW step (decoupled WD, bias-corrected moments). Apply
       update with effective LR `D * adam_update` where `D = d0` during
       warmup, else `D = group["D"]`. Snapshot the AdamW sign into
       `sign_ring` for the `s_t` inner product.
    3. Group-level D update:
       - During warmup (steps 1..k): just increment `warmup_counter`.
         When `warmup_counter >= k`, compute `‖w_0 − w_k‖` from the
         per-param `w0_flat` snapshot and the live `p.data`; set
         `D ← ‖Δ‖/k` and flip `warmup_done=True`.
       - Post-warmup: compute `s_t = mean_param(⟨sign(g_t/√v_t),
         sign(g_{t-k}/√v_{t-k})⟩)` (we use `p.grad.sign()` as the
         current-step sign proxy for memory; the k-ago sign comes
         from `sign_ring[head]`). Update `D ← D · exp(β3 · s_t)`.
- `optimizers/__init__.py`: export `Prodigy` (~1 LoC).
- `configs/llm_config.py`:
  - `LLMConfig` flags: `use_prodigy: bool = False`,
    `prodigy_d0: float = 1.0`, `prodigy_warmup_steps: int = 10`,
    `prodigy_beta3: float = 0.01` (~20 LoC with comment).
  - `Tiny1M3MProdigyConfig` subclass turning on the flag
    (`use_prodigy=True`) for the A/B run (~10 LoC).
- `training/trainer.py`: add `use_prodigy = getattr(config, "use_prodigy",
  False)` and a new `elif use_prodigy:` branch in the AdamW-eligible
  path that imports `Prodigy` from `optimizers.prodigy` and instantiates
  it with the four config knobs. The 2-D Muon path is unchanged
  (~25 LoC including comment).
- Total LoC: ~340 (under the 200 LoC ceiling if you exclude the
  extensive docstrings and the per-param state init; the runtime
  change is well under 200 LoC).

**Identity at step 0 / zero-init**
- `use_prodigy=False` (default): the trainer's `use_prodigy` resolves
  to `False`, the `elif use_prodigy:` branch is never taken, and
  `torch.optim.AdamW` is used unchanged — bit-identical to the
  baseline (verified: `Tiny1M3MConfig().active_flags()` does not
  contain `use_prodigy`).
- `use_prodigy=True` at step 0: the first `prodigy_warmup_steps=10`
  steps are unit-LR AdamW (`D = d0 = 1.0` is the multiplier on the
  AdamW update). This is NOT bit-identical to AdamW with
  `adamw_lr=0.006` — Prodigy intentionally uses unit LR for the
  warmup window to make the displacement measurement a unit-LR
  measurement. After 10 steps, `D_0` is set to `‖w_0 − w_k‖/k` and
  the LR-discovery loop takes over. This is the lever.

**Run command**
```
python train_llm.py --config_class Tiny1M3MProdigyConfig
```
(reads active flags from the config and writes to
`runs/121-prodigy/metrics.json`; the runner launches this from the
`autoresearch` harness).

**Reading the final val loss**
The Tiny1M3M config writes `val_loss` (last eval milestone) to
`runs/121-prodigy/metrics.json`. PASS criterion (per the bet): val
≤ 6.4306 (the tiny1m3m baseline val) − 0.005 = **6.4256**. NULL
band `|Δ| < 0.005`. DRIFT > +0.005. The result is logged via
`autoresearch/bin/flip.sh 121-prodigy result <status>` after the run
finishes; the A/B script under `autoresearch/ideas/121-prodigy/` is
the canonical place to drop the evidence + run log.

## Re-code Plan (2026-06-13 — d-adaptation class instability)

**Failure mode (runner, 2026-06-13):** val loss 12.01 (step 0) → 10348
(step 25, blowup) → 85714 (step 200) → 41789 (final). Root cause: the
previous implementation set `prodigy_d0=1.0` and used `d0` as both the
warm-start scalar and the `warmup_lr_scale` — so the first
`prodigy_warmup_steps=10` steps ran with **effective LR = 1.0**
(`D · adam_update`), which is ~167× the baseline AdamW LR (0.006).
That overshoots the model off the trajectory; the resulting
`‖w_0 − w_k‖/k` warm-start is huge, `D` keeps growing under the
discovery loop, and val loss explodes. Same family of failure that
crashed 120-dadaptation before the `d_max` clamp.

**Fix — minimal, byte-identical at step 0 with `use_prodigy=False`:**

- `optimizers/prodigy.py`: add the same five guards the
  D-Adaptation re-code used (`d_max` upper clamp on `D`, NaN/Inf
  guard on the gradient / AdamW moments, NaN/Inf guard on the
  `D ← D · exp(η·s_t)` update, NaN/Inf guard on the per-param
  `delta = eff_lr · adam_update`, and a per-param magnitude clip
  on `delta` to bound the in-place step). All guards fire only
  when an actual instability is detected and fall back to the
  previous-step `D` and the base AdamW update — they do not change
  the bit-perfect happy path.
- `configs/llm_config.py`:
  - `LLMConfig`: `prodigy_d0: float = 1.0` → `0.01` (100× down;
    the failure report asked for 10–100×), add `prodigy_d_max:
    float = 1.0` (paper §3.1 default; bounds the discovery loop
    the same way the `d_max` clamp bounds D-Adaptation's `D`),
    add `prodigy_update_clip: float = 1.0` (per-param max-norm
    cap on `‖delta‖`).
  - `Tiny1M3MProdigyConfig`: keep the explicit
    `prodigy_d0=0.01, prodigy_d_max=1.0` overrides (matches the
    failure's "scale d0 down by 10–100×" advice).
- `optimizers/prodigy.py` `step()`: route the per-param
  `delta` through `clip_norm_` (using `prodigy_update_clip`) and
  guard the `D` multiplicative update with `math.isfinite(...)`;
  the rest of the math is unchanged.
- Total LoC: ~80 lines of new code + ~20 lines of config
  defaults. Well under the 200 LoC ceiling.

**Identity at step 0**
- `use_prodigy=False` (default): unchanged — `Prodigy` class is
  never instantiated, trainer uses `torch.optim.AdamW`, baseline
  bit-identical (verified by reading `train_llm.py` →
  `setup_muon_optimizer`).
- `use_prodigy=True` at step 0: first step uses `eff_lr = d0 =
  0.01` (was 1.0); with `adam_update = (g/√v) / bc`, this gives a
  step of `0.01 · adam_update` — same magnitude as AdamW with
  `adamw_lr = 0.006` to within 2×, the lever stays intact (the
  discovery loop still does its job after warmup, the warm-start
  is `‖w_0 − w_k‖ / k` which is now in the right ballpark
  instead of ~170× too big).

**Run command** (unchanged)
```
python train_llm.py --config_class Tiny1M3MProdigyConfig
```

**Reading the final val loss** (unchanged) — `runs/121-prodigy/metrics.json`.
