---
id: 110-weight-ema
status: done
round: 1
updated: 2026-06-13T14:21:30Z
transfer-risk: low
plain: It tries to average the model's weights over training so the version it scores on is the calm middle of the noise, not the latest jittery step.
---

# 110 — Model-Weight EMA (Polyak Averaging) for Val

## Source
Polyak, B.T. (1990) "A new method of stochastic approximation type";
Polyak-Ruppert averaging as used in RoBERTa (Liu et al. 2019, ~355M),
BYOL (Grill et al. 2020), MoCo v3 (Chen et al. 2021, ViT-B/L at 100M+),
MAE (He et al. 2021, ViT-L at 300M+), and AdamW with EMA evaluation in
modded-nanogpt speedrun recipes.

## Mechanism
Maintain a shadow copy `θ_ema ← μ·θ_ema + (1−μ)·θ` updated each step
(with `μ ≈ 0.999` and a "warm-up" ramp from 0 to `μ` over the first ~100
steps to avoid the EMA being pinned to the init). At each eval milestone,
compute the val loss on `θ_ema` *in addition to* the live `θ`. Two
plumbing choices are plausible: (a) score only the EMA copy and ignore
`θ` for val, (b) keep the live `θ` as the saved checkpoint but report
EMA val loss as the run metric. (a) is the "Polyak-Ruppert as the
inference model" choice (closer to the paper). Implementation lives in
`train_llm.py` after `optimizer.step()`: a one-liner EMA update over
named trainable parameters; an opt-in eval branch swaps the model
parameters in via a `with torch.no_grad(): copy_(ema, model)` context
then runs the standard `evaluate(...)` function, restoring on exit.

## Design sketch
- `configs/llm_config.py`: add `use_ema_eval: bool = False`,
  `ema_decay: float = 0.999`, `ema_warmup_steps: int = 100`,
  `ema_eval_only: bool = True` (when True, val scores EMA only).
- `train_llm.py`: instantiate `ema_params = {n: p.detach().clone() for
  n,p in model.named_parameters() if p.requires_grad}` after model
  construction. After each `optimizer.step()`, run
  `for n,p in model.named_parameters(): ema_params[n].mul_(decay).add_(p.detach(), alpha=1-decay)`.
  In eval, when `ema_eval_only=True`, `model.load_state_dict({n: p for n,p in ema_params.items()}, strict=False)`, run eval, restore.
- LoC: ~40 (excluding the dataclass fields).
- Identity at step 0: `μ=0` (during the 100-step warm-up) ⇒
  `θ_ema = θ_init`, so the EMA copy at step 0 is the live model, and
  the val score at step 0 is bit-identical to baseline. By step 100
  the EMA tracks `θ` closely and gradually tightens to `μ=0.999`.
- The intuition: at tiny1m3m the gradient noise is huge relative to
  the signal (only ~92 steps × 32k tokens/step = 3M tokens total, with
  no warmup past 2%), and the live `θ` oscillates around the EMA. The
  EMA averages out the oscillation and reports the "calm" minimum the
  trajectory is circling — a real effect at any scale but more
  pronounced when the per-step signal-to-noise ratio is low.

## Scale evidence
- RoBERTa (~125M-355M): Polyak averaging is the default inference
  recipe in the original paper.
- MAE ViT-L (~300M): EMA evaluation is standard.
- modded-nanogpt speedrun: `torch.optim.swa_utils.AveragedModel` is
  used in several top entries; decay 0.999 is a default.
All three are ≥100M; transfer risk is **low**.

## Why it's worth a slot
This is one of the cleanest "the eval point, not the model, is the
lever" tests in the whole zoo — the mechanism is a one-line after-step
update, identity at step 0, and a null says "polyak averaging doesn't
matter at this scale" (which is itself useful, because it would mean
our baseline trajectory is already calm enough at 92 steps that
averaging buys nothing). A win of even 0.005-0.01 on val would compound
with the other WINS in the leaderboard and not require touching any
architecture code.

## Plan

- **Files touched:**
  - `configs/llm_config.py` — add 4 EMA flags to `LLMConfig`:
    `use_ema_eval: bool = False`, `ema_decay: float = 0.999`,
    `ema_warmup_steps: int = 100`, `ema_eval_only: bool = True`. Add
    `Tiny1M3MEMAConfig(Tiny1M3MConfig)` preset with `use_ema_eval=True`
    (the A/B treatment subclass).
  - `training/trainer.py` — add `ModelEMA` helper class with
    `.state_dict()` so checkpointing round-trips, a one-line EMA update
    after `optimizer.step()` (decay ramps from 0 → `ema_decay` over
    `ema_warmup_steps` ⇒ step-0 EMA ≡ live model ⇒ step-0 val byte-
    identical to baseline), and an `ema_eval_only` branch that swaps
    `p.data` ← `ema` before `evaluate_model(...)`, runs the standard
    eval, then restores `p.data` ← `live`. The live `θ` is the
    checkpointed / saved / resumed model — only the *val score* moves.
  - `train_llm.py` — add CLI flags `--use_ema_eval`, `--ema_decay`,
    `--ema_warmup_steps`, `--ema_eval_only` (matching the existing
    flag pattern).

- **Config flag name:** `use_ema_eval` (plus `ema_decay`,
  `ema_warmup_steps`, `ema_eval_only`).

- **Zero-init at step 0:** `decay(step=0) = 0 / ema_warmup_steps = 0`,
  so the EMA update is `θ_ema ← 0·θ_ema + 1·θ = θ_init`. The val score
  at step 0 reads `θ_ema == θ_live` ⇒ bit-identical to baseline.

- **Run command (treatment):**
  ```bash
  python train_llm.py \
    --config_class configs.llm_config.Tiny1M3MEMAConfig \
    --seed 42 --dataset_path processed_data/pretrain_1B \
    --warmup false
  ```
  Or the local-equivalent `python _arq_110-weight-ema.py`.

- **Reading final val loss:** `plots/metrics_<TOKENS>_<TS>.json` →
  `final_metrics.val_loss` (same convention as every other idea; the
  runner pulls this from `remote-results/.../results.json`).
