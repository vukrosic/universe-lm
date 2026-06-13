---
id: 112-lookahead-opt
status: done
round: 1
updated: 2026-06-13T10:43:16Z
transfer-risk: med
plain: It tries to slow the optimizer down by taking a few fast steps in one direction and then averaging back, so the model doesn't overshoot in any one step.
---

# 112 — Lookahead Optimizer Wrapper

## Source
Zhang et al. 2019, "Lookahead Optimizer: k steps forward, 1 step back",
arXiv:1907.08610 https://arxiv.org/abs/1907.08610.
Validated at ResNet-50 ImageNet, Transformer-WMT14, and various
reinforcement learning setups. The "k inner steps + 1 outer step"
mechanism is orthogonal to the base optimizer — Lookahead wraps *any*
inner optimizer (SGD, AdamW, Muon) and pulls the model weights back
toward a slow trajectory every k steps.

## Mechanism
Maintain two sets of weights: "fast" (the live model) and "slow"
(an EMA-style copy). The inner optimizer (Muon + AdamW, unchanged)
updates the fast weights for `k` steps. Then every `k` steps, the
slow weights pull halfway toward the fast weights:
`slow ← slow + α·(fast − slow)`, and the fast weights are reset to
`slow`. The hyperparameter `α ∈ (0, 1]` is the "slow step size" (paper
default 0.5); `k` is the "inner cycle length" (paper default 5-10).

Concretely: the wrapper sits *outside* `optimizer.step()` in
`train_llm.py`. Every training step:
1. Run `optimizer.step()` as usual (updates `model.parameters()`).
2. Increment a counter; if counter mod `k == 0`:
   `for n,p in model.named_parameters(): slow[n].add_(p.detach() - slow[n], alpha=alpha); p.data.copy_(slow[n])`.

Identity at step 0: `slow = θ_init` and the first inner step is
`optimizer.step()` on the baseline Muon/AdamW. The lookahead update
only fires at step `k`; before that, the wrapper is a no-op. With
`use_lookahead=False` (default), the wrapper is fully inert.

## Design sketch
- `configs/llm_config.py`: add `use_lookahead: bool = False`,
  `lookahead_k: int = 5` (sync every 5 inner steps; paper default
  is 5-10), `lookahead_alpha: float = 0.5` (slow step size; paper
  default).
- `train_llm.py`: after `optimizer = build_optimizer(...)`, wrap
  with a thin Python class or just two dicts and a counter. The
  wrapper does not touch the inner optimizer's state — it just
  overwrites the fast weights and (importantly) overwrites the
  inner optimizer's state back to the slow position (otherwise the
  inner optimizer's momentum buffers would accumulate stale
  gradients from before the slow reset, causing the next inner step
  to overshoot). This buffer-reset is the one subtle bit: on the
  outer step, `optimizer.state[p] = None` (or re-init) so the inner
  momentum doesn't carry across the slow pull.
- LoC: ~30 (wrapper class) + 5 (config fields).
- Identity at step 0: `slow = θ_init`, first inner step uses baseline
  Muon/AdamW. At step `k` (step 5 by default), `slow ← slow +
  0.5·(fast − slow) = 0.5·θ_init + 0.5·θ_5`. The fast reset to
  `slow` is the only non-baseline effect.
- The intuition: at tiny1m3m the per-step gradient noise is high and
  the inner optimizer (Muon/AdamW) takes aggressive steps that
  overshoot a local minimum, then oscillates. The slow weights
  "average" the trajectory every 5 steps, smoothing out the
  oscillation. Paper shows this lets the inner optimizer use a
  *larger* effective LR; with `k=5` and `α=0.5` the slow trajectory
  advances at half the per-step speed but in a much straighter line.

## Scale evidence
- Paper validates on ResNet-50 ImageNet (~25M params), Transformer-
  WMT14 (~200M params), and Pile of Toy RL tasks.
- Subsequent work (e.g. Ranger, Ranger21) wraps Lookahead around
  AdamW/RAdam at ImageNet scale (~25-100M).
- modded-nanogpt: Lookahead is *not* a default, but several top
  speedrun entries use it informally.
Scale evidence is mid-range (10-100M, not 100M+); transfer risk
**med**.

## Why it's worth a slot
The closed optimizer zoo (031-040: Adam-mini, LAMB, APOLLO, etc.)
all swap the *inner* optimizer. Lookahead is the only lever that
wraps *any* inner optimizer and changes the trajectory shape
without changing the per-step math. Critically, it composes with
the WINS already on the leaderboard (Muon, QK-norm, Moonlight
Muon) — those are inner optimizers / norm placements; Lookahead
sits outside. A null would say "trajectory averaging at this step
count and depth is too coarse to help"; a win would compound with
existing WINS and open a new axis (trajectory-level wrappers) for
the portfolio.

## Plan

**Files touched**
- `configs/llm_config.py`: add `use_lookahead: bool = False`,
  `lookahead_k: int = 5`, `lookahead_alpha: float = 0.5` to `LLMConfig`.
  Add `Tiny1M3MLookaheadConfig(Tiny1M3MConfig)` that sets
  `use_lookahead=True` and keeps the paper defaults k=5, α=0.5.
- `training/trainer.py`: add `Lookahead` wrapper class. Wire it
  into `train_minimal_llm` (constructed after checkpoint load so the
  slow snapshot matches the live model state) and pass it as
  `lookahead=` into `train_model`. In `train_model`, call
  `lookahead.step()` right after the inner optimizer loop. With
  `use_lookahead=False` the wrapper is `None` → fully inert.
- `train_llm.py`: add `--use_lookahead`, `--lookahead_k`,
  `--lookahead_alpha` CLI flags + override block.

**Zero-init at step 0**: `Lookahead.__init__` snapshots `slow =
θ_init` clones; the outer-step sync only fires every `k` inner steps,
so the first `k-1` training steps are pure baseline (Muon/AdamW). The
final val loss is read from `plots/metrics_<timestamp>.json` /
`metrics.json` (the same way other experiments in this repo are
read).

**Run command** (tiny1m3m seed 42):
```
cd /root/universe-lm && /venv/main/bin/python train_llm.py \
  --config_class configs.llm_config.Tiny1M3MLookaheadConfig \
  --output_dir runs/112-lookahead-opt/seed42 \
  --seed 42
```
Mirror with `Tiny1M3MConfig` (no flag) for the ctrl.

**LoC budget**: ~70 LoC (Lookahead class ~50 + config flag block ~5 +
config class ~12 + CLI overrides ~10). Well under 200.

**PASS bar**: ≤ ctrl − 0.005 on val_loss (taste's mid-band for a
trajectory-smoothing wrapper at 12L depth). NULL band |Δ| < 0.005.
DRIFT > +0.005. ctrl_val baseline 6.4306 (Tiny1M3MConfig,
`LEADERBOARD.md` row 14) — interpreted against the in-session ctrl
run to avoid cross-session drift. Seed 42 only.
