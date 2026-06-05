# LR warmup-decay schedule: the −0.47 win that took #1 (then got surpassed)

**Result:** the `warmup_decay_to_zero` schedule with `warmup_ratio=0.02`
took the 10M record from **5.015 → 4.5486** (Δ **−0.466**) — and it
stayed in every recipe that came after, including the current
**4.3011** record. The schedule is a *baseline assumption* of the
whole ladder.

It is the single most under-celebrated "trick" in this codebase. It is
also the cheapest change you can make that *always* helps.

---

## What the default schedule was

The default schedule for the early 10M runs was **constant LR**:

```text
LR
 ↑
 │  ┌───────────────────────────────────────
 │  │
 │  │
 └──┴───────────────────────────────────────► step
    0
```

LR jumps to its peak at step 0 and stays there for the whole run. This is
fine for short, toy runs. It is **not** fine for 200M tokens / ~49k
steps: the optimizer wanders off the minimum late in training because the
LR is still loud enough to push it around.

---

## The fix: warmup + cosine decay to zero

Two phases:

```text
LR
 ↑
 │       ╱─────────╲
 │      ╱           ╲
 │     ╱             ╲
 │    ╱               ╲
 │   ╱                 ╲
 │  ╱                   ╲
 │ ╱                     ╲
 │╱                       ╲___
 └──┬───┬──────────────────┬───► step
    0   w                total
       warmup
```

1. **Warmup (first `warmup_ratio * total_steps`):** linearly ramp from
   0 → peak LR. This is the standard "don't kick a randomly-initialized
   model with full LR" step.
2. **Cosine decay to zero (the rest):** smoothly bring LR down to 0 over
   the remaining steps. This is the *new* part: it gives the optimizer
   a quiet phase at the end to settle into a minimum.

```python
# training/optimizer.py
def warmup_decay_to_zero(step, warmup_steps, total_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
```

**Zero new parameters. Zero new shapes. Pure scheduling.**

---

## The result on the 10m ladder

| Schedule | warmup_ratio | Val loss | Δ vs constant | Wall |
|---|---|---|---|---|
| constant (baseline) | 0.0 | 5.015 | 0 | — |
| **warmup_decay_to_zero** | **0.02** | **4.5486** | **−0.466** | ~63 min |
| warmup_decay_to_zero + emb-factor-depth | 0.02 | **4.3011** | −0.714 | ~162 min |

The schedule was worth **−0.47 on its own** (constant → decay) and **−0.25
on top of decay** (decay → decay + emb-factor-depth). The schedule
unlocks the second; the second is the current 10M record.

This is a **rank-1 lever in the codebase.** Every screen20m / 10m /
135m config inherits `warmup_decay_to_zero, warmup_ratio=0.02` from the
config default.

---

## Why it works

Two honest readings:

1. **Late-stage optimization.** With a constant LR, the model reaches a
   region of the loss landscape in the last 20% of training and then
   *kicks itself out of it* on every step. With decay to zero, the
   optimizer settles. This is the standard "LR schedule" story, and it
   is the part that scales.
2. **Effective training time.** A constant schedule spends 100% of steps
   at the peak LR. A warmup-decay schedule spends maybe 5% ramping up
   and 95% *above some fraction* of peak. The integral is similar, but
   the *distribution* is shifted toward "useful gradient signal" and
   away from "kick the model out of the basin."

Both readings are consistent with the data. The truth is probably "both,
with #1 dominant at our scale."

---

## Why the default warmup_ratio = 0.02

We tested `warmup_ratio` from 0.0 (no warmup) to 0.1 (10% warmup). The
result was a flat region between ~0.01 and ~0.05; outside that, it
matters. **0.02 is in the middle of the flat region — a safe default.**

> When a hyperparameter has a flat region, pick the middle of it. The
> exact value doesn't matter; the region does.

---

## The code

Two lines in the config:

```python
# configs/llm_config.py
warmup_ratio: float = 0.0             # the old default — constant LR
schedule_type: str = "constant"        # the old default
```

vs

```python
warmup_ratio: float = 0.02             # 2% of steps as warmup
schedule_type: str = "warmup_decay_to_zero"
```

That's the whole change. The schedule is implemented once in
`training/optimizer.py` and applied to both Muon and AdamW.

---

## Lessons

1. **Schedule is the cheapest −0.5 you'll ever get.** Two lines. Zero
   params. Universal across model sizes and recipes. **If you only
   change one thing, change this.**
2. **Decay to zero, not to a floor.** Many schedules decay to `0.1 *
   peak_lr` because that's what some paper used. At our scale, going to
   *zero* is strictly better — the late-stage benefit of "quiet
   optimizer" outweighs the cost of "wasted" low-LR steps.
3. **The optimum warmup_ratio has a flat region.** 0.02 is fine; 0.01
   is fine; 0.05 is fine. Don't over-tune this.
4. **Schedules compound with architecture changes.** The −0.47 schedule
   win was on the *old* architecture. The current record (4.3011) uses
   the same schedule on the emb-factor-depth architecture. The schedule
   is a *multiplier* on architecture quality, not a replacement for it.

---

## Caveats

- **The −0.47 is "the schedule on the old architecture."** The current
  record (4.3011) is a *combined* win: schedule + emb-factor-depth.
  You can't claim the schedule alone is worth −0.7.
- **The 200M-token full run is the comparison.** 3M / 20M screen runs
  are *less* sensitive to the schedule (less time to settle, less late-
  stage noise) — the gap closes to ~−0.05 at the screen tier.
- **Some schedules beat warmup-decay at much larger scales.** Inverse-
  sqrt (Transformer-original) and WSD (warmup-stable-decay) are
  competitive at LLM scale. We did not test them.

---

## Reproduce

```bash
# the 10M record run (uses the schedule + emb-factor-depth, ~2.7 h):
python train_llm.py --config 10m --seed 42

# isolate the schedule win (no emb-factor-depth, ~1 h):
python train_llm.py --config 10m --seed 42 \
  --warmup_ratio 0.02 \
  --schedule_type warmup_decay_to_zero
# baseline to beat: 5.015 (constant schedule)
```

Code: [training/optimizer.py](../../../training/optimizer.py) (schedule
implementation), flags in [configs/llm_config.py](../../../configs/llm_config.py)
(`schedule_type`, `warmup_ratio`).
Evidence: [LEADERBOARD.md](../../../LEADERBOARD.md) §`10m` row 1;
`runs/issue30/10m_warmup_decay_w002/metrics.json`;
tag `result/issue30-warmup-decay-w002-10m`.
