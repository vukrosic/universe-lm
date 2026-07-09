# Experiment design — how to pick a lever that can actually beat the champion

> Companion to [`RUN-CONTRACT.md`](RUN-CONTRACT.md) (the *mechanical* shape of an
> arq + run.json) and [`PROMOTION.md`](PROMOTION.md) (how a winner is crowned).
> This file is the *strategy*: which levers are worth a GPU slot at all. Written
> 2026-06-15 from the 208–216 batch against the **alibi champion (val 6.2403)**.

## RULE 0 — NOVEL ARCHITECTURES ONLY. NO HYPERPARAMETER SEARCH. (operator, 2026-06-17)
> **"dont do basic hyperparam search, you must try novel architectures."**
>
> A valid lever changes the **forward pass or the objective** — a *mechanism*:
> attention variants, positional schemes, normalization/residual structure,
> FFN/mixing structure, loss/objective structure, token-mixing. This is the
> brief's mandate ("identity-init MECHANISMS ... without hyperparameter tuning").
>
> **NEVER queue a sweep of a scalar training/optimizer knob:** learning rate
> (`muon_lr`/`adamw_lr` or any ×mult), LR ratio, `schedule_type`/warmup,
> `weight_decay`, `muon_momentum`, `batch_size`/`gradient_accumulation_steps`,
> `embedding_scale` or any init-magnitude. These are hyperparameter search, not
> mechanisms — out of scope regardless of how they screen.
>
> *History:* ideas 303–328 drifted into exactly this HP search (LR/wd/momentum/
> batch/schedule/ratio) and were called out. Do not repeat. If the only ideas you
> can think of are knob-tweaks, that's a signal to go read recent architecture
> papers, not to file a knob-tweak.

## The budget is the constraint — internalize it
The tier is **tiny1m3m: 0.94M params, ~92 update steps, seed 42, 3M tokens.**
Everything below follows from "92 steps." A lever only matters if it can express
*and learn* its effect inside 92 optimizer steps. This is unusual — most
architecture intuitions assume convergence. Here, **time-to-signal beats
asymptotic quality.**

## Lesson 1 — zero-init multi-parameter levers WASH OUT
A lever whose new parameters start at zero/identity and must *grow* a large
matrix to take effect cannot move in 92 steps. It reverts to the champion almost
exactly.

- **211-SwiGLU** (zero-init gate, whole matrix) → NULL **Δ0.0000** (literally the
  champion).
- 208 value-residual, 209 canon-conv, 210 qk-layernorm — gated/zero-init → all NULL.
- Contrast: **alibi itself won (+0.18 over base)** because it is **48 params** of
  high-leverage signal (one slope per head) that grow fast.

**Takeaway:** prefer levers that are **step-0 active** (change the forward from
the first step) or **few-parameter / high-leverage** (a single scalar learns in a
handful of steps even from identity init — see 216-logit-scale, +1 param, moved
the right way in the probe).

## Lesson 2 — you can't beat a big structural win with a small bolt-on
alibi is a *large* structural win on the **positional axis**. Stacking a small
orthogonal lever on top (208/209/210/211) keeps washing out because the marginal
effect is below the 0.04 noise band. Two strategies actually have EV:

1. **Step-0-active orthogonal STACK** — a different axis from the champion, active
   from step 0. Keep alibi, add the lever. Examples queued: 213 gated-attn
   (output-channel gate), 214 ssmax (length-scaled softmax temperature), 216
   logit-scale (lm-head temperature). Must be a *different* axis than the
   champion (alibi = additive positional bias on pre-softmax scores; don't stack
   another thing on that same axis).
2. **More-expressive CHALLENGER on the SAME axis (replacement, not stack)** — a
   mechanism that *subsumes* the champion's hypothesis class. To beat alibi's
   one-slope-per-head distance prior, run (instead of alibi) a richer positional
   mechanism: 212 T5-RPE (per-head bucketed bias), 215 CoPE (content-dependent
   positions). These can win where bolt-ons can't, but carry convergence risk
   (more params to learn in 92 steps).

## The local probe — run BEFORE queuing (cheap, on CPU)
Never spend a GPU slot on a lever you haven't probed. The recipe (one script, a
few seconds per candidate):

```python
# 1) build the candidate as a @dataclass subclass of the champion config
@dataclass
class C(Tiny1M3MAlibiConfig):   # or Tiny1M3MConfig for a positional challenger
    use_<flag>: bool = True
m = MinimalLLM(C())             # (a) BUILDS? a flag not threaded through crashes here

# 2) STEP-0 ACTIVE? max-abs logit diff vs the champion at init
#    > 0  => active from step 0 (good).   ~0 => zero-init; likely washes (Lesson 1).

# 3) DIRECTION sanity: ~15 AdamW steps on a random batch, compare final loss to
#    the champion's. Lower => promising; much higher => the lever hurts.
```

Decision: queue a candidate only if it **builds**, is **step-0 active (or 1–few
params)**, and is **not clearly worse** in the 15-step probe. The probe uses
random data, so treat ±0.005 as noise — its real job is killing dead flags and
catastrophic levers, not ranking close calls.

From the 208–216 sweep this killed, before any GPU time:
- **Dead / unwired or zero-init wash** (Δ0.0000 in probe): diff-attn, head-gain,
  qk-rms-scaling, softpick, attn-sink, FIRE-pe (flag appears unwired), high
  logit-softcap (cap too high to bind).
- **Wrong direction** (clearly worse): post-norm (+0.28), sub-ln (+0.02),
  embed-sqrt-d (+0.036).
- **Survivors queued:** gated-attn (Δ−0.004), ssmax (active), logit-scale
  (Δ−0.002), + the two positional challengers T5-RPE, CoPE.

## Mechanical gotchas (also in RUN-CONTRACT.md — repeated because they bite)
- **Every `_arq_*.py` MUST define a top-level class named `C`.** The daemon's
  build-smoke does `getattr(mod, "C")` then `MinimalLLM(C())`. A bare
  `--config_class some.dotted.Path` with no top-level `C` → `SMOKE_FAIL` →
  `needs-recode`. (Hit on 212; fix = wrap the config in `@dataclass class C(Parent): pass`.)
- **Use `@dataclass`** on the subclass, or a field override silently doesn't take
  on the instance (the `_arq_161-dyt-temp.py` dataclass-inheritance pitfall).
- **Pre-validate locally** with the daemon's own smoke before queuing:
  `PYTHONPATH=. python3 voidspark/tools/autoresearch/_box_smoke.py _arq_NNN.py`
  → expect `SMOKE_OK`.
- **New flags need a config field, not a CLI arg** — `train_llm.py` argparse is an
  allowlist and silently ignores unknown flags. The `C` subclass is the only
  reliable toggle.
- **Inline config in the arq** (don't edit `configs/llm_config.py`) when the
  autopilot may be concurrently editing it — avoids mid-edit import collisions.

## Quick checklist before flipping an idea to `needs-run`
- [ ] `idea.md` frontmatter: `status: needs-run`, `author`, `transfer-risk`.
- [ ] `run.json`: `name`, `arq_file` (relative to repo root), `job_timeout`.
- [ ] `_arq_<idea>.py`: top-level `@dataclass class C(...)`, seed 42, `__main__.C`.
- [ ] Local: builds, **step-0 active or few-param**, probe not-worse, `SMOKE_OK`.
- [ ] Axis is orthogonal to the champion (stack) OR a superset of it (challenger).
- [ ] One lever per experiment (so the verdict is interpretable).
