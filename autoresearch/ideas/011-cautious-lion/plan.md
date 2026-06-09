# Plan — 011 Cautious Lion

## Flag

Two new flags in `configs/llm_config.py` (default OFF, read via `getattr` in
`training/trainer.py`):

- `use_lion: bool = False` — replaces Muon on the 2-D non-embedding,
  non-norm routing slot. Default OFF → Muon path bit-identical.
- `use_cautious_lion: bool = False` — applies the Liang et al. (2024)
  sign-mask + `1 / mask.mean().clamp(min=0.1)` rescale to the Lion sign
  update. Default OFF → bare Lion, bit-identical to Chen et al. (2023).
  Only fires when `use_lion=True`; gated by the trainer.
- Supporting HP: `lion_lr: float = 3e-4` (Chen et al.'s default at much
  larger scale — pinned, do not sweep at tiny1m3m), `lion_beta1: float = 0.9`,
  `lion_beta2: float = 0.98`.

## Change

Files touched:

1. **`optimizers/lion.py` (new, ~80 LoC)** — `Lion(Optimizer)` implementing
   Chen et al. (2023) sign-update. `cautious: bool = False` flag adds the
   Liang et al. (2024) mask after `update = sign(c)`, then
   `update = update * mask / mask.mean().clamp(min=0.1)`. Mirrors
   `optimizers/muon.py` style (torch.optim.Optimizer subclass, fp32
   momentum buffer, in-place updates with `torch.no_grad`). Bit-identical
   to vanilla Lion when `cautious=False`.

2. **`optimizers/__init__.py`** — `from .lion import Lion` + `__all__`.

3. **`training/trainer.py:_setup_optimizers`** — adds a Lion branch to
   `setup_muon_optimizer`:
   - Routing: when `use_lion=True`, the 2-D non-embed, non-norm slot
     (currently going to Muon at `trainer.py:152`) goes to `lion_params`
     instead. 1-D / embedding / head stay on AdamW. Mirrors Muon's
     2-D / 1-D split.
   - Mutex: when `use_lion=True`, the `Muon` optimizer is **not**
     instantiated (sets `muon_optimizer = None`); the return list
     becomes `[lion_optimizer, adamw_optimizer]` (plus SOAP if
     `use_soap=True`). Default OFF → `muon_optimizer = Muon(...)` runs
     unchanged.
   - Mutual-exclusion note: `use_lion=True` ⇒ no Muon instance. The
     trainer asserts the param bucket is non-empty (default for tiny1m3m
     since most 2-D weight matrices satisfy `is_muon_candidate`).

4. **`configs/llm_config.py`** — adds `use_lion`, `use_cautious_lion`,
   `lion_lr`, `lion_beta1`, `lion_beta2` fields. Adds two recipe
   dataclasses:
   - `Tiny1M3MLionConfig(Tiny1M3MConfig)` — `use_lion=True`. The bare-
     Lion prerequisite ctrl.
   - `Tiny1M3MCautiousLionConfig(Tiny1M3MLionConfig)` — adds
     `use_cautious_lion=True`. The treatment.

## Control

Two-run protocol (idea §Wiring — option (ii)):

| Row | Config | Purpose |
|---|---|---|
| 1 (ctrl) | `Tiny1M3MLionConfig` | Bare-Lion baseline. Required prerequisite. |
| 2 (treatment) | `Tiny1M3MCautiousLionConfig` | Cautious-Lion, Δ vs row 1. |

Tier: `tiny1m3m` (0.94M params · 3M tokens · ~750 steps). Seed: **42**
(one seed only — pinned; see `feedback-one-seed-only.md`). The Δ-vs-Muon-
AdamW ctrl is **not** the pass criterion — it is a secondary number for
context only. If bare-Lion itself is far worse than Muon-AdamW, that is
not drift — it is the Lion ctrl, which is the right baseline for the
hypothesis ("Cautious sign-mask helps Lion's sign-update").

## Cost

- Params: **0** delta. Lion and Muon are both 0-state-parameter
  optimizers (no learnable optimizer state; only the per-param momentum
  buffer, which both hold).
- FLOPs: small reduction. Muon runs a 5-step polar-express
  orthogonalization (ns_steps=5) every step. Lion is one `sign()`
  call plus (when cautious) one mask multiply. The cautious mask
  branch costs a `(update * g > 0)` compare + a `mask.mean()` per
  tensor — negligible vs the polar-express cost.
- Memory: identical (one fp32 momentum buffer per param on both paths).

## Run

```
# Prerequisite ctrl (run first; required for the A/B to be interpretable)
config_class=Tiny1M3MLionConfig seed=42

# Treatment
config_class=Tiny1M3MCautiousLionConfig seed=42
```

Tier: `tiny1m3m` (~750 steps · ~30 min on RTX 3060). Seed: **42**.

**Pass/fail bar** (copied verbatim from `idea.md` §Hypothesis / §Wiring):

- **PASS** ≤ −0.015 vs **bare-Lion** ctrl.
- **NULL / INCONCLUSIVE** = |Δ| < 0.01.
- **DRIFT** > +0.01.

Sub-noise effects are inconclusive, not wins. A sub-noise Cautious-Lion
result is itself informative ("Cautious doesn't fire at tiny1m3m on a
sign-update optimizer") and is logged as NULL.

### Self-check (per code-implementer §5)

- **Flag OFF reproduces the control**: `use_lion=False` (default) ⇒
  `lion_optimizer = None`, `muon_optimizer = Muon(...)`, optimizers list
  is `[Muon, AdamW]` — identical to pre-PR. `use_cautious_lion=False`
  with `use_lion=True` ⇒ bare Lion step (mask code never executes,
  guarded by `if cautious:`). Verified by direct comparison of
  `Lion(..., cautious=False).step()` against a hand-rolled Lion
  implementation: `torch.allclose(atol=1e-5)` for a 20-step smoke
  test. (Mask kicks in only when momentum EMA and current gradient
  disagree on sign — at step 0, m=0 and c=(1-β1)·g, so sign(c)=sign(g)
  and the mask is all-ones by construction; the cautious behavior
  is a multi-step effect.)
- **Treatment path is exercised**: with
  `use_lion=True, use_cautious_lion=True`, the trainer instantiates
  `Lion(lion_params, cautious=True)`; with `use_lion=True,
  use_cautious_lion=False`, it instantiates bare `Lion(lion_params)`.
  Verified via the smoke test: optimizer count is 2 in both cases;
  opt types are `[Lion, AdamW]`.
- **plan.md pass/fail bar matches idea.md**: identical (PASS ≤ −0.015,
  NULL |Δ| < 0.01, DRIFT > +0.01, two-run protocol, Δ vs bare-Lion
  not vs Muon-AdamW).
